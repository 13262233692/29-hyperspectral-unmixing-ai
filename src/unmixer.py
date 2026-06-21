"""
光谱解编引擎核心
整合 1D-CNN 自编码器、显存管理，提供端到端的无监督解编服务
"""
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from scipy import ndimage

from .data_loader import HSIDataLoader, HyperSpectralCube
from .autoencoder import (
    Autoencoder1D,
    MSE_SAD_Loss,
    AbundanceSparsityLoss,
)
from .memory_manager import MemoryManager, MemoryConfig, DeviceType


@dataclass
class UnmixingResult:
    """解编结果封装"""

    endmembers: np.ndarray
    abundance_maps: np.ndarray
    reconstructed_cube: Optional[np.ndarray] = None
    wavelengths: Optional[np.ndarray] = None

    loss_history: Optional[List[float]] = None
    final_loss: Optional[float] = None

    endmember_names: List[str] = field(default_factory=lambda: [
        "Endmember_1", "Endmember_2", "Endmember_3", "Endmember_4"
    ])

    lines: int = 0
    samples: int = 0
    bands: int = 0
    num_endmembers: int = 4

    def get_abundance_map(self, index: int) -> np.ndarray:
        return self.abundance_maps[index, :, :]

    def get_endmember_spectrum(self, index: int) -> np.ndarray:
        return self.endmembers[index, :]


@dataclass
class UnmixingConfig:
    """解编配置参数"""

    num_endmembers: int = 4

    learning_rate: float = 1e-3
    num_epochs: int = 200
    batch_size: Optional[int] = None

    loss_alpha: float = 0.5
    sparsity_beta: float = 0.01

    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    kernel_size: int = 3
    dropout_rate: float = 0.1
    decoder_type: str = "linear"

    normalize: bool = True
    normalize_method: str = "minmax"

    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5

    seed: int = 42

    @classmethod
    def default(cls) -> "UnmixingConfig":
        return cls()


class SpectralUnmixer:
    """光谱解编器
    
    无监督模式下，利用 1D-CNN 自编码器进行高光谱数据解编
    自动提取端元波谱和丰度分布图
    """

    def __init__(
        self,
        config: Optional[UnmixingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.config = config or UnmixingConfig.default()
        self.memory_manager = MemoryManager(memory_config)

        self.model: Optional[Autoencoder1D] = None
        self._is_trained = False

        self._set_seed()

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if self.memory_manager.is_cuda():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_model(self, in_bands: int):
        """构建自编码器模型"""
        self.model = Autoencoder1D(
            in_bands=in_bands,
            num_endmembers=self.config.num_endmembers,
            hidden_dims=self.config.hidden_dims,
            kernel_size=self.config.kernel_size,
            dropout_rate=self.config.dropout_rate,
            decoder_type=self.config.decoder_type,
        )
        self.model = self.model.to(self.memory_manager.get_device())

    def _preprocess(self, cube: HyperSpectralCube) -> HyperSpectralCube:
        """数据预处理"""
        if self.config.normalize:
            cube = HSIDataLoader.normalize(cube, method=self.config.normalize_method)
        return cube

    def unmix(
        self,
        cube: HyperSpectralCube,
        verbose: bool = True,
    ) -> UnmixingResult:
        """执行光谱解编

        Args:
            cube: 高光谱数据立方体
            verbose: 是否输出训练信息

        Returns:
            UnmixingResult 解编结果
        """
        cube = self._preprocess(cube)
        self._build_model(cube.bands)

        pixel_spectra = cube.to_tensor_shape()
        n_pixels = pixel_spectra.shape[0]

        if self.config.batch_size is None:
            sample_bytes = pixel_spectra.shape[1] * 4
            model_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            batch_size = self.memory_manager.optimize_batch_size(
                sample_bytes, model_bytes
            )
        else:
            batch_size = self.config.batch_size

        if verbose:
            print(f"输入数据形状: {cube.shape} (bands, lines, samples)")
            print(f"像素总数: {n_pixels}")
            print(f"批大小: {batch_size}")
            print(f"模型参数量: {self.model.count_parameters():,}")
            device = self.memory_manager.get_device()
            print(f"运行设备: {device}")

        loss_history = self._train(pixel_spectra, batch_size, verbose)
        endmembers, abundances = self._extract(pixel_spectra, batch_size)

        abundance_maps = np.zeros((self.config.num_endmembers, cube.lines, cube.samples), dtype=np.float32)
        for i in range(self.config.num_endmembers):
            abundance_maps[i, :, :] = abundances[:, i].reshape(cube.lines, cube.samples)

        result = UnmixingResult(
            endmembers=endmembers,
            abundance_maps=abundance_maps,
            wavelengths=cube.wavelengths.copy(),
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else None,
            lines=cube.lines,
            samples=cube.samples,
            bands=cube.bands,
            num_endmembers=self.config.num_endmembers,
        )

        self._is_trained = True
        self._assign_endmember_names(result)

        return result

    def _train(
        self,
        pixel_spectra: np.ndarray,
        batch_size: int,
        verbose: bool,
    ) -> List[float]:
        """训练自编码器"""
        device = self.memory_manager.get_device()
        self.model.train()

        tensor_spectra = torch.from_numpy(pixel_spectra).float()
        dataset = TensorDataset(tensor_spectra)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.memory_manager.is_cuda(),
            drop_last=False,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        recon_loss_fn = MSE_SAD_Loss(alpha=self.config.loss_alpha)
        sparsity_loss_fn = AbundanceSparsityLoss(beta=self.config.sparsity_beta)

        scaler = self.memory_manager.get_grad_scaler()
        use_amp = self.memory_manager.config.enable_mixed_precision and self.memory_manager.is_cuda()

        loss_history = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_data in dataloader:
                x = batch_data[0].to(device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        recon, abundances = self.model(x)
                        recon_loss = recon_loss_fn(recon, x)
                        sparse_loss = sparsity_loss_fn(abundances)
                        loss = recon_loss + sparse_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, abundances = self.model(x)
                    recon_loss = recon_loss_fn(recon, x)
                    sparse_loss = sparsity_loss_fn(abundances)
                    loss = recon_loss + sparse_loss

                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                del x, recon, abundances

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            scheduler.step()

            if avg_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"早停触发于 Epoch {epoch+1}, Best Loss: {best_loss:.6f}")
                break

            if self.memory_manager.config.auto_gc:
                self.memory_manager.empty_cache()

        return loss_history

    def _extract(
        self,
        pixel_spectra: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提取端元和丰度"""
        device = self.memory_manager.get_device()
        self.model.eval()

        n_pixels = pixel_spectra.shape[0]
        abundances_list = []

        with torch.no_grad():
            for i in range(0, n_pixels, batch_size):
                batch = pixel_spectra[i:i + batch_size]
                x = torch.from_numpy(batch).float().to(device)

                with self.memory_manager.autocast_context():
                    enc_abundances = self.model.encode(x)

                abundances_list.append(enc_abundances.cpu().numpy())
                del x, enc_abundances

        abundances = np.concatenate(abundances_list, axis=0)

        if self.config.decoder_type == "linear":
            endmembers = self.model.get_endmembers().cpu().numpy()
        else:
            endmembers = self._extract_endmembers_from_abundances(
                pixel_spectra, abundances
            )

        endmembers, abundances = self._reorder_endmembers(endmembers, abundances)

        return endmembers, abundances

    def _extract_endmembers_from_abundances(
        self,
        pixel_spectra: np.ndarray,
        abundances: np.ndarray,
    ) -> np.ndarray:
        """从丰度矩阵反解端元（最小二乘）
        
        当使用卷积解码器时，通过最小二乘从丰度和原始光谱反解端元
        确保端元具有物理意义
        """
        from scipy.linalg import lstsq

        endmembers, _, _, _ = lstsq(abundances, pixel_spectra)
        endmembers = np.clip(endmembers, 0, None)

        return endmembers

    def _reorder_endmembers(
        self,
        endmembers: np.ndarray,
        abundances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按平均丰度从高到低排序端元"""
        mean_abundances = np.mean(abundances, axis=0)
        order = np.argsort(mean_abundances)[::-1]

        endmembers = endmembers[order, :]
        abundances = abundances[:, order]

        return endmembers, abundances

    def _assign_endmember_names(self, result: UnmixingResult):
        """为端元分配默认矿物名（基于光谱特征的简单启发式）"""
        names = []
        for i in range(result.num_endmembers):
            spectrum = result.endmembers[i, :]

            if result.wavelengths is not None and len(result.wavelengths) == len(spectrum):
                names.append(f"Mineral_{i+1}")
            else:
                names.append(f"Endmember_{i+1}")

        result.endmember_names = names

    def unmix_file(
        self,
        hdr_path: str,
        raw_path: Optional[str] = None,
        verbose: bool = True,
    ) -> UnmixingResult:
        """从文件加载并解编

        Args:
            hdr_path: .hdr 文件路径
            raw_path: .raw 文件路径（可选）
            verbose: 是否输出详细信息

        Returns:
            UnmixingResult 解编结果
        """
        loader = HSIDataLoader()
        cube = loader.load(hdr_path, raw_path)
        return self.unmix(cube, verbose=verbose)

    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化，无法保存")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "in_bands": self.model.in_bands,
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.memory_manager.get_device())

        config_dict = checkpoint.get("config", {})
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        in_bands = checkpoint.get("in_bands", 200)
        self._build_model(in_bands)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._is_trained = True

    def reconstruct(self, abundances: np.ndarray) -> np.ndarray:
        """根据丰度重建高光谱数据"""
        if self.model is None:
            raise ValueError("模型未初始化")

        device = self.memory_manager.get_device()
        self.model.eval()

        with torch.no_grad():
            z = torch.from_numpy(abundances).float().to(device)
            with self.memory_manager.autocast_context():
                recon = self.model.decode(z)
            return recon.cpu().numpy()

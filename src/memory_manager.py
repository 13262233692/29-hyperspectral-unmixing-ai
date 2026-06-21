"""
显存对齐与内存优化管理模块
对高光谱张量流进行严密的显存控制，支持：
- 显存实时监控与阈值告警
- 自动批大小调节
- 分块处理策略
- 混合精度推理
- GPU/CPU 自动切换
- 内存对齐优化
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List
from enum import Enum
import gc
import time


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class MemoryConfig:
    """显存配置参数"""

    device: DeviceType = DeviceType.AUTO
    gpu_memory_fraction: float = 0.8
    max_batch_size: int = 8192
    min_batch_size: int = 64
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    memory_alignment: int = 256
    safety_margin_mb: int = 512
    auto_gc: bool = True

    max_patch_pixels: int = 100000
    patch_overlap: int = 0

    @classmethod
    def default(cls) -> "MemoryConfig":
        return cls()


class MemoryManager:
    """显存管理器
    
    提供统一的显存监控、分配和优化接口
    确保高光谱数据处理过程中显存不溢出
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.device = self._resolve_device()
        self._current_batch_size = self.config.max_batch_size

        if self.device.type == "cuda" and self.config.gpu_memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                self.config.gpu_memory_fraction, self.device
            )

    def _resolve_device(self) -> torch.device:
        """根据配置和可用硬件自动选择设备"""
        if self.config.device == DeviceType.CPU:
            return torch.device("cpu")
        elif self.config.device == DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用，但配置指定使用 CUDA")
            return torch.device("cuda:0")
        elif self.config.device == DeviceType.MPS:
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS 不可用，但配置指定使用 MPS")
            return torch.device("mps")
        else:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

    def get_device(self) -> torch.device:
        return self.device

    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def get_memory_info(self) -> Tuple[float, float, float]:
        """获取当前显存信息

        Returns:
            (已用显存 MB, 空闲显存 MB, 总显存 MB)
        """
        if not self.is_cuda():
            return (0.0, float("inf"), float("inf"))

        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 * 1024)
        free = total - allocated

        return allocated, free, total

    def get_available_memory_mb(self) -> float:
        """获取可用显存（考虑安全边际）"""
        if not self.is_cuda():
            return float("inf")

        _, free, _ = self.get_memory_info()
        return max(0, free - self.config.safety_margin_mb)

    def check_memory_sufficient(self, required_bytes: int) -> bool:
        """检查显存是否足够

        Args:
            required_bytes: 所需字节数

        Returns:
            是否有足够显存
        """
        if not self.is_cuda():
            return True

        available = self.get_available_memory_mb() * 1024 * 1024
        return required_bytes < available

    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        """分配显存对齐的张量

        Args:
            shape: 张量形状
            dtype: 数据类型
            pin_memory: 是否使用锁页内存（CPU）

        Returns:
            分配好的张量
        """
        if self.is_cuda() or self.device.type == "mps":
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self._align_tensor_memory(tensor)
            return tensor
        else:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
            return tensor

    def _align_tensor_memory(self, tensor: torch.Tensor):
        """显存对齐（对于 CUDA 通常自动对齐，此处提供接口）"""
        pass

    def to_device(self, tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """将张量移动到目标设备，考虑显存约束"""
        if tensor.device == self.device:
            return tensor

        if self.is_cuda():
            tensor_size = tensor.element_size() * tensor.numel()
            if not self.check_memory_sufficient(tensor_size):
                raise RuntimeError(
                    f"显存不足，需要 {tensor_size / (1024*1024):.1f} MB，"
                    f"可用 {self.get_available_memory_mb():.1f} MB"
                )

        return tensor.to(self.device, non_blocking=non_blocking)

    def optimize_batch_size(
        self,
        sample_bytes: int,
        model_bytes: int = 0,
    ) -> int:
        """根据可用显存自动计算最优批大小

        Args:
            sample_bytes: 单个样本的字节数
            model_bytes: 模型占用字节数

        Returns:
            推荐的批大小
        """
        if not self.is_cuda():
            return self.config.max_batch_size

        available = self.get_available_memory_mb() * 1024 * 1024
        available = max(0, available - model_bytes)

        per_sample = sample_bytes * 4
        max_batch = int(available // per_sample)

        max_batch = min(max_batch, self.config.max_batch_size)
        max_batch = max(max_batch, self.config.min_batch_size)

        self._current_batch_size = max_batch
        return max_batch

    def get_optimal_batch_size(self) -> int:
        return self._current_batch_size

    def empty_cache(self):
        """清空显存缓存"""
        if self.is_cuda():
            torch.cuda.empty_cache()
        if self.config.auto_gc:
            gc.collect()

    def tensor_bytes(self, tensor: torch.Tensor) -> int:
        """计算张量占用的字节数"""
        return tensor.element_size() * tensor.numel()

    def numpy_to_tensor(
        self,
        array: np.ndarray,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """将 numpy 数组转换为设备上的张量

        使用分块传输避免大数组一次性拷贝导致显存溢出
        """
        if dtype is None:
            dtype = torch.float32

        total_elements = array.size
        chunk_size = self._calculate_transfer_chunk_size(array.dtype.itemsize)

        if total_elements <= chunk_size or not self.is_cuda():
            tensor = torch.from_numpy(array).to(dtype)
            return self.to_device(tensor)

        flat = array.ravel()
        result = torch.empty(total_elements, dtype=dtype, device=self.device)

        for i in range(0, total_elements, chunk_size):
            end = min(i + chunk_size, total_elements)
            chunk = torch.from_numpy(flat[i:end]).to(dtype)
            result[i:end] = chunk.to(self.device)
            del chunk

        return result.reshape(array.shape)

    def _calculate_transfer_chunk_size(self, element_size: int) -> int:
        """计算传输块大小"""
        if not self.is_cuda():
            return 10**9

        available_mb = self.get_available_memory_mb()
        chunk_mb = min(available_mb * 0.3, 512)
        return int(chunk_mb * 1024 * 1024 // element_size)

    def process_in_batches(
        self,
        data: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """分批处理数据

        Args:
            data: 输入数据 (N, D)
            process_fn: 处理函数
            batch_size: 批大小，自动计算

        Returns:
            处理后的结果
        """
        n_samples = data.shape[0]

        if batch_size is None:
            sample_bytes = self.tensor_bytes(data[0:1])
            batch_size = self.optimize_batch_size(sample_bytes)

        results = []

        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            result = process_fn(batch)
            results.append(result.detach() if isinstance(result, torch.Tensor) else result)
            del batch

            if self.config.auto_gc and self.is_cuda() and i % (batch_size * 10) == 0:
                self.empty_cache()

        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        return results

    def autocast_context(self):
        """获取混合精度上下文管理器"""
        if self.config.enable_mixed_precision and self.is_cuda():
            return torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            return nullcontext()

    def get_grad_scaler(self):
        """获取梯度缩放器（用于混合精度训练）"""
        if self.config.enable_mixed_precision and self.is_cuda():
            return torch.cuda.amp.GradScaler()
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.empty_cache()

    def print_memory_stats(self):
        """打印显存统计信息"""
        if self.is_cuda():
            allocated, free, total = self.get_memory_info()
            print(f"GPU 显存 - 已用: {allocated:.1f} MB, 空闲: {free:.1f} MB, 总计: {total:.1f} MB")
            print(f"当前批大小: {self._current_batch_size}")
        else:
            print("当前运行在 CPU 模式")


class PatchProcessor:
    """高光谱图像分块处理器
    
    对于超大高光谱立方体，采用分块处理策略
    避免一次性加载整张图像到显存
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        patch_size: Optional[int] = None,
        overlap: int = 0,
    ):
        self.memory_manager = memory_manager
        self.overlap = overlap

        if patch_size is None:
            self.patch_size = self._auto_calculate_patch_size()
        else:
            self.patch_size = patch_size

    def _auto_calculate_patch_size(self) -> int:
        """自动计算块大小"""
        if not self.memory_manager.is_cuda():
            return 1024

        available = self.memory_manager.get_available_memory_mb()
        target_mb = available * 0.3
        return int(np.sqrt(target_mb * 1024 * 1024 / 4))

    def process_cube(
        self,
        cube_data: np.ndarray,
        process_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """分块处理高光谱立方体

        Args:
            cube_data: 形状 (bands, lines, samples)
            process_fn: 处理函数，输入输出形状为 (bands, lines, samples)

        Returns:
            处理后的立方体
        """
        bands, lines, samples = cube_data.shape
        result = np.zeros_like(cube_data)
        weight = np.zeros((1, lines, samples), dtype=np.float32)

        patch_h = min(self.patch_size, lines)
        patch_w = min(self.patch_size, samples)

        for i in range(0, lines, patch_h - self.overlap):
            for j in range(0, samples, patch_w - self.overlap):
                i_end = min(i + patch_h, lines)
                j_end = min(j + patch_w, samples)

                patch = cube_data[:, i:i_end, j:j_end]
                processed = process_fn(patch)

                result[:, i:i_end, j:j_end] += processed
                weight[:, i:i_end, j:j_end] += 1.0

        weight[weight == 0] = 1.0
        result = result / weight

        return result

    def process_pixels(
        self,
        pixel_spectra: np.ndarray,
        process_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """分块处理像素光谱

        Args:
            pixel_spectra: 形状 (n_pixels, bands)
            process_fn: 处理函数

        Returns:
            处理后的结果
        """
        return self.memory_manager.process_in_batches(
            torch.from_numpy(pixel_spectra),
            lambda x: process_fn(x.cpu().numpy()),
        ).cpu().numpy()


class MemoryProfiler:
    """显存分析器
    
    用于性能调优，记录各阶段显存使用情况
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.checkpoints = []

    def checkpoint(self, label: str):
        """记录显存检查点"""
        if self.memory_manager.is_cuda():
            allocated, free, total = self.memory_manager.get_memory_info()
            self.checkpoints.append({
                "label": label,
                "time": time.time(),
                "allocated_mb": allocated,
                "free_mb": free,
                "total_mb": total,
            })

    def get_report(self) -> str:
        """生成显存分析报告"""
        if not self.checkpoints:
            return "无检查点数据"

        lines = ["=== 显存分析报告 ==="]
        prev_alloc = None

        for cp in self.checkpoints:
            delta_str = ""
            if prev_alloc is not None:
                delta = cp["allocated_mb"] - prev_alloc
                delta_str = f" (Δ {delta:+.1f} MB)"

            lines.append(
                f"[{cp['label']}] 已用: {cp['allocated_mb']:.1f} MB{delta_str}"
            )
            prev_alloc = cp["allocated_mb"]

        return "\n".join(lines)

    def reset(self):
        self.checkpoints = []

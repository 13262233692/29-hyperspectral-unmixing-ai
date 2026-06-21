"""
高光谱数据加载模块
支持 ENVI 格式 .hdr/.raw 文件的读取与预处理
"""
import os
import re
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class HyperSpectralCube:
    """高光谱数据立方体封装
    
    存储三维高光谱数据及其元信息，形状约定为 (bands, lines, samples)
    即 [波段数, 空间行, 空间列]，与 ENVI BIL/BIP/BSQ 格式统一转换
    """
    data: np.ndarray
    wavelengths: np.ndarray
    lines: int
    samples: int
    bands: int
    interleave: str = "bsq"
    dtype: str = "float32"
    byte_order: int = 0
    header_offset: int = 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def to_tensor_shape(self) -> np.ndarray:
        """转换为模型输入形状 (pixels, bands)
        
        将空间维度展平，每行是一个光谱向量
        """
        bands, lines, samples = self.shape
        return self.data.reshape(bands, lines * samples).T

    def from_tensor_shape(self, tensor_data: np.ndarray) -> np.ndarray:
        """从 (pixels, bands) 恢复为 (bands, lines, samples)"""
        bands, lines, samples = self.shape
        return tensor_data.T.reshape(bands, lines, samples)


class HSIDataLoader:
    """ENVI 格式高光谱数据加载器
    
    支持读取 ENVI 标准格式的 .hdr 头文件和 .raw 二进制数据文件
    支持 BSQ/BIL/BIP 三种存储交错格式
    """

    DTYPE_MAP = {
        "1": ("uint8", "B"),
        "2": ("int16", "h"),
        "3": ("int32", "i"),
        "4": ("float32", "f"),
        "5": ("float64", "d"),
        "6": ("complex64", "f"),
        "9": ("complex128", "d"),
        "12": ("uint16", "H"),
        "13": ("uint32", "I"),
        "14": ("int64", "q"),
        "15": ("uint64", "Q"),
    }

    def __init__(self, byte_order: Optional[int] = None):
        self.byte_order = byte_order

    def load(self, hdr_path: str, raw_path: Optional[str] = None) -> HyperSpectralCube:
        """加载高光谱数据

        Args:
            hdr_path: .hdr 头文件路径
            raw_path: .raw 数据文件路径，若为空则根据 hdr 路径推断

        Returns:
            HyperSpectralCube 对象
        """
        if not os.path.exists(hdr_path):
            raise FileNotFoundError(f"头文件不存在: {hdr_path}")

        if raw_path is None:
            base, _ = os.path.splitext(hdr_path)
            for ext in [".raw", ".dat", ".img"]:
                candidate = base + ext
                if os.path.exists(candidate):
                    raw_path = candidate
                    break
            if raw_path is None:
                raise FileNotFoundError(f"未找到对应的数据文件: {base}.raw/.dat/.img")

        header = self._parse_header(hdr_path)
        data = self._read_binary(raw_path, header)
        data = self._convert_to_bsq(data, header)

        wavelengths = header.get("wavelength", np.arange(header["bands"], dtype=np.float32))

        return HyperSpectralCube(
            data=data.astype(np.float32),
            wavelengths=np.array(wavelengths, dtype=np.float32),
            lines=header["lines"],
            samples=header["samples"],
            bands=header["bands"],
            interleave=header.get("interleave", "bsq").lower(),
            dtype=header.get("data type", "float32"),
            byte_order=header.get("byte order", 0),
            header_offset=header.get("header offset", 0),
        )

    def _parse_header(self, hdr_path: str) -> dict:
        """解析 ENVI .hdr 头文件"""
        header = {}
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith(";"):
                i += 1
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip()

                if value.startswith("{"):
                    multi_line_value = value
                    while "}" not in multi_line_value and i + 1 < len(lines):
                        i += 1
                        multi_line_value += " " + lines[i].strip()
                    value = multi_line_value.strip("{}").strip()
                    items = [v.strip() for v in value.split(",") if v.strip()]
                    header[key] = items
                else:
                    header[key] = value
            i += 1

        header["lines"] = int(header.get("lines", 0))
        header["samples"] = int(header.get("samples", 0))
        header["bands"] = int(header.get("bands", 0))
        header["header offset"] = int(header.get("header offset", 0))
        header["byte order"] = int(header.get("byte order", 0))

        if "wavelength" in header:
            try:
                header["wavelength"] = [float(w) for w in header["wavelength"]]
            except (ValueError, TypeError):
                pass

        return header

    def _read_binary(self, raw_path: str, header: dict) -> np.ndarray:
        """读取二进制数据文件"""
        samples = header["samples"]
        lines = header["lines"]
        bands = header["bands"]
        offset = header["header offset"]
        byte_order = header["byte order"]
        if self.byte_order is not None:
            byte_order = self.byte_order

        data_type = str(header.get("data type", "3"))
        dtype_str, fmt_char = self.DTYPE_MAP.get(data_type, ("float32", "f"))
        dtype = np.dtype(dtype_str)
        if byte_order == 1:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")

        total_elements = samples * lines * bands
        expected_size = total_elements * dtype.itemsize

        with open(raw_path, "rb") as f:
            f.seek(offset)
            raw_data = f.read(expected_size)

        data = np.frombuffer(raw_data, dtype=dtype).copy()
        data = data.astype(np.float32)

        interleave = header.get("interleave", "bsq").lower()

        if interleave == "bsq":
            data = data.reshape(bands, lines, samples)
        elif interleave == "bil":
            data = data.reshape(lines, bands, samples)
        elif interleave == "bip":
            data = data.reshape(lines, samples, bands)
        else:
            raise ValueError(f"不支持的交错格式: {interleave}")

        return data

    def _convert_to_bsq(self, data: np.ndarray, header: dict) -> np.ndarray:
        """统一转换为 BSQ 格式 (bands, lines, samples)"""
        interleave = header.get("interleave", "bsq").lower()
        bands = header["bands"]
        lines = header["lines"]
        samples = header["samples"]

        if interleave == "bsq":
            return data
        elif interleave == "bil":
            return np.transpose(data, (1, 0, 2))
        elif interleave == "bip":
            return np.transpose(data, (2, 0, 1))
        else:
            raise ValueError(f"不支持的交错格式: {interleave}")

    @staticmethod
    def normalize(cube: HyperSpectralCube, method: str = "minmax") -> HyperSpectralCube:
        """光谱归一化处理

        Args:
            cube: 输入高光谱立方体
            method: 归一化方法 - minmax / mean / zscore

        Returns:
            归一化后的高光谱立方体
        """
        data = cube.data.copy()

        if method == "minmax":
            bands_min = np.min(data, axis=(1, 2), keepdims=True)
            bands_max = np.max(data, axis=(1, 2), keepdims=True)
            denom = bands_max - bands_min
            denom[denom == 0] = 1.0
            data = (data - bands_min) / denom
        elif method == "mean":
            bands_mean = np.mean(data, axis=(1, 2), keepdims=True)
            data = data / bands_mean
        elif method == "zscore":
            bands_mean = np.mean(data, axis=(1, 2), keepdims=True)
            bands_std = np.std(data, axis=(1, 2), keepdims=True)
            bands_std[bands_std == 0] = 1.0
            data = (data - bands_mean) / bands_std
        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        cube.data = data
        return cube

    @staticmethod
    def remove_bad_bands(cube: HyperSpectralCube, bad_bands: Optional[List[int]] = None,
                        threshold: float = 0.0) -> HyperSpectralCube:
        """去除坏波段

        Args:
            cube: 输入高光谱立方体
            bad_bands: 指定的坏波段索引列表
            threshold: 标准差阈值，低于此阈值的波段自动去除

        Returns:
            处理后的高光谱立方体
        """
        bands_to_keep = list(range(cube.bands))

        if bad_bands:
            bands_to_keep = [b for b in bands_to_keep if b not in bad_bands]

        if threshold > 0:
            stds = np.std(cube.data, axis=(1, 2))
            bands_to_keep = [b for b in bands_to_keep if stds[b] >= threshold]

        cube.data = cube.data[bands_to_keep, :, :]
        cube.bands = len(bands_to_keep)
        cube.wavelengths = cube.wavelengths[bands_to_keep]

        return cube

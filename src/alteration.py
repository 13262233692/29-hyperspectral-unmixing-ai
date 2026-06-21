"""
海底烃类渗漏蚀变特征谱段分析模块

算法原理：
- 1900nm 吸收峰：蒙脱石、伊利石等粘土矿物的 H2O/OH 伸缩+弯曲合频吸收
- 2300nm 吸收峰：碳酸盐矿物（方解石）的 CO3^2- 基团合频吸收
- 2300/1900 深度比值（CAR = Carbonate-Alteration Ratio）：
    - 烃类渗漏 → 碳酸盐蚀变（方解石沉淀） → 2300nm 深度增大
    - 同时粘土矿物蚀变影响 1900nm
    - 高比值区域对应潜在天然气水合物/油气渗漏带
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d


@dataclass
class AbsorptionFeature:
    """单波段吸收特征"""

    wavelength: float
    band_index: int
    depth: float
    area: float
    width: float
    asymmetry: float
    interpolated: bool = False


@dataclass
class AlterationIndex:
    """烃类蚀变指数计算结果"""

    wavelengths: np.ndarray
    endmember_spectra: np.ndarray
    num_endmembers: int

    depth_1900: np.ndarray
    depth_2300: np.ndarray
    ratio_2300_1900: np.ndarray

    features_1900: List[AbsorptionFeature]
    features_2300: List[AbsorptionFeature]

    enrichment_mask: Optional[np.ndarray] = None
    enrichment_score: Optional[np.ndarray] = None
    enrichment_threshold: float = 0.0
    enrichment_fraction: float = 0.0

    endmember_carbonate_rich: Optional[List[int]] = None

    _mask_uint8: Optional[np.ndarray] = None

    def get_mask_uint8(self) -> np.ndarray:
        if self._mask_uint8 is None:
            if self.enrichment_mask is not None:
                self._mask_uint8 = (self.enrichment_mask * 255).astype(np.uint8)
            else:
                self._mask_uint8 = np.zeros((1, 1), dtype=np.uint8)
        return self._mask_uint8


class HydrocarbonAlterationAnalyzer:
    """海底烃类渗漏蚀变自动化分析器

    特征波段：
    - 1900nm ± 50nm：粘土矿物羟基/水吸收窗口
    - 2300nm ± 50nm：碳酸盐 CO3 吸收窗口
    - 连续统去除（Continuum Removal）计算吸收深度
    """

    WINDOW_1900_CENTER = 1900.0
    WINDOW_1900_HALF = 80.0
    WINDOW_2300_CENTER = 2300.0
    WINDOW_2300_HALF = 100.0

    def __init__(
        self,
        window_1900_half: float = 80.0,
        window_2300_half: float = 100.0,
        savgol_window: int = 7,
        savgol_polyorder: int = 3,
        enrichment_percentile: float = 75.0,
        min_enrichment_ratio: float = 0.4,
    ):
        self.WINDOW_1900_HALF = window_1900_half
        self.WINDOW_2300_HALF = window_2300_half
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.enrichment_percentile = enrichment_percentile
        self.min_enrichment_ratio = min_enrichment_ratio

    def _smooth(self, spectrum: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 平滑降噪"""
        n = len(spectrum)
        if n < self.savgol_window:
            return spectrum
        win = min(self.savgol_window, n if n % 2 == 1 else n - 1)
        if win < 3:
            return spectrum
        poly = min(self.savgol_polyorder, win - 1)
        try:
            return savgol_filter(spectrum, win, poly)
        except Exception:
            return spectrum

    def _find_nearest_band(
        self, wavelengths: np.ndarray, target: float, half_width: float
    ) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
        """在波长数组中定位目标波段窗口"""
        wl = np.asarray(wavelengths, dtype=np.float64)
        idx_closest = int(np.argmin(np.abs(wl - target)))

        lo = target - half_width
        hi = target + half_width
        window_mask = (wl >= lo) & (wl <= hi)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) < 3:
            lo_idx = max(0, idx_closest - 5)
            hi_idx = min(len(wl), idx_closest + 6)
            window_indices = np.arange(lo_idx, hi_idx)

        start_idx = int(window_indices[0])
        end_idx = int(window_indices[-1])
        return idx_closest, wl[window_indices], window_indices, start_idx, end_idx

    def _continuum_removal(
        self, wavelengths: np.ndarray, spectrum: np.ndarray
    ):
        """连续统去除（凸包络归一化）

        使用分段线性凸包（上端连线）方法，无递归，收敛速度快 O(n)
        输出：CR（值 ∈ [0, 1]），upper_envelope（上端包络线）
        """
        n = len(wavelengths)
        if n < 3:
            ones = np.ones_like(spectrum)
            env = spectrum.copy() if n > 0 else np.array([])
            return ones, env

        hull_indices = [0]
        i = 0
        while i < n - 1:
            best_j = i + 1
            best_slope = -np.inf
            for j in range(i + 1, n):
                if wavelengths[j] == wavelengths[i]:
                    continue
                slope = (spectrum[j] - spectrum[i]) / (wavelengths[j] - wavelengths[i])
                if slope > best_slope:
                    best_slope = slope
                    best_j = j
            hull_indices.append(best_j)
            i = best_j

        hull_indices = list(dict.fromkeys(hull_indices))
        if hull_indices[-1] != n - 1:
            hull_indices.append(n - 1)

        upper_envelope = np.zeros(n)
        for k in range(len(hull_indices) - 1):
            i0 = hull_indices[k]
            i1 = hull_indices[k + 1]
            x0, y0 = wavelengths[i0], spectrum[i0]
            x1, y1 = wavelengths[i1], spectrum[i1]
            dx = x1 - x0
            if abs(dx) < 1e-12:
                upper_envelope[i0:i1 + 1] = max(y0, y1)
                continue
            slope = (y1 - y0) / dx
            intercept = y0 - slope * x0
            upper_envelope[i0:i1 + 1] = slope * wavelengths[i0:i1 + 1] + intercept

        upper_envelope[0] = spectrum[0]
        upper_envelope[-1] = spectrum[-1]

        denom = np.where(np.abs(upper_envelope) < 1e-10, 1e-10, upper_envelope)
        cr = spectrum / denom
        cr = np.clip(cr, 0.0, 1.0 + 1e-6)
        return cr, upper_envelope

    def _band_depth(
        self,
        wavelengths: np.ndarray,
        spectrum: np.ndarray,
        center: float,
        half_width: float,
    ) -> Tuple[float, AbsorptionFeature, np.ndarray, np.ndarray, np.ndarray]:
        """计算指定中心波长处的吸收深度（Band Depth）

        Band Depth = 1 - CR(center)，其中 CR 为连续统去除后值
        """
        idx_center, wl_win, idx_win, start_i, end_i = self._find_nearest_band(
            wavelengths, center, half_width
        )

        spec_win = spectrum[idx_win]
        if len(wl_win) < 3:
            cr_win = np.ones_like(spec_win)
            envelope = spec_win.copy()
        else:
            cr_win, envelope = self._continuum_removal(wl_win, spec_win)

        local_idx = int(np.argmin(np.abs(wl_win - center)))
        cr_at_center = float(cr_win[local_idx]) if len(cr_win) > 0 else 1.0
        depth = max(0.0, 1.0 - cr_at_center)

        min_cr = float(np.min(cr_win))
        max_depth = max(0.0, 1.0 - min_cr)
        min_idx_local = int(np.argmin(cr_win))
        actual_center_wl = float(wl_win[min_idx_local]) if len(wl_win) > 0 else center

        area = float(np.trapz(1.0 - cr_win, wl_win)) if len(cr_win) > 1 else 0.0

        if max_depth > 1e-6:
            half_level = 1.0 - max_depth / 2.0
            above_half = np.where(cr_win <= half_level)[0]
            if len(above_half) >= 2:
                width = float(wl_win[above_half[-1]] - wl_win[above_half[0]])
            else:
                width = half_width * 2.0
        else:
            width = half_width * 2.0

        if len(cr_win) >= 3 and max_depth > 1e-6:
            mid = len(cr_win) // 2
            left_area = float(np.trapz(1.0 - cr_win[:mid], wl_win[:mid])) if mid > 0 else 0.0
            right_area = float(np.trapz(1.0 - cr_win[mid:], wl_win[mid:]))
            total_area = left_area + right_area + 1e-12
            asymmetry = (right_area - left_area) / total_area
        else:
            asymmetry = 0.0

        feature = AbsorptionFeature(
            wavelength=actual_center_wl,
            band_index=int(idx_win[min_idx_local]) if len(idx_win) > 0 else idx_center,
            depth=max_depth,
            area=area,
            width=width,
            asymmetry=asymmetry,
            interpolated=(np.abs(actual_center_wl - center) > half_width * 0.5),
        )

        return depth, feature, cr_win, envelope, wl_win

    def analyze_endmembers(
        self,
        endmember_spectra: np.ndarray,
        wavelengths: np.ndarray,
    ) -> AlterationIndex:
        """对解编出的所有端元波谱进行烃类蚀变分析

        Args:
            endmember_spectra: 形状 (num_endmembers, num_bands)，端元反射率光谱
            wavelengths: 波长数组 (num_bands,) 单位 nm

        Returns:
            AlterationIndex 分析结果
        """
        num_em, num_bands = endmember_spectra.shape
        wl = np.asarray(wavelengths, dtype=np.float64)

        depths_1900 = np.zeros(num_em, dtype=np.float64)
        depths_2300 = np.zeros(num_em, dtype=np.float64)
        features_1900: List[AbsorptionFeature] = []
        features_2300: List[AbsorptionFeature] = []

        for i in range(num_em):
            raw_spec = endmember_spectra[i, :].astype(np.float64)
            raw_spec = np.nan_to_num(raw_spec, nan=0.0, posinf=0.0, neginf=0.0)
            spec = self._smooth(raw_spec)
            spec = np.clip(spec, 0.0, None)

            d19, feat19, _, _, _ = self._band_depth(
                wl, spec, self.WINDOW_1900_CENTER, self.WINDOW_1900_HALF
            )
            d23, feat23, _, _, _ = self._band_depth(
                wl, spec, self.WINDOW_2300_CENTER, self.WINDOW_2300_HALF
            )

            depths_1900[i] = d19
            depths_2300[i] = d23
            features_1900.append(feat19)
            features_2300.append(feat23)

        denom = depths_1900.copy()
        denom[denom < 1e-6] = 1e-6
        ratio_2300_1900 = depths_2300 / denom
        ratio_2300_1900 = np.clip(ratio_2300_1900, 0.0, 10.0)

        carbonate_rich = [
            int(i)
            for i in range(num_em)
            if ratio_2300_1900[i] >= self.min_enrichment_ratio
            and depths_2300[i] >= 0.02
        ]

        return AlterationIndex(
            wavelengths=wl,
            endmember_spectra=endmember_spectra.copy(),
            num_endmembers=num_em,
            depth_1900=depths_1900,
            depth_2300=depths_2300,
            ratio_2300_1900=ratio_2300_1900,
            features_1900=features_1900,
            features_2300=features_2300,
            endmember_carbonate_rich=carbonate_rich,
        )

    def compute_enrichment_mask(
        self,
        alteration: AlterationIndex,
        abundance_maps: np.ndarray,
    ) -> AlterationIndex:
        """基于端元丰度图 + 蚀变指数，计算油气蚀变富集区掩膜

        算法：
        富集分数 S(pixel) = Σ_i (ratio_i * abundance_i(pixel))
        掩膜 M(pixel) = S(pixel) > threshold
        threshold 取 S 的 percentile 分位数

        Args:
            alteration: analyze_endmembers 的输出
            abundance_maps: 形状 (num_endmembers, lines, samples)

        Returns:
            更新后的 AlterationIndex（含 enrichment_mask / enrichment_score）
        """
        num_em, lines, samples = abundance_maps.shape
        ratio = alteration.ratio_2300_1900.copy()

        d23 = alteration.depth_2300.copy()
        weights = ratio * d23

        if alteration.endmember_carbonate_rich:
            em_mask = np.zeros(num_em)
            em_mask[alteration.endmember_carbonate_rich] = 1.0
            weights = weights * em_mask

        weights = weights / (np.sum(np.abs(weights)) + 1e-12)

        score = np.zeros((lines, samples), dtype=np.float64)
        for i in range(num_em):
            if abs(weights[i]) > 1e-12:
                score += weights[i] * abundance_maps[i, :, :]

        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = np.clip(score, 0.0, None)

        if np.max(score) - np.min(score) > 1e-8:
            score = (score - np.min(score)) / (np.max(score) - np.min(score))

        valid = score[score > 1e-8]
        if len(valid) > 10:
            threshold = float(np.percentile(valid, self.enrichment_percentile))
        else:
            threshold = float(np.percentile(score, self.enrichment_percentile))

        if threshold < 0.05:
            threshold = 0.05

        mask = (score >= threshold).astype(np.uint8)

        enrichment_fraction = float(np.mean(mask))

        alteration.enrichment_score = score
        alteration.enrichment_mask = mask.astype(np.float32)
        alteration.enrichment_threshold = threshold
        alteration.enrichment_fraction = enrichment_fraction
        alteration._mask_uint8 = None

        return alteration

    def full_analysis(
        self,
        endmember_spectra: np.ndarray,
        wavelengths: np.ndarray,
        abundance_maps: np.ndarray,
    ) -> AlterationIndex:
        """完整分析：端元谱段分析 + 富集区掩膜"""
        alteration = self.analyze_endmembers(endmember_spectra, wavelengths)
        alteration = self.compute_enrichment_mask(alteration, abundance_maps)
        return alteration


class AlterationEventHooks:
    """蚀变分析钩子集合 - 内部事件总线

    支持的事件：
    - after_unmix: 解编完成后自动拉起蚀变分析
    - after_mask: 掩膜生成后触发可视化回调
    """

    def __init__(self):
        self._hooks: Dict[str, List] = {}
        self._register_default_hooks()

    def _register_default_hooks(self):
        self._hooks["after_unmix"] = []
        self._hooks["after_mask"] = []
        self._hooks["on_error"] = []

    def register(self, event: str, callback):
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def fire(self, event: str, *args, **kwargs):
        if event not in self._hooks:
            return []
        results = []
        for cb in self._hooks[event]:
            try:
                results.append(cb(*args, **kwargs))
            except Exception:
                pass
        return results

    def clear(self, event: Optional[str] = None):
        if event is None:
            self._register_default_hooks()
        else:
            self._hooks[event] = []

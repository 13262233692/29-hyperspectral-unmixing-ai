import sys
import os
import io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def g(x, mu, sigma, amp):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

print("=== 验证 1: 吸收峰深度计算 ===")
from src.alteration import HydrocarbonAlterationAnalyzer

np.random.seed(42)
n_bands = 200
wl = np.linspace(400, 2500, n_bands)

spectrum = 0.6 * np.ones(n_bands)
spectrum -= g(wl, 1900, 40, 0.25)
spectrum -= g(wl, 2300, 50, 0.4)
spectrum = np.clip(spectrum, 0.0, 1.0)

analyzer = HydrocarbonAlterationAnalyzer()
alt = analyzer.analyze_endmembers(np.stack([spectrum], axis=0), wl)

print(f"  d1900 = {alt.depth_1900[0]:.4f} (expect ~0.2)")
print(f"  d2300 = {alt.depth_2300[0]:.4f} (expect ~0.35)")
print(f"  ratio = {alt.ratio_2300_1900[0]:.4f} (expect >1.0)")
assert alt.depth_1900[0] > 0.05
assert alt.depth_2300[0] > 0.1
assert alt.ratio_2300_1900[0] > 1.0
print("  PASS")

print("\n=== 验证 2: 富集掩膜生成 ===")
endmembers = []
configs = [(0.15, 0.05), (0.15, 0.50), (0.20, 0.10), (0.15, 0.55)]
for d19, d23 in configs:
    em = 0.6 * np.ones(n_bands)
    em -= g(wl, 1900, 40, d19)
    em -= g(wl, 2300, 50, d23)
    em = np.clip(em, 0.05, 1.0)
    endmembers.append(em)
endmembers = np.stack(endmembers, axis=0)

lines, samples = 32, 32
yy, xx = np.mgrid[0:lines, 0:samples]
center = (lines // 2, samples // 2)
dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
gradient = 1.0 - dist / (np.max(dist) + 1e-6)

abundance_maps = np.zeros((4, lines, samples), dtype=np.float32)
abundance_maps[0] = 0.2 + 0.2 * gradient
abundance_maps[1] = 0.6 * gradient
abundance_maps[2] = 0.2 + 0.1 * np.random.rand(lines, samples)
abundance_maps[3] = 0.5 * gradient + 0.1 * np.random.rand(lines, samples)
s = np.sum(abundance_maps, axis=0, keepdims=True)
abundance_maps = abundance_maps / (s + 1e-8)

alt = analyzer.full_analysis(endmembers, wl, abundance_maps)
mu8 = alt.get_mask_uint8()
print(f"  mask shape: {alt.enrichment_mask.shape}")
print(f"  fraction: {alt.enrichment_fraction:.2%}")
print(f"  threshold: {alt.enrichment_threshold:.4f}")
print(f"  carbonate-rich endmembers: {alt.endmember_carbonate_rich}")
print(f"  mask dtype uint8 max: {np.max(mu8)}")
assert alt.enrichment_mask is not None
assert 0.0 < alt.enrichment_fraction < 1.0
assert np.max(mu8) in (0, 255)
print("  PASS")

print("\n=== 验证 3: 事件钩子 ===")
from src import SpectralUnmixer, UnmixingConfig
cfg = UnmixingConfig.default()
unmixer = SpectralUnmixer(config=cfg)
events = []
def on_after_unmix(*a, **kw): events.append("after_unmix")
def on_after_mask(*a, **kw): events.append("after_mask")
unmixer.hooks.register("after_unmix", on_after_unmix)
unmixer.hooks.register("after_mask", on_after_mask)
unmixer.hooks.fire("after_unmix", None, unmixer, None)
unmixer.hooks.fire("after_mask", None, None, unmixer, None)
print(f"  events fired: {events}")
assert "after_unmix" in events
assert "after_mask" in events
print("  PASS")

print("\n=== 验证 4: SpectralUnmixer 配置透传 + alteration 分析器初始化 ===")
cfg2 = UnmixingConfig(enable_alteration_analysis=True, alteration_enrichment_percentile=70)
unmixer2 = SpectralUnmixer(config=cfg2)
assert unmixer2.alteration_analyzer is not None
assert unmixer2.alteration_analyzer.enrichment_percentile == 70.0
print(f"  analyzer enabled, percentile={unmixer2.alteration_analyzer.enrichment_percentile}")
cfg3 = UnmixingConfig(enable_alteration_analysis=False)
unmixer3 = SpectralUnmixer(config=cfg3)
assert unmixer3.alteration_analyzer is None
print("  analyzer disabled correctly")
print("  PASS")

print("\n=== 验证 5: UnmixingResult 增加 alteration 字段与方法 ===")
from src import UnmixingResult
r = UnmixingResult(
    endmembers=np.zeros((4, 100)),
    abundance_maps=np.zeros((4, 20, 20)),
)
assert r.has_alteration() is False
assert r.get_enrichment_mask() is None
r.alteration = alt
assert r.has_alteration() is True
assert r.get_enrichment_mask() is not None
assert r.get_enrichment_score() is not None
print("  has_alteration / get_enrichment_mask / get_enrichment_score OK")
print("  PASS")

print("\n=== 验证 6: FastAPI schemas 新增字段 ===")
from app.schemas import AlterationResponse, AlterationEndmemberStats, UnmixingResponse
ar = AlterationResponse(
    enabled=True, success=True, enrichment_threshold=0.3, enrichment_fraction=0.25,
    endmember_stats=[
        AlterationEndmemberStats(
            endmember_index=0, endmember_name="A", depth_1900=0.1,
            depth_2300=0.3, ratio_2300_1900=3.0, is_carbonate_rich=True
        )
    ],
    carbonate_rich_endmembers=[0],
    enrichment_mask=[[0,255],[255,0]],
    enrichment_score=[[0.0,0.9],[0.8,0.1]],
    enrichment_mask_shape=[2, 2],
)
ur = UnmixingResponse(success=True, alteration=ar)
print(f"  AlterationResponse model_validate OK")
print(f"  UnmixingResponse with alteration OK")
print("  PASS")

print("\n=== 验证 7: API endpoints 表单新参数导入 ===")
from app.api.endpoints import unmix_hyperspectral
import inspect
sig = inspect.signature(unmix_hyperspectral)
param_names = list(sig.parameters.keys())
assert "enable_alteration_analysis" in param_names
assert "alteration_enrichment_percentile" in param_names
assert "alteration_min_ratio" in param_names
print(f"  new Form params found: enable_alteration_analysis, alteration_enrichment_percentile, alteration_min_ratio")
print("  PASS")

print("\n" + "=" * 60)
print("全部 7 项快速验证通过 ✓")
print("=" * 60)

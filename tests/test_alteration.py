"""
海底烃类蚀变分析旁路 - 端到端测试套件
"""
import os
import sys
import io
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def g(x, mu, sigma, amp):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def test_absorption_band_depth():
    from src.alteration import HydrocarbonAlterationAnalyzer

    np.random.seed(42)
    n_bands = 200
    wl = np.linspace(400, 2500, n_bands)

    spectrum = 0.6 * np.ones(n_bands)
    spectrum -= g(wl, 1900, 40, 0.25)
    spectrum -= g(wl, 2300, 50, 0.4)
    spectrum += 0.01 * np.random.randn(n_bands)
    spectrum = np.clip(spectrum, 0.0, 1.0)

    analyzer = HydrocarbonAlterationAnalyzer()
    spectra = np.stack([spectrum, spectrum * 0.9 + 0.1], axis=0)

    alt = analyzer.analyze_endmembers(spectra, wl)

    assert alt.depth_1900.shape == (2,)
    assert alt.depth_2300.shape == (2,)
    assert alt.ratio_2300_1900.shape == (2,)
    assert np.all(np.isfinite(alt.depth_1900))
    assert np.all(np.isfinite(alt.depth_2300))
    assert np.all(np.isfinite(alt.ratio_2300_1900))
    assert alt.depth_2300[0] > 0.1
    assert alt.depth_1900[0] > 0.05
    assert alt.ratio_2300_1900[0] > 1.0
    assert len(alt.features_1900) == 2
    assert len(alt.features_2300) == 2
    print(f"  [测试 1] 吸收峰深度计算: d1900={alt.depth_1900[0]:.4f}, d2300={alt.depth_2300[0]:.4f}, ratio={alt.ratio_2300_1900[0]:.4f} ✓")


def test_enrichment_mask():
    from src.alteration import HydrocarbonAlterationAnalyzer

    np.random.seed(42)
    n_bands = 200
    wl = np.linspace(400, 2500, n_bands)

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

    analyzer = HydrocarbonAlterationAnalyzer()
    alt = analyzer.full_analysis(endmembers, wl, abundance_maps)

    assert alt.enrichment_mask is not None
    assert alt.enrichment_score is not None
    assert alt.enrichment_mask.shape == (lines, samples)
    assert alt.enrichment_score.shape == (lines, samples)
    assert 0.0 < alt.enrichment_fraction < 1.0
    assert alt.enrichment_threshold > 0.0
    assert np.all(np.isfinite(alt.enrichment_score))
    assert np.all((alt.enrichment_mask == 0) | (alt.enrichment_mask == 1))

    mu8 = alt.get_mask_uint8()
    assert mu8.dtype == np.uint8
    assert np.max(mu8) in (0, 255)

    print(f"  [测试 2] 富集掩膜生成: shape={alt.enrichment_mask.shape}, fraction={alt.enrichment_fraction:.3%}, threshold={alt.enrichment_threshold:.4f} ✓")
    print(f"  [测试 2]   碳酸盐端元索引: {alt.endmember_carbonate_rich} ✓")


def test_event_hooks():
    from src import SpectralUnmixer, UnmixingConfig

    np.random.seed(42)
    cfg = UnmixingConfig.default()
    unmixer = SpectralUnmixer(config=cfg)

    events = []

    def on_after_unmix(result, unmixer_obj, cube):
        events.append("after_unmix")

    def on_after_mask(result, alt, unmixer_obj, cube):
        events.append("after_mask")

    unmixer.hooks.register("after_unmix", on_after_unmix)
    unmixer.hooks.register("after_mask", on_after_mask)

    unmixer.hooks.fire("after_unmix", None, unmixer, None)
    unmixer.hooks.fire("after_mask", None, None, unmixer, None)

    assert "after_unmix" in events
    assert "after_mask" in events
    print(f"  [测试 3] 事件钩子触发: events={events} ✓")


def test_unmixing_with_alteration():
    from src import SpectralUnmixer, UnmixingConfig, HyperSpectralCube

    np.random.seed(42)
    lines, samples, bands = 20, 20, 80
    wl = np.linspace(900, 2500, bands)

    configs = [(0.30, 0.08), (0.10, 0.45), (0.05, 0.02), (0.15, 0.55)]
    endmembers_true = []
    for d19, d23 in configs:
        em = 0.6 * np.ones(bands)
        em -= g(wl, 1900, 50, d19)
        em -= g(wl, 2300, 60, d23)
        em = np.clip(em, 0.05, 1.0)
        endmembers_true.append(em)
    endmembers_true = np.stack(endmembers_true, axis=0)

    yy, xx = np.mgrid[0:lines, 0:samples]
    center = (lines // 2, samples // 2)
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)

    abundances = np.zeros((lines, samples, 4))
    abundances[:, :, 0] = 0.3 + 0.1 * np.random.rand(lines, samples)
    abundances[:, :, 1] = np.where(dist < 6, 0.6, 0.1)
    abundances[:, :, 2] = 0.2 + 0.05 * np.random.rand(lines, samples)
    abundances[:, :, 3] = np.where((dist > 3) & (dist < 9), 0.7, 0.05)
    s = abundances.sum(axis=2, keepdims=True)
    abundances = abundances / (s + 1e-8)

    cube_data = np.zeros((lines, samples, bands))
    for i in range(4):
        cube_data += abundances[:, :, i:i+1] * endmembers_true[i:i+1, :]
    cube_data += 0.01 * np.random.randn(lines, samples, bands)
    cube_data = np.clip(cube_data, 0.0, 1.0)
    cube_np = np.transpose(cube_data, (2, 0, 1))

    cube = HyperSpectralCube(
        data=cube_np, wavelengths=wl, lines=lines, samples=samples, bands=bands
    )

    cfg = UnmixingConfig(
        num_endmembers=4,
        num_epochs=100,
        learning_rate=5e-4,
        batch_size=64,
        seed=42,
        enable_alteration_analysis=True,
        alteration_enrichment_percentile=70,
    )

    unmixer = SpectralUnmixer(config=cfg)
    result = unmixer.unmix(cube, verbose=False)

    assert result.has_alteration()
    assert result.alteration is not None
    alt = result.alteration

    assert alt.enrichment_mask is not None
    assert alt.enrichment_score is not None
    assert alt.enrichment_mask.shape == (lines, samples)
    assert 0.0 < alt.enrichment_fraction <= 1.0
    assert np.all(np.isfinite(alt.depth_1900))
    assert np.all(np.isfinite(alt.depth_2300))
    assert np.all(np.isfinite(alt.ratio_2300_1900))

    max_ratio_idx = int(np.argmax(alt.ratio_2300_1900))
    assert alt.depth_2300[max_ratio_idx] > 0.05

    mask_uint8 = alt.get_mask_uint8()
    assert mask_uint8.dtype == np.uint8
    assert mask_uint8.shape == (lines, samples)

    print(f"  [测试 4] 端到端解编+蚀变分析完成 ✓")
    print(f"  [测试 4]   碳酸盐富集端元: {alt.endmember_carbonate_rich}")
    print(f"  [测试 4]   富集区比例: {alt.enrichment_fraction:.2%}")
    print(f"  [测试 4]   最终 Loss: {result.final_loss:.4f}")


def test_fastapi_alteration_response():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    np.random.seed(42)
    lines, samples, bands = 16, 16, 80
    wl = np.linspace(900, 2500, bands)

    configs = [(0.20, 0.45), (0.20, 0.10), (0.05, 0.08), (0.15, 0.55)]
    endmembers_true = []
    for d19, d23 in configs:
        em = 0.6 * np.ones(bands)
        em -= g(wl, 1900, 50, d19)
        em -= g(wl, 2300, 60, d23)
        em = np.clip(em, 0.05, 1.0)
        endmembers_true.append(em)
    endmembers_true = np.stack(endmembers_true, axis=0)

    abundances = np.random.dirichlet([1.0] * 4, size=(lines, samples))
    cube_data = np.zeros((lines, samples, bands))
    for i in range(4):
        cube_data += abundances[:, :, i:i+1] * endmembers_true[i:i+1, :]
    cube_data = np.clip(cube_data + 0.005 * np.random.randn(lines, samples, bands), 0.0, 1.0)
    cube_np = np.ascontiguousarray(np.transpose(cube_data, (2, 0, 1)), dtype=np.float32)

    hdr_content = (
        "ENVI\n"
        "description = { Test HSI data }\n"
        f"samples = {samples}\n"
        f"lines = {lines}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 0\n"
        "wavelength units = Nanometers\n"
        "wavelength = {" + ", ".join([f"{w:.2f}" for w in wl]) + "}\n"
    )
    raw_bytes = cube_np.tobytes()

    hdr_file = ("test.hdr", io.BytesIO(hdr_content.encode()))
    raw_file = ("test.raw", io.BytesIO(raw_bytes))

    resp = client.post(
        "/api/v1/unmix",
        files={"hdr_file": hdr_file, "raw_file": raw_file},
        data={
            "num_endmembers": "4",
            "num_epochs": "80",
            "seed": "42",
            "enable_alteration_analysis": "true",
            "alteration_enrichment_percentile": "70",
        },
    )

    assert resp.status_code == 200, f"status={resp.status_code}, body={resp.text[:300]}"
    data = resp.json()
    assert data["success"] is True

    assert "alteration" in data
    alt_resp = data["alteration"]
    assert alt_resp["enabled"] is True
    assert alt_resp["success"] is True
    assert "enrichment_mask" in alt_resp
    assert "enrichment_score" in alt_resp
    assert "enrichment_threshold" in alt_resp
    assert "enrichment_fraction" in alt_resp
    assert "endmember_stats" in alt_resp
    assert len(alt_resp["endmember_stats"]) == 4
    assert "carbonate_rich_endmembers" in alt_resp

    for es in alt_resp["endmember_stats"]:
        assert "endmember_index" in es
        assert "depth_1900" in es
        assert "depth_2300" in es
        assert "ratio_2300_1900" in es
        assert "is_carbonate_rich" in es

    mask_list = alt_resp["enrichment_mask"]
    assert len(mask_list) == lines
    assert len(mask_list[0]) == samples

    print(f"  [测试 5] FastAPI 蚀变响应: 字段齐全 ✓")
    print(f"  [测试 5]   富集区比例: {alt_resp['enrichment_fraction']:.2%}")
    print(f"  [测试 5]   碳酸盐端元: {alt_resp['carbonate_rich_endmembers']}")


def run_all():
    print("\n" + "=" * 70)
    print("海底烃类蚀变分析旁路 测试套件")
    print("=" * 70)

    test_absorption_band_depth()
    test_enrichment_mask()
    test_event_hooks()
    test_unmixing_with_alteration()
    test_fastapi_alteration_response()

    print("\n" + "=" * 70)
    print("全部 5 项测试通过 ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

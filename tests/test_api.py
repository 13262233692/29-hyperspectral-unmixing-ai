"""
测试 FastAPI 接口
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """测试健康检查接口"""
    print("测试健康检查接口...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    print(f"  状态: {data['status']}")
    assert data["status"] == "healthy"
    print("  ✓ 通过")


def test_root():
    """测试根路径"""
    print("\n测试根路径...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    print(f"  名称: {data['name']}")
    print(f"  版本: {data['version']}")
    print("  ✓ 通过")


def test_api_health():
    """测试 API 健康检查"""
    print("\n测试 API 健康检查...")
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    print(f"  设备: {data['device']}")
    print(f"  CUDA 可用: {data['cuda_available']}")
    print(f"  模型就绪: {data['model_ready']}")
    print("  ✓ 通过")


def test_unmix_endpoint():
    """测试解编接口（模拟数据）"""
    print("\n测试解编接口...")

    import io
    import struct

    lines, samples, bands = 20, 20, 50
    num_endmembers = 4

    np.random.seed(42)
    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)

    true_endmembers = np.zeros((num_endmembers, bands), dtype=np.float32)
    for i in range(num_endmembers):
        peak = 500 + i * 500
        width = 100 + i * 30
        base = 0.5 + i * 0.1
        gaussian = np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))
        true_endmembers[i, :] = base - 0.3 * gaussian
    true_endmembers = np.clip(true_endmembers, 0, 1)

    abundances = np.random.dirichlet(np.ones(num_endmembers) * 2, size=lines * samples).astype(np.float32)
    pixel_spectra = abundances @ true_endmembers
    pixel_spectra += 0.01 * np.random.randn(*pixel_spectra.shape).astype(np.float32)
    pixel_spectra = np.clip(pixel_spectra, 0, 1)
    cube_data = pixel_spectra.T.reshape(bands, lines, samples)

    raw_bytes = cube_data.astype(np.float32).tobytes()

    hdr_content = f"""ENVI
description = {{ Test HSI data }}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 0
wavelength = {{
"""
    wl_str = ", ".join([f"{w:.2f}" for w in wavelengths])
    hdr_content += "  " + wl_str + "\n}\n"

    hdr_file = ("test.hdr", hdr_content.encode(), "text/plain")
    raw_file = ("test.raw", raw_bytes, "application/octet-stream")

    response = client.post(
        "/api/v1/unmix",
        files={
            "hdr_file": hdr_file,
            "raw_file": raw_file,
        },
        data={
            "num_endmembers": "4",
            "num_epochs": "100",
            "learning_rate": "0.001",
            "decoder_type": "linear",
            "normalize": "true",
            "device": "cpu",
            "verbose": "false",
        },
    )

    print(f"  状态码: {response.status_code}")

    if response.status_code != 200:
        print(f"  错误: {response.text}")
        return False

    data = response.json()
    print(f"  成功: {data['success']}")
    print(f"  空间分辨率: {data['lines']} x {data['samples']}")
    print(f"  波段数: {data['bands']}")
    print(f"  端元数: {data['num_endmembers']}")
    print(f"  最终损失: {data['final_loss']:.6f}")
    print(f"  处理时间: {data['processing_time']:.2f}s")

    assert data["success"] == True
    assert data["num_endmembers"] == 4
    assert len(data["endmembers"]) == 4
    assert len(data["endmember_info"]) == 4

    print("  ✓ 通过")
    return True


def main():
    """运行所有 API 测试"""
    print("=" * 60)
    print("FastAPI 接口测试")
    print("=" * 60)

    try:
        test_health_check()
        test_root()
        test_api_health()
        test_unmix_endpoint()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n测试失败: {e}")
        return 1

    print("\n" + "=" * 60)
    print("所有 API 测试通过!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

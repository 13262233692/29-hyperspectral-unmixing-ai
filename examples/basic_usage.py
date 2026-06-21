"""
高光谱解编引擎 - 使用示例
演示如何使用 SpectralUnmixer 进行无监督光谱解编
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    SpectralUnmixer,
    UnmixingConfig,
    MemoryConfig,
    HSIDataLoader,
    HyperSpectralCube,
    DeviceType,
)


def example_basic_usage():
    """基础使用示例：生成模拟数据并解编"""
    print("=" * 60)
    print("示例 1: 基础解编流程")
    print("=" * 60)

    lines, samples, bands = 50, 50, 120
    num_endmembers = 4

    print(f"生成模拟高光谱数据: {bands} 波段, {lines}x{samples} 空间分辨率")

    np.random.seed(42)
    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)

    true_endmembers = np.zeros((num_endmembers, bands), dtype=np.float32)
    peak_centers = [550, 900, 1400, 2200]
    peak_widths = [80, 120, 150, 200]
    base_levels = [0.7, 0.6, 0.8, 0.5]

    for i in range(num_endmembers):
        peak = peak_centers[i]
        width = peak_widths[i]
        base = base_levels[i]
        gaussian = np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))
        true_endmembers[i, :] = base - 0.4 * gaussian
    true_endmembers = np.clip(true_endmembers, 0, 1)

    abundances = np.random.dirichlet(np.ones(num_endmembers) * 1.5, size=lines * samples)
    abundances = abundances.astype(np.float32)
    pixel_spectra = abundances @ true_endmembers
    pixel_spectra += 0.01 * np.random.randn(*pixel_spectra.shape).astype(np.float32)
    pixel_spectra = np.clip(pixel_spectra, 0, 1)
    cube_data = pixel_spectra.T.reshape(bands, lines, samples)

    cube = HyperSpectralCube(
        data=cube_data,
        wavelengths=wavelengths,
        lines=lines,
        samples=samples,
        bands=bands,
    )

    unmix_config = UnmixingConfig(
        num_endmembers=num_endmembers,
        num_epochs=200,
        learning_rate=5e-4,
        decoder_type="linear",
        normalize=True,
        loss_alpha=0.6,
        sparsity_beta=0.005,
        early_stopping_patience=30,
        seed=42,
    )

    memory_config = MemoryConfig(
        device=DeviceType.AUTO,
        gpu_memory_fraction=0.7,
        enable_mixed_precision=True,
    )

    print("\n初始化解编引擎...")
    unmixer = SpectralUnmixer(config=unmix_config, memory_config=memory_config)

    print("开始无监督解编...")
    result = unmixer.unmix(cube, verbose=True)

    print("\n" + "=" * 60)
    print("解编结果汇总")
    print("=" * 60)
    print(f"空间分辨率: {result.lines} x {result.samples}")
    print(f"波段数: {result.bands}")
    print(f"端元数: {result.num_endmembers}")
    print(f"最终损失: {result.final_loss:.6f}")
    print(f"训练轮数: {len(result.loss_history)}")

    print("\n端元信息:")
    for i in range(result.num_endmembers):
        name = result.endmember_names[i]
        mean_ab = np.mean(result.abundance_maps[i, :, :])
        max_ab = np.max(result.abundance_maps[i, :, :])
        spec = result.endmembers[i, :]
        print(f"  {name}: 平均丰度={mean_ab:.3f}, 最大丰度={max_ab:.3f}, "
              f"波谱范围=[{spec.min():.3f}, {spec.max():.3f}]")

    return result


def example_save_load_model():
    """模型保存与加载示例"""
    print("\n" + "=" * 60)
    print("示例 2: 模型保存与加载")
    print("=" * 60)

    from tempfile import TemporaryDirectory

    cube_data = np.random.rand(50, 20, 20).astype(np.float32)
    cube = HyperSpectralCube(
        data=cube_data,
        wavelengths=np.linspace(400, 2400, 50, dtype=np.float32),
        lines=20,
        samples=20,
        bands=50,
    )

    config = UnmixingConfig(num_endmembers=3, num_epochs=50, seed=42)
    mem_config = MemoryConfig(device=DeviceType.CPU)

    unmixer = SpectralUnmixer(config=config, memory_config=mem_config)
    result = unmixer.unmix(cube, verbose=False)

    with TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pt")
        unmixer.save_model(model_path)
        print(f"模型已保存到: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path) / 1024:.1f} KB")

        new_unmixer = SpectralUnmixer(memory_config=mem_config)
        new_unmixer.load_model(model_path)
        print("模型加载成功")

        test_spectrum = np.random.rand(1, 50).astype(np.float32)
        abundances = new_unmixer.model.encode(
            __import__("torch").from_numpy(test_spectrum)
        ).detach().numpy()
        print(f"测试推理 - 输出丰度形状: {abundances.shape}")
        print(f"丰度和: {abundances.sum():.4f}")


def example_memory_management():
    """显存管理示例"""
    print("\n" + "=" * 60)
    print("示例 3: 显存管理")
    print("=" * 60)

    from src.memory_manager import MemoryManager, MemoryConfig

    config = MemoryConfig(
        device=DeviceType.CPU,
        max_batch_size=4096,
        safety_margin_mb=256,
    )

    mm = MemoryManager(config)
    print(f"设备: {mm.get_device()}")

    allocated, free, total = mm.get_memory_info()
    if mm.is_cuda():
        print(f"已用显存: {allocated:.1f} MB")
        print(f"空闲显存: {free:.1f} MB")
        print(f"总显存: {total:.1f} MB")
    else:
        print("CPU 模式，无 GPU 显存限制")

    batch_size = mm.optimize_batch_size(sample_bytes=200 * 4, model_bytes=1024 * 1024)
    print(f"优化后批大小: {batch_size}")

    mm.empty_cache()
    print("显存缓存已清理")


def main():
    """运行所有示例"""
    print("高光谱数据解编推理引擎 - 使用示例")
    print("=" * 60)

    example_basic_usage()
    example_save_load_model()
    example_memory_management()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

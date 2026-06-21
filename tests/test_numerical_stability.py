"""
数值稳定性极端场景测试
模拟：
1. 死像元/坏像元（纯零向量，sensor dead pixels）
2. 钻井液严重污染（极端噪声 + NaN/Inf 注入）
3. 坏带（某些波段全零或 NaN）
4. 梯度爆炸触发场景
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    SpectralUnmixer,
    UnmixingConfig,
    MemoryConfig,
    HyperSpectralCube,
    SADLoss,
    SAMLoss,
    MSE_SAD_Loss,
    AbundanceSparsityLoss,
    clip_gradients,
    Autoencoder1D,
    DeviceType,
)


def test_loss_functions_dead_pixels():
    """测试 1: 损失函数对死像元（纯零向量）的稳定性"""
    print("=" * 70)
    print("测试 1: 损失函数对死像元（纯零向量）的稳定性")
    print("=" * 70)

    batch = 32
    bands = 100

    x_true = np.random.rand(batch, bands).astype(np.float32)
    x_pred = x_true + 0.01 * np.random.randn(batch, bands).astype(np.float32)

    n_dead = 8
    x_true[:n_dead, :] = 0.0
    x_pred[n_dead:n_dead*2, :] = 0.0

    x_true_t = torch.from_numpy(x_true)
    x_pred_t = torch.from_numpy(x_pred)

    print(f"Batch size: {batch}, Dead pixels (zero-vec): {n_dead} in true, {n_dead} in pred")

    sad = SADLoss()
    sam = SAMLoss()
    mse_sad = MSE_SAD_Loss(alpha=0.5)
    sparse = AbundanceSparsityLoss(beta=0.01)

    sad_val = sad(x_pred_t, x_true_t)
    sam_val = sam(x_pred_t, x_true_t)
    mse_sad_val = mse_sad(x_pred_t, x_true_t)

    abundances = torch.softmax(torch.randn(batch, 4), dim=1)
    abundances[:4, :] = 0.0
    sparse_val = sparse(abundances)

    print(f"  SAD Loss:  {sad_val.item():.6f}  NaN? {torch.isnan(sad_val).item()}")
    print(f"  SAM Loss:  {sam_val.item():.6f}  NaN? {torch.isnan(sam_val).item()}")
    print(f"  MSE+SAD:   {mse_sad_val.item():.6f}  NaN? {torch.isnan(mse_sad_val).item()}")
    print(f"  Sparsity:  {sparse_val.item():.6f}  NaN? {torch.isnan(sparse_val).item()}")

    all_finite = all([
        torch.isfinite(sad_val),
        torch.isfinite(sam_val),
        torch.isfinite(mse_sad_val),
        torch.isfinite(sparse_val),
    ])
    print(f"  ✓ 全部有限值: {all_finite}")

    return all_finite


def test_loss_functions_nan_inf():
    """测试 2: 损失函数对 NaN/Inf 输入的鲁棒性"""
    print("\n" + "=" * 70)
    print("测试 2: 损失函数对 NaN/Inf 输入的鲁棒性")
    print("=" * 70)

    batch = 16
    bands = 80

    x_true = np.random.rand(batch, bands).astype(np.float32)
    x_pred = x_true.copy()

    x_true[0, :] = np.nan
    x_true[1, 10:20] = np.inf
    x_pred[2, :] = -np.inf
    x_pred[3, 30:40] = np.nan

    x_true_t = torch.from_numpy(x_true)
    x_pred_t = torch.from_numpy(x_pred)

    print(f"注入: NaN 向量 x1, Inf 片段 x1, -Inf 向量 x1, NaN 片段 x1")

    sad = SADLoss()
    sam = SAMLoss()
    mse_sad = MSE_SAD_Loss(alpha=0.5)

    sad_val = sad(x_pred_t, x_true_t)
    sam_val = sam(x_pred_t, x_true_t)
    mse_sad_val = mse_sad(x_pred_t, x_true_t)

    print(f"  SAD Loss:  {sad_val.item():.6f}  NaN? {torch.isnan(sad_val).item()}")
    print(f"  SAM Loss:  {sam_val.item():.6f}  NaN? {torch.isnan(sam_val).item()}")
    print(f"  MSE+SAD:   {mse_sad_val.item():.6f}  NaN? {torch.isnan(mse_sad_val).item()}")

    all_finite = all([
        torch.isfinite(sad_val),
        torch.isfinite(sam_val),
        torch.isfinite(mse_sad_val),
    ])
    print(f"  ✓ 全部有限值: {all_finite}")

    return all_finite


def test_loss_functions_boundary():
    """测试 3: acos() 边界极端情况"""
    print("\n" + "=" * 70)
    print("测试 3: cos(angle) 边界极端情况 (acos ±1 附近)")
    print("=" * 70)

    batch = 10
    bands = 50

    cases_passed = []

    for i, (scale_true, scale_pred) in enumerate([
        (1e-10, 1e-10),
        (1e-15, 1e-15),
        (1.0, 1.0 + 1e-6),
        (1.0 + 1e-6, 1.0),
        (-1.0, 1.0),
    ]):
        x_true = np.random.randn(batch, bands).astype(np.float32) * scale_true
        x_pred = np.random.randn(batch, bands).astype(np.float32) * scale_pred

        if i == 2:
            x_true = np.ones_like(x_true) * 0.5
            x_pred = x_true * (1.0 + 1e-4)
        elif i == 3:
            x_true = np.ones_like(x_true) * 0.5 * (1.0 + 1e-4)
            x_pred = np.ones_like(x_pred) * 0.5
        elif i == 4:
            x_true = np.ones_like(x_true) * 0.5
            x_pred = -np.ones_like(x_pred) * 0.5

        x_true_t = torch.from_numpy(x_true)
        x_pred_t = torch.from_numpy(x_pred)

        sad = SADLoss()
        val = sad(x_pred_t, x_true_t)
        finite = torch.isfinite(val).item()
        cases_passed.append(finite)
        print(f"  Case {i+1}: scale=({scale_true:.0e},{scale_pred:.0e}) → Loss={val.item():.6f}, 有限={finite}")

    all_pass = all(cases_passed)
    print(f"  ✓ 全部边界测试通过: {all_pass}")
    return all_pass


def test_gradient_clipping():
    """测试 4: 梯度裁剪阀门"""
    print("\n" + "=" * 70)
    print("测试 4: 梯度裁剪阀门 (Gradient Clipping)")
    print("=" * 70)

    model = Autoencoder1D(in_bands=60, num_endmembers=4, decoder_type="linear")
    x = torch.randn(16, 60)

    x_corrupted = x.clone()
    x_corrupted[0, :] = 1e6

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = MSE_SAD_Loss(alpha=0.5)

    optimizer.zero_grad()
    recon, ab = model(x_corrupted)
    loss = loss_fn(recon, x_corrupted)
    loss.backward()

    gn_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    print(f"  裁剪前总梯度范数: {gn_before.item():.4f}")

    optimizer.zero_grad()
    recon, ab = model(x_corrupted)
    loss = loss_fn(recon, x_corrupted)
    loss.backward()

    gn = clip_gradients(model, max_norm=1.0, norm_type=2.0)
    print(f"  clip_gradients 返回值: {gn:.4f}")

    gn_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    print(f"  裁剪后总梯度范数: {gn_after.item():.4f}")
    print(f"  ✓ 梯度被有效约束: {gn_after.item() <= 1.0 + 1e-5}")

    return gn_after.item() <= 1.0 + 1e-5


def test_unmixing_with_dead_pixels():
    """测试 5: 完整解编流程 - 含严重死像元与坏带"""
    print("\n" + "=" * 70)
    print("测试 5: 完整解编流程 - 严重死像元 + 钻井液污染 + 坏带")
    print("=" * 70)

    lines, samples, bands = 40, 40, 80
    num_endmembers = 4
    np.random.seed(42)

    wavelengths = np.linspace(400, 2500, bands, dtype=np.float32)
    true_endmembers = np.zeros((num_endmembers, bands), dtype=np.float32)

    for i in range(num_endmembers):
        peak = 500 + i * 500
        width = 100 + i * 30
        base = 0.5 + i * 0.1
        gaussian = np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))
        true_endmembers[i, :] = base - 0.4 * gaussian
    true_endmembers = np.clip(true_endmembers, 0, 1)

    abundances = np.random.dirichlet(np.ones(num_endmembers) * 1.5, size=lines * samples).astype(np.float32)
    pixel_spectra = abundances @ true_endmembers
    pixel_spectra = np.clip(pixel_spectra, 0, 1)

    cube_data = pixel_spectra.T.reshape(bands, lines, samples)

    # --- 注入死像元 (10% 纯零) ---
    total_pixels = lines * samples
    n_dead = int(total_pixels * 0.10)
    dead_indices = np.random.choice(total_pixels, n_dead, replace=False)
    flat = cube_data.reshape(bands, -1)
    flat[:, dead_indices] = 0.0

    # --- 注入坏带 (3个波段全零) ---
    bad_bands = [5, 25, 60]
    cube_data[bad_bands, :, :] = 0.0

    # --- 注入钻井液污染 (强噪声 + 异常值) ---
    noise_mask = np.random.rand(bands, lines, samples) < 0.15
    cube_data[noise_mask] += 0.5 * np.random.randn(np.sum(noise_mask)).astype(np.float32)

    # --- 注入 NaN 像素 (5%) ---
    nan_mask = np.random.rand(bands, lines, samples) < 0.05
    cube_data[nan_mask] = np.nan

    # --- 注入 Inf 像素 (1%) ---
    inf_mask = np.random.rand(bands, lines, samples) < 0.01
    cube_data[inf_mask] = np.inf

    cube_data = np.clip(cube_data, -100, 100)

    total_elements = bands * lines * samples
    zero_pixels = np.sum(np.all(cube_data.reshape(bands, -1) == 0, axis=0))
    nan_count = np.sum(np.isnan(cube_data))
    inf_count = np.sum(np.isinf(cube_data))

    print(f"空间分辨率: {lines}x{samples}, 波段数: {bands}")
    print(f"纯死像元数: {zero_pixels}/{lines*samples} ({zero_pixels/(lines*samples)*100:.1f}%)")
    print(f"NaN 元素: {nan_count}/{total_elements} ({nan_count/total_elements*100:.2f}%)")
    print(f"Inf 元素: {inf_count}/{total_elements} ({inf_count/total_elements*100:.2f}%)")
    print(f"坏带 (全零波段): {bad_bands}")
    print(f"钻井液污染噪声比例: 15%")

    config = UnmixingConfig.robust()
    config.num_endmembers = 4
    config.num_epochs = 100
    config.verbose = False
    config.gradient_clip_max_norm = 1.0

    mem_config = MemoryConfig(device=DeviceType.CPU)

    unmixer = SpectralUnmixer(config=config, memory_config=mem_config)

    cube = HyperSpectralCube(
        data=cube_data.copy(),
        wavelengths=wavelengths,
        lines=lines,
        samples=samples,
        bands=bands,
    )

    try:
        result = unmixer.unmix(cube, verbose=True)

        loss_np = np.array(result.loss_history)
        has_nan = np.any(np.isnan(loss_np))
        has_inf = np.any(np.isinf(loss_np))
        loss_monotonic = loss_np[-1] <= loss_np[0] * 2.0

        print(f"\n--- 解编结果 ---")
        print(f"完成轮数: {len(result.loss_history)}")
        print(f"初始 Loss: {loss_np[0]:.6f}")
        print(f"最终 Loss: {loss_np[-1]:.6f}")
        print(f"Loss 存在 NaN: {has_nan}")
        print(f"Loss 存在 Inf: {has_inf}")
        print(f"Loss 未爆炸: {loss_monotonic}")

        print(f"\n端元丰度分布:")
        for i in range(result.num_endmembers):
            mean_ab = float(np.mean(result.abundance_maps[i, :, :]))
            max_ab = float(np.max(result.abundance_maps[i, :, :]))
            nan_in_ab = np.any(np.isnan(result.abundance_maps[i, :, :]))
            print(f"  Endmember_{i+1}: 均值={mean_ab:.4f}, 最大={max_ab:.4f}, NaN={nan_in_ab}")

        endm_nan = np.any(np.isnan(result.endmembers))
        print(f"\n端元矩阵 NaN: {endm_nan}")

        success = (not has_nan) and (not has_inf) and loss_monotonic and (not endm_nan)
        print(f"\n✓ 完整极端场景测试通过: {success}")
        return success

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n✗ 解编过程抛出异常: {e}")
        return False


def test_acos_numerical_edge():
    """测试 6: 专门验证 acos 的定义域保护"""
    print("\n" + "=" * 70)
    print("测试 6: acos() 定义域边界的数值截断验证")
    print("=" * 70)

    x = torch.randn(8, 50)
    y = torch.randn(8, 50)

    x[:1, :] = 1e-30
    y[1:2, :] = 1e-30

    dot = torch.sum(x * y, dim=1)
    norm_x = torch.norm(x, dim=1)
    norm_y = torch.norm(y, dim=1)
    cos_raw = dot / (norm_x * norm_y + 1e-20)

    print(f"未经保护的 cos_angle 范围: [{cos_raw.min().item():.6e}, {cos_raw.max().item():.6e}]")
    print(f"  超出 [ -1, 1 ]: {(cos_raw < -1).sum().item() + (cos_raw > 1).sum().item()} 个样本")

    sad_fn = SADLoss(eps_div=1e-10, eps_clamp=1e-6)
    val = sad_fn(x, y)
    print(f"经 SADLoss 处理后: Loss={val.item():.6f}, NaN={torch.isnan(val).item()}")

    result = torch.isfinite(val).item()
    print(f"  ✓ acos 定义域保护生效: {result}")
    return result


def main():
    print("\n" + "=" * 70)
    print("高光谱解编引擎 - 数值稳定性极端测试套件")
    print("=" * 70)

    tests = [
        ("损失函数-死像元(纯零向量)", test_loss_functions_dead_pixels),
        ("损失函数-NaN/Inf 注入", test_loss_functions_nan_inf),
        ("acos 边界极端值", test_loss_functions_boundary),
        ("梯度裁剪阀门", test_gradient_clipping),
        ("完整解编-极端污染场景", test_unmixing_with_dead_pixels),
        ("acos 定义域数值截断", test_acos_numerical_edge),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            passed = False
            print(f"  ✗ 异常: {e}")
        results.append((name, passed))

    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_pass = all(p for _, p in results)
    print(f"\n总体: {'全部通过' if all_pass else '存在失败'}")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

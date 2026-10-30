# 原始无缩放 FP4 的逐位实验

这是 N21 的一个可运行实验配置：使用原始 `fp4_to_bf16` 的两个 JIT 函数，比较 bf16 输出的原始位和输入、输出分配内存的完整写入范围。随包参考值已由 Lean 导出，并与独立整数格式表核对全部 256 个输入字节。运行 GPU 实验无需重新构建 Lean。

从仓库根目录运行，结果目录必须不存在：

```bash
python experiments/floating_point/value_validation/fp4_unscaled/run.py run \
  --reference experiments/floating_point/value_validation/fp4_unscaled/reference \
  --output /tmp/fp4-unscaled-native
```

需要 CUDA 设备、PyTorch、NumPy 和 Triton。结果保存在 `report.json`，包含设备、驱动和编译版本、实际编译选项、TTIR/TTGIR/LLVM/PTX/cubin（编译器提供时）、指针与分配范围、全部观测位以及差异。NaN payload、正负零和任何位差异均保留；不使用浮点容差。

`MATCH_ON_TESTED_INPUTS` 只报告本次位值与写入范围一致。另查 `environment_binding_complete`：若驱动查询失败或设备 UUID 无法对应，字段为 `false`，环境绑定仍缺证据，不能据值匹配把完整实验工作项标为通过。

| 返回码/状态 | 含义 |
|---|---|
| 0 / `MATCH_ON_TESTED_INPUTS` | 本次 21 个配置的输出和完整分配内存检查一致。 |
| 1 / `MISMATCH` | 至少一个已运行配置存在位值或写入范围差异。 |
| 2 / `NOT_RUN` | CUDA 或依赖不可用，尚未尝试原核启动。 |
| 2 / `EXECUTION_ERROR` | 已尝试启动但运行、同步或采集失败；查看逐项记录。 |

配置使用独立且不重叠的输入和输出分配，非零起始地址，`BLOCK_SIZE_IN=16/512`、`num_warps=4`、`enable_fp_fusion=False`。21 组输入涵盖空输入、整块、尾部以及跨块，每次多启动一个完全被 mask 的 program；每种 block 都覆盖全部 256 字节。原核先输出高四位对应的值，再输出低四位。额外分配空间采用不同初值，检查输入没有被修改、输出活动区域之外没有被写入。输入和输出仍是分开的物理分配，没有覆盖别名配置。

**本工具尚未在 GPU 上执行。** 当前证据为 26 项 CPU 检查通过、SM90 两个原核配置离线编译通过，以及本机 CUDA 不可用的 `NOT_RUN` 回执。有限输入一致不能证明全部 GPU 执行正确。内部中间指令值没有采集，原始 N21 的编译器对应、所有地址/别名和调度关系也没有因此解除。它只补齐 N21 上述配置的采集器和独立参考，不把 37 个完整实验工作项改为完成。

仅需检查编译时运行：

```bash
python experiments/floating_point/value_validation/fp4_unscaled/run.py compile \
  --reference experiments/floating_point/value_validation/fp4_unscaled/reference \
  --output /tmp/fp4-unscaled-sm90 --arch 90
python -m unittest discover \
  -s experiments/floating_point/value_validation/fp4_unscaled -p test_fp4.py -v
```

离线模式显式选择目标架构，不产生 GPU 数值证据。GPU 模式由 Triton 选择当前设备目标，报告保存实际元数据。

`reference/original.py` 是冻结原文件，SHA-256 为 `8571cb56ce932af6a7dd68e4047ee0863984812ac303d0d2959fc544a65e9958`。运行时只导入 `original_jit.py`；它逐行提取原有两个装饰后的函数定义，函数体未改写，不执行原文件中的测试或 host wrapper。该提取模块也会在每次运行前后重新验证。参考值、工具、Lean 导出记录、原始源码及实验配置的哈希均被绑定；变更工具或配置必须生成新参考，不能复用旧结果。

工具保留 Triton namedtuple 编译元数据的字段名，并执行经过校验的 JIT 与独立 oracle 源码字节。oracle 的加载时、当前文件和参考记录三方哈希必须一致。生成参考、离线编译和 GPU 验证是分别执行、分别报告的步骤。

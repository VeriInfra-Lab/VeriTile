# 补充基础原子实验

补充目录定义基础标量表达式对，不把完整 softmax、Welford、LayerNorm 或归约作为
新的准入原子。当前运行统一使用 [two-gates 入口和配置](../README.md)。

```bash
python3 scripts/check_two_gates.py check --catalog supplement
python3 scripts/check_two_gates.py run --catalog supplement --output Logs/two-gates-supplement
python3 scripts/check_two_gates.py import Logs/two-gates-supplement --output Logs/two-gates-supplement.json
```

每个可运行配置固定采样 `2**32` 组，单位权重，不使用主目录三条 FP32 规则的近零提议。
均值门是有效样本逐元素 local-ULP 差的 `abs(mean)+5*SE <= 0.05`，峰值门是
原始未加权绝对 oracle 误差的 `candidate_peak <= 10*reference_peak`。
保留原有域和非有限筛选，原始组数和有效组数分别记录，不补抽。

[config.py](config.py) 提供关系、输入分布和精度，[rules.json](rules.json) 登记表达式，
[kernels.py](kernels.py) 是实际 Triton 源程序。总预算、分块、门槛和并发在
[共享配置](../two_gates_config.py) 中设置。

普通 log 和 exp 的 intrinsic/libdevice 候选保持不同函数与规则身份。
每个比较保留其他 intrinsic、分支边界、中间转换、输入域和最终输出转换。
编译保持 `enable_fp_fusion=False`，显式 FMA 仍按源码执行。
FP64-work 配置仅适用于 `DIV-MUL-RCP`，保留 FP32 输出和编译的残差 oracle。
不支持的精度组合明确列为 `UNSUPPORTED`。

有界整数计数使用 [专用配置](../primitives/count_config.py)，也已包含在默认全目录运行中：

```bash
python3 scripts/check_two_gates.py run --catalog counts --output Logs/two-gates-counts
```

已发布的 [数值表](report/summary.md) 是带有原源码身份的归档测量，不冒充当前入口的
新结果。严格证书替代实验义务时保留原测量，见
[当前证明依赖](../../../documents/FPRemainingAdmissionGaps.md)。
复现归档表以及更详细的表达式/分支说明见 [归档说明](HistoricalRuns.md)。
当前报告不会自动写入 Lean 准入；规则必须与实际证明使用的精度、域和源码对应。

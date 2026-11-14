# 从当前源码到 Real 与 FP 证明

以核心仓库的 [VectorAdd](../../bench/examples/VectorAdd/Kernels.lean) 为例。
`Kernels.lean` 保存 Python 原程序和交换操作数的候选；`IO.lean`、`Contract.lean`
保存共享接口与契约，`RealCorrect.lean` 和 `FPEquiv.lean` 分别给出独立的公开证明，
执行和读回助手放在 `Proofs/`。两个公开入口互不依赖。

## 严格模型证明：VectorAdd

[Real 入口](../../bench/examples/VectorAdd/RealCorrect.lean) 声明独立的数学加法公式。
[FP 入口](../../bench/examples/VectorAdd/FPEquiv.lean) 的实际声明是：

```lean
specification add_kernel_equiv (blockSize : Nat) :
    original blockSize ≡[([] : Spec.Assumptions FP.GuardedFragment)] optimized blockSize := by
  split_equiv
  all_goals solve_fp
```

`original`、`optimized` 连接同一份源码的两个实例，唯一变化是 `x + y` 与 `y + x`。
这里复用共享库严格 FP32 交换律证书，实验规则集为空。
`additionSite` 的输入域保留实际源前缀执行后两个操作数的有限性；固定软件理论、
精度、load/store、mask 和执行语义也保留。空实验规则集不表示对任意数值解释成立，
也不表示已证明 GPU 指令与软件模型一致。尺寸保持符号化。

```bash
lake build VeriTile
bash scripts/check-core-examples.sh
python3 scripts/check_comparator.py --file bench/examples/VectorAdd/RealCorrect.lean --trust
python3 scripts/check_comparator.py --file bench/examples/VectorAdd/FPEquiv.lean --trust
```

源码对照、Lean 编译与独立 comparator 重放分别检查不同义务。
comparator 的安装和隔离要求见 [工具说明](../../scripts/README.md#shared-comparator-gate)。
VeriTileBench 的 [TritonBenchVectorAddition](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/bench/examples/TritonBenchVectorAddition/FPEquiv.lean)
使用严格模型证明；其带 mask 的源程序额外保留实际求值 lane 的有限性条件。
该题的源码、构建和验证入口位于 VeriTileBench。

## 实验准入的证明：LogExp

[LogExp](../../bench/examples/LogExp/FPEquiv.lean) 说明真正使用数值假设的情形：

```lean
specification log_exp_equiv (R : Rules)
    (xReg yReg : RegionName) (blockSize : Nat) :
    originalIO xReg yReg blockSize ≡[R] optimizedIO xReg yReg blockSize
```

`Rules` 来自 `FP.LogExp`。它为受保护的 `log(exp(x))` 消去提供 `log_exp_cancel`，
证明检查实际源码的成功执行、输出、定义域和内存 frame。
Lean 在 `R` 下检查组合推导，不把有限采样接受变成全域 IEEE 位值相等，
也不自动证明整个 kernel 的统计误差界。

```bash
python3 scripts/check_comparator.py --file bench/examples/LogExp/FPEquiv.lean --trust
```

## 检查依赖和证据

两个 FP 文件均执行 `#print_fp_assumptions`。输出分开列出严格模型证书与实际引用的
实验规则；规则表中未被证明引用的条目不算使用，无法展开的 FP 证明前提会标为
unresolved。该遍历涵盖可达辅助证明，不声称计算了逻辑上最小依赖集。
公理审计、statement 检查和 comparator 仍需独立运行；打印命令不关闭这些义务。

当前分类和用量见 [FPRemainingAdmissionGaps.md](../../documents/FPRemainingAdmissionGaps.md)。
测量报告、Lean 准入表与原始 GPU bundle 分别记录实验结果、可引用的规则和可重放证据。
报告适用于其中绑定的源码、精度、输入域和配置。数值测量不自动更新 Lean 准入表。
导出工具的 `--trust-report` 明确信任报告，并检查完整性和来源；该操作本身不执行
CPU/GPU 重放。运行、独立 CPU 重放和证据格式见 [实验指南](README.md)。

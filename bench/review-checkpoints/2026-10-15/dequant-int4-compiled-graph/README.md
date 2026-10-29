# matmul_dequantize_int4：原编译图与舍入反例

原始 JIT 在 **8 个离线配置**下均编译通过：SM80/SM90 × `NO_GROUPS` 两分支 × fusion 开关，使用原 17 个 autotune 配置中的 64×128×32、4 warps、4 stages。原函数体和签名逐字绑定，仅绕过 host autotune 包装以显式选择其中一个配置。没有启动 GPU。

完整 TTIR/PTX 的独立数据流审查确认：

- 8 个配置都把循环的 live accumulator 作为 `tt.dot` 第三个操作数，再 yield 回循环并存储。关闭 fusion 仍保留这个结构。
- fusion 关闭时，weight 与 scale 的半精度乘法后执行半精度减法。
- fusion 开启时，weight×scale−zeroProduct 成为 `fma.rn.f16`；zeroProduct 仍单独计算，然后取负送入 FMA。不能略过这次 zero 乘法或认为它与 weight 共用一次未舍入乘法。

[独立审查](independent-review/receipt.json)通过 76 项核对，检查了 8 条完整 SSA 链和 256 条 PTX 输出寄存器链。采集器 `ptx_dequant_lines` 只是筛选摘录，会漏掉 `mul.f16`/`neg.f16`；完整数据流见 [PTX 审查](independent-review/ptx-dataflow.json)。

[Lean 反例](initial-capture/DequantFusionGap.lean)使用有限、正常大小的半精度输入：weight=3、zero=2、scale=`0x3c01`（1025/1024）。先分别舍入乘积再减得到 `0x3c02`；将 weight 乘法和减法合并为一次舍入得到 `0x3c01`。3 条普通定理编译和公理检查通过，[独立 Fraction 枚举](independent-review/independent-rational-counterexample.json)得到相同结果。

该结果反驳一个普适标量融合等式，**没有证明完整 tile 的输出差异或实际 GPU 输出**。它不增加公开 specification，也不改变 51/132/1、原 41 项 30/11 的任务分类。已有 source-AST 合同保留，下一步需要针对实际编译图建立完整值合同；不能把已反驳的等式作为等待实验批准的假设。

[最新采集回执](capture/receipt.json)绑定完整产物与实际工具链：415 个 Triton 包/ptxas/libdevice/Python 文件在编译前后检查，0 漂移；操作系统和全部共享库没有被宣称为完整封装。旧较小环境清单的 v1 采集保留在 `initial-capture/`，没有将两个版本相加为 16 个独立配置。

具体待绑定关系见 [remaining-relations.json](remaining-relations.json)。本包与[原 41 项缺口报告](../original41-remaining-analysis/README.md)、[当前软件增量验收](../software-closure-metadata/README.md)分开记录。

# 无缩放 FP4 实验工具补充验收

本包交付 N21 的一个具体实验配置：原始无缩放 FP4 JIT、独立整数参考和可直接在 GPU 上采集原始位的工具。工具已安装到 [fp4_unscaled](../../../../../experiments/floating_point/value_validation/fp4_unscaled/README.md)。

| 检查 | 实际结果 |
|---|---|
| 原始函数体绑定 | 原文件 SHA 固定，两个装饰后的 JIT 函数逐行提取并复核。 |
| 软件参考 | Lean CPU 导出全部 256 字节，独立整数格式表逐项一致。 |
| 安装后检查 | 26 项通过，涵盖正负零、掩码尾部、内存写入范围和证据篡改拒绝。 |
| 原核 SM90 离线编译 | Triton 3.8.0，block 16/512 两个配置均通过，编译产物完整保留。 |
| GPU 实测 | `NOT_RUN`，CUDA 不可用，0 次启动；没有 GPU 通过记录。 |

21 个测试配置使用独立不重叠分配、非零地址和一个额外完全被 mask 的 program。对输入全分配及输出全分配进行逐位检查，包含全部 256 个输入字节。未覆盖物理别名，未采集内部操作数，有限样本不构成通用编译器或硬件正确性证明。因此本补充不改变原 184 项的 50/133/1 分类，也不把 N21 或 37 项完整工作清单标为完成。

入口：[已安装运行说明](../../../../../experiments/floating_point/value_validation/fp4_unscaled/README.md)、[安装后检查](installed-verification.json)、[离线结果](offline/report.json)、[GPU 未运行回执](native-attempt/report.json)、[Lean 导出来源](producer/export-receipt.json)、[本包回执](receipt.json)。原统一验收 [final-acceptance](../final-acceptance/SUMMARY.zh-CN.md) 保持冻结；本包作为后续补充单独审查。

本 v2 修复了元数据字段名丢失，以及原 JIT/独立 oracle 可能读取未绑定 timestamp .pyc 的来源问题。三个问题均有独立复现和修复检查，见 independent-reviews；旧 v1 保留，不声称其历史结果已受到缓存污染。新回执仅对应修复后源码、重新生成的参考及重新执行的离线编译。

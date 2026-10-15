# 当前增量验收：matmul_dequantize_int4 完整软件位语义

当前 **184 项：56 项软件位值证明、127 项依赖后端、1 项修复候选**。最初那 41 项为 **35／6**。本次仅 matmul_dequantize_int4 升级，见[184 项表](tasks-184.tsv)及[剩余 6 项](remaining-original41.tsv)。软件位语义闭合仍不代表 GPU 已验收。

[新增证明](../dequant-int4-source-compiled-software-closure/README.md)包括原源码、非融合 carry-dot、融合 half FMA carry-dot 三条公开合同。原源码主条固定每块从 +0 开始的 fp32 multiply/add 和独立块间加法；两种 compiled 图分别记录，不替代主条。两 NO_GROUPS、完整旧 K 循环/参数、PackedInputs、地址条件、初态 alias 和全内存/frame 保留，没有新增有限数、无 NaN 或单块前提。

融合图通过任务局部指令解释器读取实际 SSA 寄存器，保留共享 DSL prefix/suffix、prelude 的实际循环计数及原 epilogue。它作为显式 task_local_instruction_extension 附件绑定，未冒称共享 DSL 原生支持 scalar FMA；metadata 局部检查入口和调用链，共享 collector 未修改。

**15 个新增模块独立复编、58 条具名公理检查通过；45 项元数据测试与独立重生成通过。** 当前 889 条公开声明、3,826 个注册模块；构建证据是旧 3,811 记录重新核验加 15 条新实际编译。22 个安装文件、输入清单和 ZIP 成员匹配，时间戳差异单列。见[组合回执](incremental-build/receipt.json)、[生成比较](independent-generation/comparison.json)、[安装后检查](actual-workspace-gates/receipt.json)、[根审查](root-review.json)。

191 份 Python 输入、旧 886 条公开记录、36 条扩展任务及 59 个证据包均保留。[新证据索引](evidence-package-index.json)共 60 个包，包数不是完成任务数。889 = 747 历史 + 142 新声明，其中新增 138 条值合同、2 条 frame、1 条恒等式、1 条协议。

原冻结 native 证据只有一个 tile 的 8 个离线配置；MMA/WGMMA 精度与顺序、half NaN/FTZ、byte/address ABI 和实际 GPU 执行仍未证明。已知半精度融合差异不列为等待实验批准的普遍等式。[合法 tile 实验工具](../dequant-int4-native-characterization/README.md)已独立通过 45 项测试和 8 个离线配置；全部输入/输出与 guard 均采集，实际 GPU 状态仍为 NOT_RUN。[已交付 int4 采集包](../int4-native-characterization/README.md)保留其单独范围。

剩余为 token_softmax_bloom、token_softmax_llama、logsumexp_fwd、rbe_triton_transform、parallel_retention_attention、mixed_sparse_attention。 [Bloom 指数缺口](../token-softmax-bloom-exp-lowering-gap/README.md)已有正式反例和实验采集包；中间指数反例不等于最终 softmax 输出反例，不增加完成数量。官方 comparator 继续跑冻结 F3 的 365 文件／872 声明；本版后来新增的 17 条声明另列。[复现命令](../../../current_source_acceptance.md)、[上一版 int4 验收](../int4-source-dotacc-software-metadata/README.md)、[任务当前说明](../../../tritonbench_g/matmul_dequantize_int4/SOFTWARE-ACCEPTANCE.md)。

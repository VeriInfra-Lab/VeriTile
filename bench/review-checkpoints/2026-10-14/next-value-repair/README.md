# TritonBench-G 当前浮点证明与验收

当前 **184 项、891 条公开声明、3,834 个注册模块**；[当前验收](../../2026-10-15/raw-copy-software-metadata/README.md)及[逐项表](../../2026-10-15/raw-copy-software-metadata/tasks-184.tsv)记录 **58 项软件位值／125 项依赖后端／1 项修复候选**。完整硬件验收尚未完成。

原 41 项保持 **35／6**。这次在原 41 项之外补齐 [destindex_copy](../../2026-10-15/destindex-copy-raw-bit-closure/README.md) 和 [matrix_transpose](../../2026-10-15/matrix-transpose-raw-bit-closure/README.md) 的原完整域、真实 typed load/store、初态独立值与全内存/frame。8 个模块、44 条具名标准公理检查、52 项元数据测试通过，没有新增数值或无别名前提。

构建证据为原 3,826 条重新核验 + 新 8 个实际编译。891 = 747 历史 + 144 新声明（140 值合同、2 frame、1 恒等式、1 协议）。[剩余六项](../../2026-10-15/raw-copy-software-metadata/remaining-original41.tsv)保持原分类；Llama 文案纠正为原∀FloatDType范围，fp32只是实验默认配置。

[原 dequant 验收](../../2026-10-15/dequant-int4-source-compiled-software-metadata/README.md)和[dequant 实验工具](../../2026-10-15/dequant-int4-native-characterization/README.md)保留，GPU 仍未执行。[Bloom](../../2026-10-15/token-softmax-bloom-exp-lowering-gap/README.md)、[Llama](../../2026-10-15/token-softmax-llama-exp-lowering-gap/README.md)、[LSE](../../2026-10-15/logsumexp-exp-fusion-gap/README.md)、[RBE v2](../../2026-10-15/rbe-original-characterization-v2/README.md)提供具体缺口和采集工具，不增加完成数。

官方 comparator 的完整目标仍是冻结 F3 365 文件／872 声明，本版后来新增 19 条不混入；这里只链接[已冻结结果与维护快照](../../2026-10-15/README.md)，不把快照称为实时进度。旧129/43指标不等于硬件正确。[实验交接](native-experiment-handoff/EXPERIMENT-HANDOFF.md)、[复现指南](../../../current_source_acceptance.md)、[当前证据索引](../../2026-10-15/raw-copy-software-metadata/evidence-package-index.json)。[旧统一验收](final-acceptance/SUMMARY.zh-CN.md)和[旧索引](delivery-index.json)保持冻结。

# 当前增量验收：masked_select 原始位存储闭合

当前 **184 项：54 项软件位值证明、129 项依赖后端、1 项修复候选**。最初那 41 项为 **33／8**。[184 项验收表](tasks-184.tsv)本次仅 masked_select 升级，历史 registry 未改。

[新增证明](../masked-select-raw-bit-closure/README.md)覆盖原接受的完整七语句 mask/prefix 消费阶段，五种浮点字格式及 int64 payload transport。初态 RawCell.int 仍允许任意 Int；没有新增有限值、正前缀、正长度或区域分离限制。selection、prefix、payload 均读取初态，别名、masked 尾部及完整 frame 保留。全内存等式无需注入性，逐 lane 读回只沿用旧 active hinj。

**新增 3 模块独立编译、25 条具名公理检查通过；新增公开 specification 为 1 条。** 当前 884 条公开声明、3,802 个注册模块；构建证据为旧 3,799 记录重新核验加 3 条新实际编译，未声称 fresh 全库构建。见[组合回执](incremental-build/receipt.json)。

根代理独立重跑 62 项元数据测试和生成，10 个安装文件、输入清单及 ZIP 成员字节一致。时间戳导致快照回执哈希不同，见[生成比较](independent-generation/comparison.json)。父级 KV/kcache 的实际 source/package 绑定均保留。[根代理回执](root-review.json)记录安装后检查与源码漂移，[证据索引](evidence-package-index.json)和[复现指南](../../../current_source_acceptance.md)可直接核对。

prefix 仍为旧 Nat 模型：0 减 1 饱和为 0，不能当作机器 signed64 的 −1。host cumsum、机器地址溢出、物理 byte ABI、native64/GPU/跨 program 执行仍未证明。软件统计不代表完整硬件验收。[上一版 KV 验收](../kv-cache-software-metadata/README.md)冻结于 883 条声明；[官方 comparator 样例](../comparator-initial-samples/README.md)只覆盖冻结 F3 任务集的 1 条 FP4 声明，完整 corpus 结果和后续新增声明检查另行发布。

目前原 41 项剩余的 8 项集中在下面的运算；完整机器可读项见[剩余清单](remaining-original41.tsv)。实验需固定实际源码、dtype、编译选项和 IR/PTX，比较初态输入对应的独立位参考；有限样本匹配只记为该配置的经验结果。尚无具体位算法的函数应先确定算法/指令语义，不能用任意 callback 或数学实数 oracle 冒充闭合。

| 剩余任务 | 主要缺口 |
|---|---|
| int4_matmul | 当前累加值进入 dot 后的实际舍入顺序；不能假设与独立 dot+add 等价。 |
| matmul_dequantize_int4 | half 反量化的融合图与 dot 累加语义。 |
| token_softmax_bloom、token_softmax_llama | exp 位算法、最大值特殊值策略、归约树。 |
| logsumexp_fwd | exp/log 位算法及完整归约/scale 图。 |
| rbe_triton_transform | 原完整阶段的 pow/sin/cos 位算法。 |
| parallel_retention_attention | log2/exp2 衰减和各 dot 的累加图。 |
| mixed_sparse_attention | 完整循环的 exp2、dot、maximum 和归约树。 |

上述软件缺口与所有任务的 GPU 绑定分别记录；不能把 8 项之外的 native 对应当作已通过。已知反例推翻的图等价关系也不列为待实验批准的假设。

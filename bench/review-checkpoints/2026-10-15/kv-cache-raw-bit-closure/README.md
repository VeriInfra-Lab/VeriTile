# KV cache：混合格式原始位存储证明

新增公开合同覆盖旧的完整 masked x-block，K/V 可以各取五种格式，共 25 个组合。输入与对应 cache 同型；历史 36 种源注解组合另以任意 R 的普通定理保留，不声称 native64。

完整最终内存等于独立原始内存参考散射的编码。K、V 与 metadata 均读取初态；K stores 在前，V stores 在后。整内存公式不需要两个 cache 分离或 V 写地址注入性；逐 lane 双读回只沿用旧 hcache 和 active V hinj，没有新增正 stride、正 width、finite 或 nonNaN 限制。

根代理独立编译 **3/3 模块通过，20 条具名公理检查通过**；另有 [128 项独立语义核对](independent-semantic-review/receipt.json)。仅新增 **1 条公开 specification**；20 次公理检查含已有 source_projection，不等于 20 条新增公开证明。

非空见证覆盖 fp16 K/bf16 V、KCache 与 V 输入交叉 alias、两个 active lane 加 masked tail、NaN 载荷和负零；同 cache 见证检查最后 V 写入，另覆盖全部 inactive、width0 和单 active 零 V stride。完整 frame 保留不同 tag 的未写单元。

证明复用已审查的 kcache 编码；该依赖的源、编译对象和回执与新模块一起绑定。根代理编译见 [root-replay/build-records.json](root-replay/build-records.json)，全部精确具名公理见 [axiom-review.json](axiom-review.json)，原始候选见 [candidate/README.zh-CN.md](candidate/README.zh-CN.md)。

原 Python full static loop、GPU 字节布局/机器索引/跨 program 并发仍未覆盖。此包先冻结证明，统计升级另由 metadata 包记录；没有 GPU 运行。冻结候选里“需追加三条显式 Lake 登记”是整合说明错误，现有 kv_cache_copy.+ 已覆盖，应以实际 registration 检查为准，见 [integration-notes.json](integration-notes.json)。

# masked_select：原始位存储证明

新增 1 条公开合同覆盖原接受的完整七语句 mask/prefix 消费阶段。五种浮点字格式与 int64 payload transport 均有位保持证明；原始 RawCell.int 仍为任意 Int，未添加 int64 数值范围限制。fp32 实际走 BitVec32 的 real 通道，旧任意 R/七注解合同另保留，fp64 不计为 native64。

独立参考只读取初始内存中的 selection、prefix 和 input payload。最终整个编码内存、成功执行、decode、canonical image 和完整 frame 均有证明；整个内存等式不要求地址注入性，逐 lane 读回只沿用原 active hinj。输出与三种输入的别名、masked 尾部、零长度、全部 inactive 和非同质 tagged memory 均保留。

根代理独立编译 **3/3 模块通过、25 条具名公理检查通过**，另有 [162 项独立审查](independent-semantic-review/receipt.json)。25 次检查包含已有 TypedCopy 声明，仅新增 1 条公开 specification。第一次根代理重放在解析 KV 依赖时停止，没有调用 Lean 或改工作区；修正完整依赖根后实际通过，首次记录保留于 root-replay-history。

候选实际复用 KV 的 encode_masked_scatter 与 kcache 原始字内存桥；父包回执、实际第一命中对象和直接依赖 before/after 已绑定。编译记录见 [root-replay/build-records.json](root-replay/build-records.json)，具名检查见 [axiom-review.json](axiom-review.json)，语义复核见 [独立报告](independent-semantic-review/REVIEW.zh-CN.md)。

prefix 仍是旧 Nat 模型：0 减 1 饱和为 0，不等于机器 signed64 的 −1。这一边界没有被实验假设掩盖。host cumsum、机器地址溢出、物理字节布局、编译器/GPU 搬运及跨 program 并发仍未闭合。本包冻结软件字表示证明，统计接入另由 metadata 包记录，没有 GPU 运行。

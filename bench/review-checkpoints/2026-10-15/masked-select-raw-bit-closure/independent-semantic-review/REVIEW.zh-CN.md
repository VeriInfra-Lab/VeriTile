# masked_select 原始位表示桥独立审查

结论：未发现语义阻断。候选增加 3 个模块和 1 个公开 specification，覆盖原已接受的完整七语句 mask/prefix 消费阶段；没有把完整范围缩成固定样例。旧 public、TypedCopy 与 Python 源哈希保持原冻结值。本审查只读，不修改工作区或登记分类，不重新编译证明体。

162 项工件、声明、依赖和语义表面核对通过；另实际运行候选只读 verifier，通过其 163 个旧项目导入、36 个外部直接导入及父包校验。3 份最终编译记录均 exit 0，source/object/compiler/log SHA 和直接依赖 before/after 一致；按 ordered roots 第一命中对象逐项核验。25 个命名标准公理日志与源码中的指令逐项对应：RawBitCopy 11、RawBitCorrect 9、RawBitWitness 5。本审查不将这些记录说成全库重新构建。

## 值与完整范围

`reference` 以初始 RawCell memory 为唯一读来源：所有 selection、prefix 和 input payload 都使用初态 m；只有 scatter 的输出累积器逐步更新。原 typed source 的三个 load 都先于 store，故输出即使与 input、prefix 或 selection 别名，也不影响后续 lane 的控制或 payload。参考没有调用源执行或引用最终输出。

`execution_encoded_memory` 对任意 backend rest、Storage、参数、RawCell 内存及 seed 证明完整实际 source 的最终 memory 等于 `encodeMemory reference`。该完整存储定理不要求地址注入性或区域分离。公开声明的逐 lane 读回只要求原来的 selected 且 in-bounds 地址注入性；未选中及越界 lane 可重复 prefix。没有新增正 width、正 prefix、不同 region、有限浮点数或同质内存前提。完整 frame 对未被活跃 lane 写到的所有 tagged cells 保持原值。

继承的 KV `encode_masked_scatter` 是对任意 list、mask、offset、payload 的归纳证明，不要求地址唯一；不是未经证明的假设。此次 RawBitCopy 的实际直接 import、KV 对象哈希及 KV/kcache 父包 receipt 都已绑定。不能只归档本任务 3 个对象而漏此父依赖。

RawCell 的 `.int (value : Int)` 保留无界 Int 域；readInt/encode/decode 均原值传递，没有 truncation 或 int64 范围条件。`six_format_raw_bit_correctness` 的 memory 也仍允许任意 Int cell。`payload .int64` 的 BitVec64.toInt 和 `every_int64_bits` 的 BitVec.ofInt 逆律仅说明所有 64 位模式的 signed payload transport；它们没有把一般 Int 域偷换成有限子集。五个浮点格式为 fp16/bf16/fp32/f8e4/f8e5，实际 fp32 明确走 `RawCell.real(BitVec32)`。ordinary legacy 定理保留任意 R 与全部 7 种旧注解；fp64 的 erased real 通道不被计为 native64 load。

Canonical 是最终编码 image 的结论，不是初始 well-typed 限制。错 tag 读取的默认规则通过 read_bool_encode/read_int_encode/read_payload_encode 交换；反向 encode/decode 逆律只在 image 上使用。任意 rest、seed regs/undef/numPids 对最终 memory 的独立性已证明，只有实际参与寻址的 pids 必须相同。

## 实例与边界

别名 witness 对任意 output region 成立。lane 0/2 活跃，lane 1 不选且 prefix 重复，lane 3 selection 为真但越界；先写地址 2，再写地址 3。分别选择 output=input、prefix、selection，第一次写会破坏第二活跃 lane 的原 payload、prefix 或 mask，但第二次写仍得到初态负零；第一次写保留 signaling-NaN payload。邻接未写单元有 frame。另有 signed64 两端点、全 inactive、width=0 的实际成功执行。

prefix 仍是旧 Nat 模型：prefix=0 时减1饱和为0；原机器 signed64 减法可得到−1，两者不是全域等价。本候选明确保留这一边界，不把它列为可由实验普遍证明的假设。host cumsum、机器指针/overflow、物理字节布局/端序/native64、编译器与 GPU 数据搬运、跨 program 并发均未闭合。本次成立的是旧接受阶段的软件字存储表示证明。

现有 `masked_select.+` 递归登记覆盖这 3 个新文件；集成仍须运行真实 collector 验证增量集合。候选新增 1 个 public，其他 witness/辅助结果不应重复计数。是否更新当前分类由后续 root 元数据审查决定，本目录没有改 registry。

复验：`python3 -B /tmp/masked-select-raw-bit-independent-review-20261015/review.py`。脚本只读候选及依赖，在本审查目录更新检查结果；封存后如需重放，应先复制本目录到新的输出路径，不覆盖本次 receipt 绑定工件。

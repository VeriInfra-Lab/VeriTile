# KV-cache raw-bit 表示闭合候选

本包新增三个 Lean 模块、一个公开 specification，复用已冻结 kcache raw-bit 编码包（receipt `cd6e3aa274b463fa7f24b0dae39fbd2aa7dc87467ad761d9d743c19b84af588d`）。仅在私有目录编译，不改工作区、旧证明、共享核心、Lake 注册或分类。

`RawBitCorrect.mixed_five_format_raw_bit_correctness` 覆盖旧合同完整 masked single x-block：K/V 各选五个具名机器格式 fp16/bf16/fp32/f8e4/f8e5，共 25 个独立组合，每组内部源与缓存类型相同。所有旧 Nat 参数、splitX、pids、width=0、stride=0、blockSize=0 的既有模型语义均保留。逐 lane 的 K 读回仅保留旧 `KCache ≠ VCache` 条件；V 读回仅保留 active 地址注入性，没有改成正 stride。单 active lane、零 V stride 的见证专门检查这一点。

独立 `referenceK` 从初始 raw memory 读取 K 及 metadata 后写 K-cache；`reference` 再写 V-cache，但 V 值、metadata 和地址仍只读同一初态。`execution_encoded_memory` 对实际完整旧 typed source 证明整个最终 memory 等于 `encodeMemory reference`，并且不要求 cache 分离或注入性。因此输入输出交叉 alias、输出输出 alias、重复 V 地址都遵守原先 K store 后 V store 的顺序。逐 lane 读回才需要原地址条件；frame 覆盖两个 active 写集合之外的完整 tagged cell。

复用 RawCell 的机器宽度是 8/16/32，NaN payload 与 ±0 不做数值投影。`decode_encode`、`encode_injective`、存储默认读取和写入/整 scatter 交换形成实际表示桥。`Canonical` 是编码 image 的输出结论，不是新的输入数值限制；任意 RawCell memory 都允许。这里不是把任意非 canonical Structural carrier 宣称成有限原始字的双射。

本源 fp32 实际读取 `.real` channel，使用 `RawCell.real(BitVec32)`；机器格式 payload 映射与逐位读回直接复用已证明的映射。旧六种注解的 36 组合另由 `legacy_raw_carrier_correctness` 对任意 R 保留。名为 fp64 的注解仍是 abstract `.real`，不是第六物理格式，也不是 native64 加载证明；不承诺混合 fp32/native64 ABI。

`backend_independent` 与 `registers_undef_independent` 证明最终完整内存与任意 rest、初态 regs/undef/numPids 无关（相同 pids）。masked-off load 的字面量和 cast 可以不同，active store 不观察它们；没有假设默认值必须为0。公开合同使用固定 inert 仅填不影响观察结果的 callback，不含数值实验前提。

五个辅助见证/边界定理：

- `mixed_cross_alias_masked_source_success`：fp16 K 写入 V 输入区，后续 V 仍写出原 bf16 NaN payload/−0；两 active、一 inactive lane，并检查不活跃目标的 Bool/Nat cell frame。
- `same_output_cache_v_writes_last`：同一个输出缓存中，V 最后覆盖 K，不能伪称两者都最终读回。
- `inactive_source_success` 与 `zero_width_source_success`：实际 source 成功、全内存保持。
- `singleton_active_zero_stride`：零 V stride 在单 active lane 下保留原 injectivity 域。

三新模块完整拓扑复编与 20 个命名 axiom footprint 检查 PASS；旧依赖对象只读复用并逐 hash 对应冻结构建记录，不是全库 fresh build。没有运行 GPU。旧 `source_projection` 是对已接受 masked x-block 端口的擦除桥；原 Python 全 static x-block loop、machine int32 与 Nat metadata 的精化、字节布局/端序/设备地址溢出、跨 program 竞争、Triton/CUDA 对应仍未证明。

复核与独立重编：

```sh
python -B /tmp/veritile-kv-cache-raw-bit-closure-20261015/verify.py
python -B /tmp/veritile-kv-cache-raw-bit-closure-20261015/replay.py --output /tmp/kv-cache-raw-independent-replay
```

第一条只读核对；第二条要求全新的私有目录，复用 kcache frozen dependency 与原集成对象，只在新目录写三个对象。`installation.json` 与 `changes.patch` 为可审阅候选，未自动应用。

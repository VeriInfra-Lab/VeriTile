# kcache 完整原始位表示桥候选

四个新增 Lean 模块已定向编译通过，29 条 `#axiomsClean` 检查通过。旧源码、旧公开命题、全局登记和验收统计未修改。本包是待根代理复核的新增证明候选，不表示 GPU 正确性，也不是一次全库重编。

公开入口是 `bench/tritonbench_g/kcache_copy_triton/RawBitCorrect.lean` 的 `five_format_raw_bit_correctness`。它覆盖原来已接受的 `n_tokens = 1`、一个参数化 `splitX` 分片的完整复制程序，保留所有参数维度和地址表达式。没有把范围缩成一个样例、一个元素或 ready 阶段。

## 实质新增

旧 `TypedCopy` 对抽象载荷类型证明执行和散射。本包增加独立的、按格式位宽标记的 `RawCell` 和原始字内存。`reference` 只读取初始原始内存、参数和程序编号，按原顺序散射；其右侧不调用源程序执行器，也不引用最终状态。

新证明建立整个最终内存等于 `encodeMemory(reference ...)`，同时得到读回、全部未写单元的 frame、成功执行和输出保持编码像。它并非只比较低位解码结果。输入可以是任意 `RawCell` 内存，不要求预先满足 Canonical、不要求同一区域同一种标签，也不要求源/目标/元数据区域互不相交。

`decode(encode raw) = raw` 对所有原始单元成立。反向逆律仅对显式的 `Canonical = image encode` 成立；对任意 Structural 载荷该逆律不成立，本包提供一个含高位的非规范 half 单元反例。所有标签错配的默认读规则都有交换证明，避免用隐含的正确类型前提缩小范围。

执行结果与 `rest` 的全部数值回调无关；相同 pids 下，初始寄存器、undef 和 numPids 也不影响最终内存。公开五格式命题不留下任意数值回调。`inert` 仅填充不可达算子字段，相关独立性已证明，不能将其说成实现了算术、exp 或其他数值原语。

## 格式与范围

| 解释 | 精确载荷 | 实际源存储标签 |
|---|---|---|
| fp16 | BitVec 16 | `.fp16` |
| bf16 | BitVec 16 | `.bf16` |
| fp32 | BitVec 32 | `.real`，通过 `RawCell.real` 明确映射 |
| f8e4 | BitVec 8 | `.f8e4` |
| f8e5 | BitVec 8 | `.f8e5` |
| 历史抽象分支 | 任意 R，亦可取 BitVec 64 | `.real`，另外保留的普通定理 |

实际 fp32 源没有使用 Structural 的独立 `.fp32` 标签。独立 `RawCell.fp32` 单元仍被整个内存表示/frame 保留，但不拿它冒充本源的 fp32 读写。

历史名字为 `Storage.fp64` 的分支经过 `.real` 通道。`legacy_raw_carrier_correctness` 保留其完整抽象范围；`legacy64_payload` 证明 64 位载荷不经 32 位截断。另有 `explicit_fp64_load_not_interpreted` 直接证明当前解释器的显式 `ComputeOp.load .fp64` 返回 `none`。因此，本包不把该 legacy 结果当成第六种原生机器格式或 fp64 加载指令证明。

保留任意 Nat stride、width、blockSize、splitX 和 pids，允许 K/KCache/元数据相互 alias。width=0、stride=0、blockSize=0、length−1 截断与 Nat 除余零的行为沿用原模型；这不是宣称这些参数都是有效 GPU 启动。源程序不含 mask/other/boundary_check，本包不借用通用 masked helper 扩大覆盖。多 token 分支仍在原接受范围之外。

## 可检查见证

- 同一 region 的 `[0,1] → [1,2]` 非空重叠复制，输入为 signaling-NaN 位载荷和负零；第二输出来自初始负零，而非前一 store 写入的 NaN。
- width=0 仍有真实成功执行，全部内存不变。
- 高位超出 half 宽度的 Structural 单元使无条件反向逆律失败。
- 不同 NaN 载荷以及正负零在编码后仍可区分。

这些是通用命题的支持见证，不代替全参数证明。没有把 IEEE 数值解码到 Real，因此 NaN/±0 没有被商掉。

## 尚未建立的机器对应

这是以 `RegionName × Nat` 为地址的带类型字存储模型。没有证明 CUDA 字节打包、大小端、元素宽度乘地址、对齐、有限指针溢出或地址空间对应。原 Python 的 int32 元数据和机器索引与旧 Nat 模型的对应仍是独立边界。跨程序并发、GPU 内存和设备执行没有在本包运行或证明。对于这些边界，不用数值实验通过充当全域证明。

## 复验与证据

只读验包：

```sh
python3 -B /tmp/veritile-kcache-bit-closure-20261015/verify.py
```

在全新目录复编四模块（输出目录必须不存在）：

```sh
python3 -B /tmp/veritile-kcache-bit-closure-20261015/replay.py \
  --output /tmp/kcache-raw-bit-independent-replay
```

`build.py` 记录每条真实 Lean 命令、可执行文件 SHA、有序 LEAN_PATH、源和直接导入对象的编译前后 SHA、退出码、对象和日志 SHA。`final-build.json` 是最终四模块拓扑构建，旧失败日志保留。复编仅在新目录写对象；既有 R 对象通过只读链接复用，绝不运行 Lake，也不写 W 或 R。

`reused-import-closure.json` 另外检查旧基线记录中的项目导入闭包及对象、源和日志，并绑定直接外部导入和 Init 对象。闭包是在本包封存时从原基线记录核对的，不能说这些新展开列表是每个旧模块当年编译时重新记录的，也不能说已重新编译整个依赖库。`original-source-bindings.json` 对照已冻结 catalog 证明旧公开源保持原字节。

`installation.json` 仅给出四个新增源的建议目标和模块名。当前没有安装、没有追加 lakefile/canonical catalog，也没有变更原 41 或当前 184 项分类；该变更需要由根代理依据实际语义审阅另行整合。

# destindex_copy：完整原域的原始位搬运合同

[RawBitCorrect.lean](RawBitCorrect.lean) 的 `nine_storage_pair_raw_bit_correctness` 覆盖九种独立 fp32/fp16/bf16 组合。每一输入/输出对保持同一格式；没有异格式转换或浮点计算。fp32 位载荷使用 BitVec32，`.real` 是既有擦除存储标签；fp16/bf16 使用真实16位单元。

保留全部原 Layout 参数、无 mask padding、未使用的 head 数和 head stride、有符号 Int 目标元数据、原 Int.toNat 负地址语义、任意初始 raw cells/寄存器/undef/program id及输入输出/metadata别名。两块输入都从初态读完才执行双 store。独立参考给出完整最终内存、两个输出的位值和全帧；逐点结果仅使用原输出单射与两个输出区域不同的前提。没有数值有限性或新增 noalias 假设。

3个模块与28个具名标准公理检查通过独立复编。见[证明及独立审查](../../review-checkpoints/2026-10-15/destindex-copy-raw-bit-closure/README.md)、[P7验收](../../review-checkpoints/2026-10-15/raw-copy-software-metadata/README.md)。软件字存储不等于物理 byte ABI、fixed-width地址、内存安全、跨程序无竞争或GPU一致性；没有GPU执行结论。

[历史 README](README.md)保持原字节；原候选索引交换等价性保留，不代替新值合同。

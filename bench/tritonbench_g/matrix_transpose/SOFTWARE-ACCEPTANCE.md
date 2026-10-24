# matrix_transpose：完整 fp16 位转置合同

[FPCorrect.lean](FPCorrect.lean) 的 `matrix_transpose_bits_correctness` 直接执行既有 typedKernel 的六条语句，同时覆盖原版本和已有索引乘法交换候选。输入/输出均为原 wrapper 的 fp16；Word=BitVec16，全部NaN载荷、正负零、无穷和次正规数都直接保留。

任意原 Nat 尺寸（含零）、四个步长、初态和输入输出别名均保留。独立参考只读取初始内存；整块 load 在所有 store 之前完成。定理给出真实成功执行、完整最终内存、逐输出位值和其他所有单元的frame，唯一前提是原输出地址单射。任意两个数值backend对整次执行等价，证明惯性字段不可达；没有把实际浮点cast设为恒等。

5个模块、16个具名标准公理检查及独立参考非循环检查通过。非空2×4原地别名/rowgap样例只是辅助检查，完整范围来自参数化定理。原Python精确JIT AST的SM80/90 × 2×4、16×16、32×64共6个离线配置通过；未运行GPU。见[证明包](../../review-checkpoints/2026-10-15/matrix-transpose-raw-bit-closure/README.md)和[P7验收](../../review-checkpoints/2026-10-15/raw-copy-software-metadata/README.md)。

Nat地址模型与native signed32 stride/offset、物理字节ABI、合法分配和launch、线程布局及别名load/store顺序仍需对应。[历史 README](README.md)保持原字节。

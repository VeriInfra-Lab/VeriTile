# chunk_retention kernel entry


Public FP entry status, 2026-10-21: **comparison proof pending**.
`Misc/NativeCorrect.lean` retains the former single-source `source_fp_equiv`
result as an ordinary theorem under its original namespace, full input domain
and proof. `FPEquiv.lean` records the missing/pending comparison and declares no two-program
FP specification. Original typed-source candidate exists but its source and numeric/contextual proof have not been connected to the selected repaired KernelPacket/AddressedProgramIO staging; do not count a native implementation as pair proof.
The independent `Correct.lean` mathematical theorem and the complete native
Tensor application are preserved. This gap is a missing program comparison,
not an unproved numerical law; no new numerical experiment is claimed.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Reader.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。

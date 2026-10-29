import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Source

/-! Canonical typed operational source for the direct checked Nopad repair.
`ContextAttnNopad.kernel` retains the original floating body. Composed host-return
wrappers and their result-proof dependencies live in `Proofs.HostPrograms`,
which is exported by the public correctness entries. -/

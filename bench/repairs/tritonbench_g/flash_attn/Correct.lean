import bench.repairs.tritonbench_g.flash_attn.Contract
import bench.repairs.tritonbench_g.flash_attn.Proofs.CheckedCorrect

/-! Public independent mathematical endpoint: `Checked.whole_real` proves
complete O and L formulas from successful direct checks. Structural allocation,
signed arithmetic and absolute byte bounds are derived independently through
`Checked.structural_safe`, `formed_tensor`, `formed_prefetch`, `formed_scratch`.
Historical mathematical component APIs are retained in `Proofs/SourceCorrect`. -/

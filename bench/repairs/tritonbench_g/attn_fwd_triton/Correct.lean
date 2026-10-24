import bench.repairs.tritonbench_g.attn_fwd_triton.Contract
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedCorrect
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SourceCorrect

/-! Public independent Real endpoint: `Checked.whole_real`; complete structural
safety: `Checked.trace_safe`. Both consume the actual host checks and caller
storage, with no normalization adapter or numerical safety premise. The
historical source/host component APIs remain available in `SourceCorrect`. -/

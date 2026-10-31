import bench.tritonbench_g.softmax_flaggems.Sources.Mathematical_Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def offset (row N K n k : Nat) : Nat := row * N * K + n * K + k

def tileOffset {B C : Nat} (row N K kbase start : Nat) (i : TileIndex [B, C]) : Nat :=
  offset row N K (start + i.1.val) (kbase + i.2.1.val)

/-- Source loads omit `other`; inactive values are the initial state's actual
undefined-padding values at their computed addresses, not assumed zeros. -/
noncomputable def sample (s : BlockState) (R : RegionName) (row N K n k : Nat) : ℝ :=
  if n < N ∧ k < K then s.readMem R (offset row N K n k)
  else s.undef R (offset row N K n k)

noncomputable def tileSample {B C : Nat} (s : BlockState) (R : RegionName)
    (row N K kbase start : Nat) (i : TileIndex [B, C]) : ℝ :=
  sample s R row N K (start + i.1.val) (kbase + i.2.1.val)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

import bench.tritonbench_g.softmax_flaggems.Sources.Mathematical_Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def offset (N r n : Nat) : Nat := r*N+n

def tileOffset {B C : Nat} (N rbase start : Nat) (i : TileIndex [C,B]) : Nat :=
  offset N (rbase+i.1.val) (start+i.2.1.val)

noncomputable def sample (s : BlockState) (R : RegionName) (M N r n : Nat) : ℝ :=
  if r<M ∧ n<N then s.readMem R (offset N r n) else s.undef R (offset N r n)

noncomputable def tileSample {B C : Nat} (s : BlockState) (R : RegionName)
    (M N rbase start : Nat) (i : TileIndex [C,B]) : ℝ :=
  sample s R M N (rbase+i.1.val) (start+i.2.1.val)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner

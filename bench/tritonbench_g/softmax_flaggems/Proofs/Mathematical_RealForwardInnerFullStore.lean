import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def fullStoreMid (s : BlockState) (X : RegionName) (row N base B : Nat)
    (mx : WithBot ℝ) (z : ℝ) : BlockState :=
  let ns : Tile .nat [B] := ⟨fun i ↦ base + i.1.val⟩
  let inp := score s X row N base B
  let out := fun i ↦ WithBot.realDiv (WithBot.realExp (WithBot.realSub (inp i) mx)) (some z)
  let s₁ := s.setReg "n_offsets" .nat [B] ns
  let s₃ := s₁.setReg "inp" .real [B] ⟨inp⟩
  s₃.setReg "o" .real [B] ⟨out⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

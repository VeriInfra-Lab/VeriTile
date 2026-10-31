import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def tileActive {B C : Nat} (M N rbase start : Nat) (i : TileIndex [C,B]) : Bool :=
  decide (rbase+i.1.val<M ∧ start+i.2.1.val<N)
noncomputable abbrev scatter := @RealForwardNonInner.scatter

noncomputable def storeValue {B C : Nat} (s : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (z : Fin C → ℝ) (i : TileIndex [C,B]) : ℝ :=
  tileSample s P M N rbase start i *
    (tileSample s G M N rbase start i - z i.1)

noncomputable def storeMid {B C : Nat} (s : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (z : Fin C → ℝ) : BlockState :=
  let a := s.setReg "mask" .bool [C,B] ⟨tileActive M N rbase start⟩
  let b := a.setReg "out_tile" .real [C,B] ⟨fun i ↦ some (tileSample s P M N rbase start i)⟩
  let c := b.setReg "out_grad_tile" .real [C,B] ⟨fun i ↦ some (tileSample s G M N rbase start i)⟩
  c.setReg "in_grad_tile" .real [C,B] ⟨fun i ↦ some (storeValue s P G M N rbase start z i)⟩

noncomputable def advance {B C : Nat} (s : BlockState)
    (M N rbase start : Nat) : BlockState :=
  (s.setReg "n_offsets" .nat [B] ⟨fun i ↦ start+B+i.1.val⟩).setReg
    "offsets" .nat [C,B] ⟨tileOffset N rbase (start+B)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner

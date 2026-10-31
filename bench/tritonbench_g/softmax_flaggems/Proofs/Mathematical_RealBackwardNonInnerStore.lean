import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev tileActive := @RealForwardNonInner.tileActive
noncomputable abbrev scatter := @RealForwardNonInner.scatter

noncomputable def storeValue {B C : Nat} (s : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (z : Fin C → ℝ) (i : TileIndex [B,C]) : ℝ :=
  tileSample s P row N K kbase start i *
    (tileSample s G row N K kbase start i - z i.2.1)

noncomputable def storeMid {B C : Nat} (s : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (z : Fin C → ℝ) : BlockState :=
  let a := s.setReg "mask" .bool [B,C] ⟨tileActive N K kbase start⟩
  let b := a.setReg "out_tile" .real [B,C] ⟨fun i ↦ some (tileSample s P row N K kbase start i)⟩
  let c := b.setReg "out_grad_tile" .real [B,C] ⟨fun i ↦ some (tileSample s G row N K kbase start i)⟩
  c.setReg "in_grad_tile" .real [B,C] ⟨fun i ↦ some (storeValue s P G row N K kbase start z i)⟩

noncomputable def advance {B C : Nat} (s : BlockState)
    (row N K kbase start : Nat) : BlockState :=
  (s.setReg "offsets_n" .nat [B] ⟨fun i ↦ start+B+i.1.val⟩).setReg
    "offsets" .nat [B,C] ⟨tileOffset row N K kbase (start+B)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

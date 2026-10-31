import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerOneTileMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def oneTileMid (s : BlockState) (Y X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) : BlockState :=
  let off := tileOffset (B := B) (C := C) row N K kbase 0
  let inp := score s X row N K kbase 0 B C
  let mx := batchMax s X row N K kbase B C hB
  let z := batchSum s X row N K kbase B C hB
  let e := fun i : TileIndex [B, C] ↦ WithBot.realExp (WithBot.realSub (inp i) (mx i.2.1))
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ i.1.val⟩
  let s₂ := s₁.setReg "offset" .nat [B, C] ⟨off⟩
  let s₃ := s₂.setReg "mask" .bool [B, C] ⟨tileActive N K kbase 0⟩
  let s₄ := s₃.setReg "input_ptrs" .ptr [B, C] ⟨fun i ↦ (X, off i)⟩
  let s₅ := s₄.setReg "inp" .real [B, C] ⟨inp⟩
  let s₆ := s₅.setReg "m" .real [C] ⟨fun i ↦ mx i.1⟩
  let s₇ := s₆.setReg "e" .real [B, C] ⟨e⟩
  let s₈ := s₇.setReg "z" .real [C] (Tile.reduceSum 0 Bool.false ⟨e⟩)
  let s₉ := s₈.setReg "out" .real [B, C] ⟨fun i ↦ WithBot.realDiv (e i) (some (z i.2.1))⟩
  s₉.setReg "output_ptrs" .ptr [B, C] ⟨fun i ↦ (Y, off i)⟩

private theorem exp_finite (x : WithBot ℝ) :
    WithBot.realExp x = some (x.elim 0 Real.exp) := by
  cases x <;> rfl

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

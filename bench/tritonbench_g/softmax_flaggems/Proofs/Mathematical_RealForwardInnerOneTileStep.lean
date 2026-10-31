import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerOneTileMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def tileOffset {B : Nat} (row N : Nat) (i : TileIndex [B]) : Nat := row*N+i.1.val

def tileActive {B : Nat} (N : Nat) (i : TileIndex [B]) : Bool := decide (i.1.val<N)

noncomputable def scatter {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) : BlockState :=
  (TileShape.allIndices [B]).foldl
    (fun st i ↦ if active i then st.writeMem Y (off i) (value i) else st) s

noncomputable def midState (s : BlockState) (Y X : RegionName)
    (row N B : Nat) (hB : 0<B) : BlockState :=
  let off := tileOffset (B := B) row N
  let inp := score s X row N B
  let mx := batchMax s X row N B hB
  let z := batchSum s X row N B hB
  let e := fun i : TileIndex [B] ↦ WithBot.realExp (WithBot.realSub (inp i) mx)
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ i.1.val⟩
  let s₂ := s₁.setReg "offset" .nat [B] ⟨off⟩
  let s₃ := s₂.setReg "input_ptrs" .ptr [B] ⟨fun i ↦ (X,off i)⟩
  let s₄ := s₃.setReg "mask" .bool [B] ⟨tileActive N⟩
  let s₅ := s₄.setReg "inp" .real [B] ⟨inp⟩
  let s₆ := s₅.setReg "m" .real [] (Tile.scalar mx)
  let s₇ := s₆.setReg "e" .real [B] ⟨e⟩
  let s₈ := s₇.setReg "z" .real [] (Tile.reduceSum 0 Bool.false ⟨e⟩)
  let s₉ := s₈.setReg "out" .real [B] ⟨fun i ↦ WithBot.realDiv (e i) (some z)⟩
  s₉.setReg "output_ptrs" .ptr [B] ⟨fun i ↦ (Y,off i)⟩

private theorem exp_finite (x : WithBot ℝ) :
    WithBot.realExp x=some (x.elim 0 Real.exp) := by cases x <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile

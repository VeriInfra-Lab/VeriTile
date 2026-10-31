import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def tileOffset {B : Nat} (row N base : Nat) (i : TileIndex [B]) : Nat :=
  row * N + (base + i.1.val)

def tileActive {B : Nat} (N base : Nat) (i : TileIndex [B]) : Bool :=
  decide (base + i.1.val < N)

noncomputable def storeValue {B : Nat} (s : BlockState) (X : RegionName)
    (row N base : Nat) (mx : WithBot ℝ) (z : ℝ) (i : TileIndex [B]) : ℝ :=
  (WithBot.realDiv (WithBot.realExp (WithBot.realSub (score s X row N base B i) mx)) (some z)).unbotD 0

noncomputable def scatter {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ)
    (active : TileIndex [B] → Bool) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun st i ↦ if active i then st.writeMem Y (off i) (value i) else st) s

noncomputable def storeMid (s : BlockState) (X : RegionName) (row N base B : Nat)
    (mx : WithBot ℝ) (z : ℝ) : BlockState :=
  let ns : Tile .nat [B] := ⟨fun i ↦ base + i.1.val⟩
  let inp := score s X row N base B
  let out := fun i ↦ WithBot.realDiv (WithBot.realExp (WithBot.realSub (inp i) mx)) (some z)
  let s₁ := s.setReg "n_offsets" .nat [B] ns
  let s₂ := s₁.setReg "mask" .bool [B] ⟨tileActive N base⟩
  let s₃ := s₂.setReg "inp" .real [B] ⟨inp⟩
  s₃.setReg "o" .real [B] ⟨out⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

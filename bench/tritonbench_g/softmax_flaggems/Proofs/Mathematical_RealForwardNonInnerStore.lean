import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerDenominator
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def tileOffset {B C : Nat} (row N K kbase base : Nat) (i : TileIndex [B, C]) : Nat :=
  row * N * K + (base + i.1.val) * K + (kbase + i.2.1.val)

def tileActive {B C : Nat} (N K kbase base : Nat) (i : TileIndex [B, C]) : Bool :=
  decide (base + i.1.val < N ∧ kbase + i.2.1.val < K)

noncomputable def storeValue {B C : Nat} (s : BlockState) (X : RegionName)
    (row N K kbase base : Nat) (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ)
    (i : TileIndex [B, C]) : ℝ :=
  (WithBot.realDiv
    (WithBot.realExp (WithBot.realSub (score s X row N K kbase base B C i) (mx i.2.1)))
    (some (dz i.2.1))).unbotD 0

noncomputable def scatter {B C : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B, C] → Nat) (value : TileIndex [B, C] → ℝ)
    (active : TileIndex [B, C] → Bool) : BlockState :=
  (TileShape.allIndices [B, C]).foldl
    (fun st i ↦ if active i then st.writeMem Y (off i) (value i) else st) s

noncomputable def storeMid {B C : Nat} (s : BlockState) (X : RegionName)
    (row N K kbase base : Nat) (mx : Fin C → WithBot ℝ) (dz : Fin C → ℝ) : BlockState :=
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ base + i.1.val⟩
  let s₂ := s₁.setReg "offsets" .nat [B, C] ⟨tileOffset row N K kbase base⟩
  let s₃ := s₂.setReg "mask" .bool [B, C] ⟨tileActive N K kbase base⟩
  let s₄ := s₃.setReg "inp" .real [B, C] ⟨score s X row N K kbase base B C⟩
  s₄.setReg "o" .real [B, C] ⟨fun i ↦ WithBot.realDiv
    (WithBot.realExp (WithBot.realSub (score s X row N K kbase base B C i) (mx i.2.1)))
    (some (dz i.2.1))⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

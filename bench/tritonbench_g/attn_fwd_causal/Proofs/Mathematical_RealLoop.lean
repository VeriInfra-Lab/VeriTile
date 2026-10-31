import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealIteration
import bench.tritonbench_g.attention_fwd_triton3.Sources.Mathematical_RealKeys

/-! Full execution of either source helper range from its loop-entry
register contract. Bounds, dimensions, strides and partial tails stay
symbolic. The final carry is the independent closed formula over every
visited tile, including zero-padded lanes and finite-penalty lanes. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry closed closed_zero closed_succ)
noncomputable section
set_option maxHeartbeats 1000000

def cursor (lower BN i : Nat) : Nat := lower+i*BN

theorem cursor_succ (lower BN i : Nat) : cursor lower BN (i+1) = cursor lower BN i+BN := by
  simp [cursor, Nat.add_mul, Nat.add_assoc]

theorem cursor_lt (lower upper BN i : Nat) (hN : 0 < BN) :
    cursor lower BN i < upper ↔ i < RealFormula.tripCount lower upper BN := by
  have hc : RealFormula.tripCount lower upper BN =
      AttentionFwdTriton3.RealKeys.blockCount lower upper BN := by
    by_cases h : lower < upper
    · simp only [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, if_pos h]
      rw [show upper-lower+BN-1 = (upper-lower-1)+BN from by omega, Nat.add_div_right _ hN]
    · simp [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, h,
        Nat.sub_eq_zero_of_le (show upper ≤ lower by omega), Nat.div_eq_of_lt (show BN-1 < BN by omega)]
  simpa only [cursor, hc] using (AttentionFwdTriton3.RealKeys.iteration_lt_count lower upper BN i hN).symm

def score {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (i : Nat) : Fin BM → Fin BN → ℝ :=
  RealFormula.logit xs c.startM c.context c.activeHead ⟨cursor lower BN i,second || firstDiagonal⟩

def value {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (i : Nat) : TileIndex [BN,BD] → ℝ :=
  fun j => RealFormula.loadedValue xs c.context c.activeHead ⟨cursor lower BN i,second || firstDiagonal⟩ j.1 j.2.1

def after {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (lower : Nat) (second firstDiagonal : Bool) (seed : Carry BM BD) (i : Nat) : Carry BM BD :=
  closed seed (score xs c lower second firstDiagonal) (fun _ => RealTiles.keep)
    (value xs c lower second firstDiagonal) i

structure Registers {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (carry : Carry BM BD) (start : Nat) (second : Bool) (s : BlockState) : Prop where
  query : s.regs .real [BM,BD] "q" = some (RealTiles.query xs c.startM c.context c.activeHead)
  queryScale : s.regs .real [] "q_scale" = some (Tile.scalar (some xs.queryScale))
  rows : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  carry : Carries s carry
  pointers : Pointers s c BN BD start second
  undef : ∀ r a, s.undef r a = 0

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop

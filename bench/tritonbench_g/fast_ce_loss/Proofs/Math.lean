import bench.tritonbench_g.fast_ce_loss.Proofs.Mathematical_RealMath

namespace VeriTile.Bench.Repairs.FastCeLoss.Math
open Finset
noncomputable section
set_option maxHeartbeats 2000000

/-- Independent logical logits. Zero scalar arguments disable the corresponding
branch; negative nonzero arguments retain their ordinary mathematical meaning. -/
def scale (s : ℝ) : ℝ := if s=0 then 1 else s

def transformed (c s x : ℝ) : ℝ :=
  if c=0 then scale s*x else c*Real.tanh (scale s*x/c)

def derivative (c s x : ℝ) : ℝ :=
  scale s * (if c=0 then 1 else 1-(Real.tanh (scale s*x/c))^2)

def lse {V : Nat} (c s : ℝ) (x : Fin V → ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (transformed c s (x j)))

def rowLoss {V : Nat} (c s : ℝ) (label : Option (Fin V)) (x : Fin V → ℝ) : ℝ :=
  match label with
  | none => 0
  | some k => lse c s x - transformed c s (x k)

def loss {R V : Nat} (c s denominator : ℝ)
    (labels : Fin R → Option (Fin V)) (x : Fin R → Fin V → ℝ) : ℝ :=
  (∑ r, rowLoss c s (labels r) (x r))/denominator

def gradient {R V : Nat} (c s denominator upstream : ℝ)
    (labels : Fin R → Option (Fin V)) (x : Fin R → Fin V → ℝ)
    (row : Fin R) (col : Fin V) : ℝ :=
  match labels row with
  | none => 0
  | some k => (upstream/denominator) *
      (Real.exp (transformed c s (x row col)-lse c s (x row)) - if col=k then 1 else 0) *
      derivative c s (x row col)

def count {R V : Nat} (labels : Fin R → Option (Fin V)) : Nat :=
  (univ.filter (fun r => (labels r).isSome)).card

def denominator {R V : Nat} (labels : Fin R → Option (Fin V)) (items : Option ℝ) : ℝ :=
  items.getD (count labels)

/-- A chunk includes logical vocabulary entries only. This formula is independent
of padding, physical storage, reduction order and any implementation. -/
def chunkMass {V : Nat} (block chunk : Nat) (f : Fin V → ℝ) : ℝ :=
  ∑ j, if j.val/block=chunk then Real.exp (f j) else 0

theorem sum_chunkMass {V : Nat} (block chunks : Nat) (f : Fin V → ℝ)
    (covered : ∀ j : Fin V, j.val/block<chunks) :
    ∑ c ∈ range chunks, chunkMass block c f = ∑ j, Real.exp (f j) := by
  unfold chunkMass
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  simp [covered j]

theorem chunkMass_pos {V : Nat} (block chunk : Nat) (f : Fin V → ℝ)
    (nonempty : ∃ j : Fin V, j.val/block=chunk) : 0<chunkMass block chunk f := by
  rcases nonempty with ⟨j,hj⟩
  apply Finset.sum_pos' (fun k _ => by split_ifs <;> positivity)
  exact ⟨j,mem_univ _,by simp [hj,Real.exp_pos]⟩

theorem chunk_lse_combination {V : Nat} (block chunks : Nat) (f : Fin V → ℝ)
    (covered : ∀ j : Fin V, j.val/block<chunks)
    (nonempty : ∀ c<chunks, ∃ j : Fin V, j.val/block=c) :
    Real.log (∑ c ∈ range chunks, Real.exp (Real.log (chunkMass block c f))) =
      Real.log (∑ j,Real.exp (f j)) := by
  congr 1
  calc
    _ = ∑ c ∈ range chunks, chunkMass block c f := by
      apply Finset.sum_congr rfl
      intro c hc
      exact Real.exp_log (chunkMass_pos block c f (nonempty c (mem_range.mp hc)))
    _ = _ := sum_chunkMass block chunks f covered

theorem source_transform (c s x : ℝ) :
    VeriTile.Bench.TritonBenchG.FastCeLoss.RealMath.transform (decide (c≠0)) (decide (s≠0)) c s x =
      transformed c s x := by
  by_cases hc : c=0 <;> by_cases hs : s=0 <;>
    simp [VeriTile.Bench.TritonBenchG.FastCeLoss.RealMath.transform,transformed,scale,hc,hs]

theorem scaled_chain (c s x p u d : ℝ) :
    (u/d) * ((p * scale s) * (if c=0 then 1 else 1-Real.tanh (scale s*x/c)*Real.tanh (scale s*x/c))) =
      (u/d) * p * derivative c s x := by
  unfold derivative
  split_ifs <;> ring

#axiomsClean sum_chunkMass
#axiomsClean chunk_lse_combination
#axiomsClean source_transform
#axiomsClean scaled_chain
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Math

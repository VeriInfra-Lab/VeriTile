import bench.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableSyntax

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem tree_width_positive {B : Nat} (tree : ReductionTree B) : 0 < B := by
  induction tree with
  | leaf i => exact Nat.lt_of_le_of_lt (Nat.zero_le i.val) i.isLt
  | node l r hl _ => exact hl

theorem range_index (D B k : Nat) (hB : 0 < B) : k < blockCount D B ↔ k*B < D := by
  constructor
  · intro h
    change k < (D+B-1)/B at h
    have bound := (Nat.le_div_iff_mul_le hB).mp (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at bound
    omega
  · intro h
    have cover := block_count_covers D B hB
    by_contra hn
    have product := Nat.mul_le_mul_right B (Nat.le_of_not_gt hn)
    omega

theorem phase_run (M : Algebra α) (phase : Bool)
    (s t : State α) (X : RegionName) (S D B k : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean) :
    ∃ final, run M (phaseBody phase X S D B rt)
        (t.setReg "roffset" .nat [] (fun _ => k*B))=some final ∧
      Ready phase s final (advance M phase s X D B k mean values) mean := by
  obtain ⟨final,ran,ready⟩ := chunk_run M phase s t X D B k values mean h
  have initial : run M [.assign .nat [] "roffset" (.alg (.constNat (k*B)))] t =
      some (t.setReg "roffset" .nat [] (fun _ => k*B)) := by
    simp [run,step,evalExpr,evalOp_unfold]
  rw [chunk,run_append,initial,Option.bind_some] at ran
  exact ⟨final,ran,ready⟩

/-- The original range stops at D, advances roffset by B, and performs exactly
its reached iterations. No synthetic k loop replaces the executable source. -/
theorem phase_loop (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B start count : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean)
    (last : start+count=blockCount D B) :
    ∃ final, FP.Structural.range M "roffset" (start*B) D B
        (phaseBody phase X S D B rt) t=some final ∧
      Ready phase s final (recurrence M phase s X D B mean start count values) mean := by
  have hB := tree_width_positive rt
  induction count generalizing start t values with
  | zero =>
    have halt : ¬start*B < D := by
      rw [←range_index D B start hB]
      omega
    exact ⟨t,by rw [FP.Structural.range,if_neg (by omega),if_neg halt],h⟩
  | succ count ih =>
    have active : start*B < D := (range_index D B start hB).mp (by omega)
    obtain ⟨u,hu,hp⟩ := phase_run M phase s t X S D B start rt values mean h
    obtain ⟨v,hv,hq⟩ := ih u (start+1) _ hp (by omega)
    refine ⟨v,?_,hq⟩
    rw [FP.Structural.range,if_neg (by omega),if_pos active,hu]
    simpa only [Option.bind_some,Nat.add_mul,Nat.one_mul] using hv

#axiomsClean range_index
#axiomsClean phase_run
#axiomsClean phase_loop
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable

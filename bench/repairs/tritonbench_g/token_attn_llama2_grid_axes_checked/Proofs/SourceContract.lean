import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Scatter
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceExecution
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceReference
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceCorrectness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open SourceExecution
open VeriTile.Bench.Repairs.TokenAttnLlama2.Scatter (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- The only reached signed loop index is zero; this projection preserves
all of the nonnegative address products and sums exactly. -/
@[simp] theorem nat_query_address (a b c d e f : Nat) :
    ((a : Int)*b + (c : Int)*d + (e : Int)*f).toNat = a*b+c*d+e*f := by
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

/-- The independently defined gather/reduce/scale reference is exactly the
memory effect of successful source execution. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  by_cases hb : blockActive p s
  all_goals simp +contextual [execution, written, hb, bodyExecution, preExecution,
    masked_scatter_memory, score, products, queryValue, keyValue, queryOffset,
    keyOffset, keyLocation, outputOffset, active, windowIndex, token, length,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, integralFloorDiv, integralMod, sourceBackend]
  all_goals try delta products bop
  all_goals simp +contextual [queryValue, keyValue, queryOffset, keyOffset, keyLocation,
    active, windowIndex, token, length, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

theorem written_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hw : WritesApart p s) (i : Index p) (hb : blockActive p s) (hi : active p s i) :
    (written M p s).mem p.output (outputOffset p s i) = .mk .fp32 (score M p s i) := by
  simp only [written, if_pos hb]
  exact masked_readback_active _ _ _ _ _ i hi hw

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (r : RegionName) (o : Nat)
    (h : ¬ blockActive p s ∨ r ≠ p.output ∨ ∀ i, active p s i → o ≠ outputOffset p s i) :
    (written M p s).mem r o = s.mem r o := by
  by_cases hb : blockActive p s
  · simp only [written, if_pos hb]
    apply State.masked_scatter_frame
    rcases h with h | h | h
    · exact (h hb).elim
    · exact Or.inl h
    · exact Or.inr (fun i _ hi => h i hi)
  · simp [written, hb]

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hw : WritesApart p s) :
    ∃ final, FP.Structural.exec (sourceBackend M) (source p) s = some final ∧ Post p s final (score M p s) := by
  refine ⟨(execution M p s).val, (execution M p s).property, ?_, ?_⟩
  · intro i hb hi
    rw [execution_memory]
    exact written_readback M p s hw i hb hi
  · intro r o h
    rw [execution_memory]
    exact written_frame M p s r o h

#axiomsClean execution_memory
#axiomsClean written_readback
#axiomsClean written_frame
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceCorrectness

import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.Pair
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.TensorBridge

/-! Optional application of the exact program pair to the retained native byte
observer. No new compiler agreement, Tensor laws, host admission or numeric
assumption is required. The main entries do not import this module. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.UnitSplitBytes
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

abbrev geometry := VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry

def execution (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  Pair.executeUnitSplit rest plans (geometry t s) (List.range (geometry t s).count)
    (entry (geometry t s) (indices t before))

def finalMemory (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  (execution rest plans t s before).getD (entry (geometry t s) (indices t before))

def sourceStores (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (before : Raw.State) : Raw.State :=
  stores Raw.ops (key t s)
    (fun q => Native.eval ops schedule (bits t before)
      ((finalMemory rest plans t s before "Y" (offset (geometry t s) q)).read .real))
    (work t s before) before

theorem execution_eq (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :
    execution rest plans t s before=
      ByteSeparated.execution rest plans t s before := by
  exact Pair.execute_same rest plans (geometry t s) _
    (fun q hq => List.mem_range.mp hq) _

theorem finalMemory_eq (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :
    finalMemory rest plans t s before=ByteSeparated.finalMemory rest plans t s before := by
  unfold finalMemory ByteSeparated.finalMemory
  rw [execution_eq]

/-- Equality of complete raw byte states, including all active output words,
sentinel rows, complete read backings, backing tails and every exterior byte.
The observer evaluates the candidate's actual final source memory; the native
primitive and fusion/tree choices are exactly the retained original choices. -/
theorem sourceStores_eq (rest : Algebra Word) (ops : Native.ScalarOps)
    (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) :
    sourceStores rest ops schedule plans t s before=
      ByteSeparated.sourceStores rest ops schedule plans t s before := by
  unfold sourceStores ByteSeparated.sourceStores ByteSeparated.word
  rw [finalMemory_eq]

#axiomsClean execution_eq
#axiomsClean sourceStores_eq
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.UnitSplitBytes

import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyComplete
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOResults

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOCorrect
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section

theorem graph (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (order : List Nat)
    (all : order.Perm (List.range g.count)) :
    (io rest plans g order).ImplementsOnCells (domain g raw) (graphTarget plans g raw) := by
  intro vals hd limits hb initial hin
  subst vals
  refine ⟨?_,IOResults.complete rest plans g raw v order all initial hin⟩
  exact IOSafety.schedule_safe rest plans g raw v limits
    (IOBinding.bounds rest plans g order _ limits hb) initial.mem order [] initial
    (IOBinding.start rest plans g raw order initial hin)
    (fun q hq => List.mem_range.mp (all.mem_iff.mp hq))
    (all.nodup_iff.mpr List.nodup_range) (by simp)

theorem real (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (data : RegionName → Nat → ℝ)
    (order : List Nat) (all : order.Perm (List.range g.count)) :
    (io rest plans g order).ObservesOnCells (realObservation data g)
      (domain g raw) (realTarget data g raw) := by
  have observed := ProgramIO.ObservesOnCells.of_implementsOnCells
    (graph rest plans g raw v order all) (realObservation data g)
  have target : (fun vals => (io rest plans g order).mapOutputs (realObservation data g)
      (graphTarget plans g raw vals))=realTarget data g raw := by
    funext vals o j
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change Real.eval data (graphResult plans g raw j.1.val j.2.1.val)=
      Real.target data g raw j.1.val j.2.1.val
    exact IOResults.real_target plans g raw v data _ _
  exact (congrArg ((io rest plans g order).ObservesOnCells
    (realObservation data g) (domain g raw)) target).mp observed

theorem native (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (bits : RegionName → Nat → Native.Bits) (order : List Nat)
    (all : order.Perm (List.range g.count)) :
    (io rest plans g order).ObservesOnCells (nativeObservation ops schedule bits g)
      (domain g raw) (nativeTarget ops schedule plans g raw bits) := by
  have observed := ProgramIO.ObservesOnCells.of_implementsOnCells
    (graph rest plans g raw v order all) (nativeObservation ops schedule bits g)
  have target : (fun vals => (io rest plans g order).mapOutputs (nativeObservation ops schedule bits g)
      (graphTarget plans g raw vals))=nativeTarget ops schedule plans g raw bits := by
    funext vals o j
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rfl
  exact (congrArg ((io rest plans g order).ObservesOnCells
    (nativeObservation ops schedule bits g) (domain g raw)) target).mp observed

#axiomsClean graph
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOCorrect

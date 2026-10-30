/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryWhole
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig backwardOutputMemory matrixTranspose)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section

/-- The original backward shell's two helper calls, expanded after its
program-index prologue. Caller-computed indices remain explicit in c. The
barrier is retained, with the existing single-program source semantics. -/
def source (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  .mk [c.q,c.k,c.v,c.do_] [dq,c.dk,c.dv]
    ((FPQuery.source c dq).surfaceBody ++ [.effectMarker "tl.debug_barrier"] ++
      (FPStages.backwardKernel c).surfaceBody)

/-- The DK/DV helper observes memory after DQ's stores, including aliases.
This reference is built from input-memory recurrences and ordered stores;
it never executes the kernel or assumes an already computed output. -/
def memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) : State α :=
  let afterQuery := FPQuery.outputMemory M c dq s (FPQuery.values M D c s)
  backwardOutputMemory M c afterQuery (FPStages.backwardValues M D c afterQuery)

theorem backward_values_congr {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (hm : s.mem = t.mem) (hu : s.undef = t.undef) :
    FPStages.backwardValues M D c s = FPStages.backwardValues M D c t := by
  cases s with
  | mk sm sr sp sn su =>
    cases t with
    | mk tm tr tp tn tu =>
      dsimp at hm hu
      cases hm
      cases hu
      rfl

theorem backward_memory_congr {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (hm : s.mem = t.mem) (hu : s.undef = t.undef) :
    (backwardOutputMemory M c s (FPStages.backwardValues M D c s)).mem =
      (backwardOutputMemory M c t (FPStages.backwardValues M D c t)).mem := by
  rw [backward_values_congr M D c s t hm hu]
  unfold backwardOutputMemory
  apply scatter_memory_congr
  apply scatter_memory_congr
  exact hm

theorem backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (source c dq) s = some final ∧
      final.mem = (memory M D c dq s).mem := by
  obtain ⟨query, hq, hqm, hqu⟩ := FPQuery.query_fp32 M D c dq s
  obtain ⟨final, hf, hfm⟩ := FPStages.backward_fp32 M D c query
  refine ⟨final, ?_, ?_⟩
  · change run (withDot M D) _ s = some final
    rw [run_append, run_append]
    have hqr : run (withDot M D) (FPQuery.source c dq).surfaceBody s = some query := by
      simpa only [FP.Structural.exec, FPQuery.source, ComputeKernel.surfaceBody] using hq
    have hfr : run (withDot M D) (FPStages.backwardKernel c).surfaceBody query = some final := by
      simpa only [FP.Structural.exec, FPStages.backwardKernel, FPStages.backwardSource,
        ComputeKernel.surfaceBody] using hf
    simp only [hqr, run, step, bind, Option.bind, hfr]
  · rw [hfm]
    exact backward_memory_congr M D c query _ hqm
      (hqu.trans (FPQuery.outputMemory_undef M c dq s (FPQuery.values M D c s)).symm)

#axiomsClean backward_values_congr
#axiomsClean backward_memory_congr
#axiomsClean backward_fp32
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward

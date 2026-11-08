/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.CallerArguments

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig backwardOutputMemory matrixTranspose)
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section

/-- The complete pasted parent function, including its own casts, PID decode,
both original helper calls and the intervening barrier. -/
def source (c : BackwardConfig) (dq : RegionName) : ComputeKernel := Caller.kernel c dq

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

theorem query_values_congr {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s t : State α) (hm : s.mem=t.mem) (hu : s.undef=t.undef) :
    FPQuery.values M D c s=FPQuery.values M D c t := by
  cases s with
  | mk sm sr sp sn su =>
    cases t with
    | mk tm tr tp tn tu =>
      dsimp at hm hu
      cases hm
      cases hu
      rfl

theorem query_memory_congr {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (hm : s.mem=t.mem) (hu : s.undef=t.undef) :
    (FPQuery.outputMemory M c dq s (FPQuery.values M D c s)).mem=
      (FPQuery.outputMemory M c dq t (FPQuery.values M D c t)).mem := by
  rw [query_values_congr M D c s t hm hu]
  unfold FPQuery.outputMemory
  apply scatter_memory_congr
  exact hm

/-- Ordered execution of the actual parent source. Its decoded indices and
int32 bounds are internal launch facts, established by the public shape guards. -/
theorem backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α)
    (indices : Caller.Indices c s) (bounds : BackwardIndexBounds c) (positive : 0<c.BTS) :
    ∃ final, FP.Structural.exec (withDot M D) (source c dq) s=some final ∧
      final.mem=(memory M D c dq s).mem := by
  obtain ⟨decoded,hp,hd,registers⟩:=Caller.prologue_run (withDot M D) c dq s indices
  obtain ⟨queryArgs,hqa,hqargs⟩:=Caller.query_arguments_run (withDot M D) c dq decoded registers
  obtain ⟨query,hq,hqm,hqu,frame⟩:=FPQuery.tail_run M D c dq decoded queryArgs hqargs bounds.toHelperIndexBounds
  obtain ⟨keyArgs,hka,hkargs⟩:=Caller.key_arguments_run (withDot M D) c dq query
    (Caller.registers_after_query registers frame)
  obtain ⟨final,hf,hfm⟩:=FPStages.backward_tail_run M D c query keyArgs hkargs bounds positive
  refine ⟨final,?_,?_⟩
  · change run (withDot M D) (Caller.kernel c dq).surfaceBody s=some final
    rw [Caller.source_split]
    simp only [run_append,hp,hqa,Caller.query_tail_eq,hq,run,step,bind,pure,Option.bind,
      hka,Caller.key_tail_eq,hf]
  · rw [hfm]
    apply backward_memory_congr M D c query (FPQuery.outputMemory M c dq s (FPQuery.values M D c s))
    · exact hqm.trans (query_memory_congr M D c dq decoded s hd.mem hd.undef)
    · simpa only [FPQuery.outputMemory_undef] using hqu.trans hd.undef

#axiomsClean query_values_congr
#axiomsClean query_memory_congr
#axiomsClean backward_values_congr
#axiomsClean backward_memory_congr
#axiomsClean backward_fp32
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward

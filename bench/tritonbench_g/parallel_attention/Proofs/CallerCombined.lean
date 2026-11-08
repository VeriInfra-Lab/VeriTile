import bench.tritonbench_g.parallel_attention.Proofs.CallerBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages (scatter_memory_congr)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000

/-- Decompose the actual caller, including its own PID decoder and barrier. -/
theorem source_split (c : QueryConfig) :
    (kernel c).surfaceBody=callerPrefix c++queryCode c++[.effectMarker "tl.debug_barrier"]++backwardCode c:=rfl

theorem query_values_mem (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (s t : State α) (same : s.mem=t.mem) : queryValues M D c s=queryValues M D c t := by
  cases s with
  | mk sm sr sp sn su =>
    cases t with
    | mk tm tr tp tn tu =>
      cases same
      rfl

theorem backward_values_mem (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (s t : State α) (same : s.mem=t.mem) : backwardValues M D c s=backwardValues M D c t := by
  cases s with
  | mk sm sr sp sn su =>
    cases t with
    | mk tm tr tp tn tu =>
      cases same
      rfl

theorem query_memory_mem (M : Algebra α) (c : QueryConfig) (s t : State α)
    (values : TileIndex [c.BTL,c.BK] → α) (same : s.mem=t.mem) :
    (queryOutputMemory M c s values).mem=(queryOutputMemory M c t values).mem := by
  unfold queryOutputMemory
  apply scatter_memory_congr
  exact same

theorem backward_memory_mem (M : Algebra α) (c : BackwardConfig) (s t : State α)
    (values : BackwardAccumulator α c.BTL c.BK c.BV) (same : s.mem=t.mem) :
    (backwardOutputMemory M c s values).mem=(backwardOutputMemory M c t values).mem := by
  unfold backwardOutputMemory
  apply scatter_memory_congr
  apply scatter_memory_congr
  exact same

/-- Independent value recurrences and ordered writes; this definition contains
no execution, statement, register lookup, or success predicate. Aliased regions
are handled in the original DQ-before-DKV order. -/
def valuesMemory (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) : State α :=
  let q:=queryOutputMemory M c s (queryValues M D c s)
  backwardOutputMemory M c.toBackwardConfig q (backwardValues M D c.toBackwardConfig q)

specification full_backward_values (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) :
    ∃ final,exec (withDot M D) (kernel c) s=some final ∧
      final.mem=(valuesMemory M D (config c s) s).mem := by
  obtain ⟨initial,hprefix,registers,memory,_,_⟩:=prefix_run (withDot M D) c s
  obtain ⟨middle,hquery,queryMemory,nextRegisters⟩:=query_fp32 M D (config c s) initial registers
  obtain ⟨final,hbackward,backwardMemory⟩:=backward_fp32 M D (config c s) middle nextRegisters
  have queryRun : run (withDot M D) (queryCode c) initial=some middle := by
    simpa only [queryCode,kernel_config] using hquery
  have backwardRun : run (withDot M D) (backwardCode c) middle=some final := by
    simpa only [backwardCode,kernel_config] using hbackward
  refine ⟨final,?_,?_⟩
  · have execution : exec (withDot M D) (kernel c) s=run (withDot M D) (kernel c).surfaceBody s := by
      cases kernel c
      rfl
    rw [execution,source_split]
    simp only [run_append,hprefix,Option.bind_some,queryRun,run,step,backwardRun,bind,pure,Option.bind]
  · let q:=queryOutputMemory M (config c s) s (queryValues M D (config c s) s)
    have middleMemory : middle.mem=q.mem := by
      rw [queryMemory,query_values_mem M D (config c s) initial s memory]
      exact query_memory_mem M _ initial s _ memory
    rw [backwardMemory,backward_values_mem M D (config c s).toBackwardConfig middle q middleMemory]
    exact backward_memory_mem M _ middle q _ middleMemory

/-- Compare the actual call shell with independently invoked current helpers.
The condition concerns only the source PID decoder, not numerical values. -/
theorem helpers_values (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) :
    ∃ final,(exec (withDot M D) (queryKernel c) s).bind
        (exec (withDot M D) (backwardKernel c.toBackwardConfig))=some final ∧
      final.mem=(valuesMemory M D c s).mem := by
  obtain ⟨q,hq,mq⟩:=VeriTile.Bench.Repairs.ParallelAttentionKernel.query_fp32 M D c s
  obtain ⟨final,hb,mb⟩:=VeriTile.Bench.Repairs.ParallelAttentionKernel.backward_fp32 M D c.toBackwardConfig q
  refine ⟨final,by rw [hq,Option.bind_some,hb],?_⟩
  let reference:=queryOutputMemory M c s (queryValues M D c s)
  rw [mb,backward_values_mem M D c.toBackwardConfig q reference mq]
  exact backward_memory_mem M _ q reference _ mq

theorem source_from_helpers (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (s target : State α) (matched : config c s=c)
    (executed : (exec (withDot M D) (queryKernel c) s).bind
      (exec (withDot M D) (backwardKernel c.toBackwardConfig))=some target) :
    ∃ final,exec (withDot M D) (kernel c) s=some final ∧ final.mem=target.mem := by
  obtain ⟨final,ran,memory⟩:=full_backward_values M D c s
  obtain ⟨reference,staged,referenceMemory⟩:=helpers_values M D c s
  rw [executed] at staged
  cases Option.some.inj staged
  exact ⟨final,ran,by rw [matched] at memory;exact memory.trans referenceMemory.symm⟩

#axiomsClean source_split
#axiomsClean full_backward_values
#axiomsClean source_from_helpers
#specNonCircular valuesMemory avoiding [kernel,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller

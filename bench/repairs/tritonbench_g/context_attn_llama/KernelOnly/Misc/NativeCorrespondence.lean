import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeSource
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeSigned

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
open Integer
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual typed operand evaluations equal finite native signed expressions.
This covers each query, page, key, causal, value and output address/mask site. -/
def FiniteObserved (r : Request) (idx : GridIndex r.call.grid) (step : Nat)
    (s : State NativeGraph.Expr) : Prop :=
  let c := r.call
  let M := (backend r).withDefaultPrecision .fp32
  evalOp M none (PreludeProgram.offset c.layout) s = some (fun i =>
    (qOffsetBits c Safety.widths .query idx i.1.val i.2.1.val).toInt) ∧
  evalOp M none (PreludeProgram.queryMask c.layout) s = some (fun i =>
    queryMaskBits c Safety.widths idx i.1.val) ∧
  evalOp M none (LoopProgram.pageIndex c.layout) s = some (fun i =>
    (pageOffsetBits c.layout (c.launchAt idx) Safety.widths.request (c.request (batchAt c idx)) step i.1.val).toInt) ∧
  evalOp M none (LoopProgram.pageMask c.layout) s = some (fun i => NativeSigned.keyMask c idx step i.1.val) ∧
  evalOp M none (LoopProgram.keyOffsets c.layout) s = some (fun i =>
    (NativeSigned.payloadOffset c .key idx step i.2.1.val i.1.val).toInt) ∧
  evalOp M none (LoopProgram.keyMask c.layout) s = some (fun i => NativeSigned.keyMask c idx step i.2.1.val) ∧
  evalOp M none (LoopProgram.causalMask c.layout) s = some (fun i => NativeSigned.causalMask c idx step i.1.val i.2.1.val) ∧
  evalOp M none (LoopProgram.valueOffsets c.layout) s = some (fun i =>
    (NativeSigned.payloadOffset c .value idx step i.1.val i.2.1.val).toInt) ∧
  evalOp M none (LoopProgram.valueMask c.layout) s = some (fun i => NativeSigned.keyMask c idx step i.1.val) ∧
  evalOp M none (EpilogueProgram.offsets c.layout) s = some (fun i =>
    (qOffsetBits c Safety.widths .output idx i.1.val i.2.1.val).toInt)

theorem bool_decide (b : Bool) (p : Prop) [Decidable p] (h : b=Bool.true ↔ p) : decide p=b := by
  cases b <;> simp_all

theorem finite_observations (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) (step : Nat) (ht : step<countAt r.call idx)
    (s : State NativeGraph.Expr) (observed : Observed r.call idx step ((backend r).withDefaultPrecision .fp32) s) :
    FiniteObserved r idx step s := by
  rcases observed with ⟨q,qm,pg,pm,k,km,cm,v,vm,o⟩
  have finite := Safety.machine r heap h
  unfold FiniteObserved
  refine ⟨q.trans ?_,qm.trans ?_,pg.trans ?_,pm.trans ?_,k.trans ?_,km.trans ?_,cm.trans ?_,v.trans ?_,vm.trans ?_,o.trans ?_⟩
  all_goals
    apply congrArg some
    funext i
  · exact (Safety.query_formed r heap h idx i.1 i.2.1 .query (Or.inl rfl)).1.symm
  · exact bool_decide _ _ (queryMask_exact r.call Safety.widths finite idx i.1)
  · exact (Safety.page_formed r heap h idx step i.1 ht).1.symm
  · exact bool_decide _ _ (NativeSigned.key_mask r heap h idx step i.1 ht)
  · exact (NativeSigned.payload_offset r heap h idx step i.2.1 i.1 ht .key (Or.inl rfl)).symm
  · exact bool_decide _ _ (NativeSigned.key_mask r heap h idx step i.2.1 ht)
  · exact bool_decide _ _ (NativeSigned.causal_mask r heap h idx step i.1 i.2.1 ht)
  · exact (NativeSigned.payload_offset r heap h idx step i.1 i.2.1 ht .value (Or.inr rfl)).symm
  · exact bool_decide _ _ (NativeSigned.key_mask r heap h idx step i.1 ht)
  · exact (Safety.query_formed r heap h idx i.1 i.2.1 .output (Or.inr rfl)).1.symm

/-- Constructed source-grid and loop prefixes reach every native operand site;
no reached-state, host success, trace or result equation is a premise. -/
theorem native_source (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (jobs : List (GridIndex r.call.grid)) :
    ∃ initialState,FPSuccess.runGrid r.call (backend r) jobs (initial r heap)=some initialState ∧
      ∀ idx : GridIndex r.call.grid,∀ step,step<countAt r.call idx →
      ∃ s u,Reached r.call (backend r) idx initialState step s ∧
        run ((backend r).withDefaultPrecision .fp32)
          ((FPSuccess.parts r.call.layout regions (Host.scale r.call.layout.D)).body.take 2)
          (indexState r.call step s)=some u ∧ FiniteObserved r idx step u := by
  obtain ⟨initialState,run,bridge,_,_,_⟩ := source_connection r heap h jobs
  refine ⟨initialState,run,?_⟩
  intro idx step ht
  obtain ⟨s,u,reached,page,_,observed⟩ := bridge.2 idx step ht
  exact ⟨s,u,reached,page,finite_observations r heap h idx step ht u observed⟩

#axiomsClean native_source
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSource

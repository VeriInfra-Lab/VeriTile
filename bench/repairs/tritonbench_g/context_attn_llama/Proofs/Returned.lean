import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FormedBounds

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Returned
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
open Bytes
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _

structure Context where
  (suppliedOutput ownerDevice callerDevice : Nat)
structure Result where
  graph : State NativeInstructions.Expr
  bytes : Memory
  returned : Option Unit
  (suppliedOutput currentDevice : Nat)
  launchDevice : Option Nat

def invoke (c : Call) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (ctx : Context) (before : Memory) : Option Result := do
  let graph ← FPSuccess.runGrid c (backend c) (order c) (initial c B before)
  return ⟨graph,stores c B ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩

def WholeBytes (c : Call) (A : FlatAlloc) (B : Windows A.extent)
    (ops : NativeInstructions.ScalarOps) (ctx : Context) (before : Memory) : Prop :=
  ∃ result,invoke c B ops ctx before=some result ∧
    result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧ result.currentDevice=ctx.callerDevice ∧
    result.launchDevice=(if Needed c then some ctx.ownerDevice else none) ∧
    FPSuccess.runGrid c (backend c) (order c) (initial c B before)=some result.graph ∧
    result.graph.mem=(reference c B before).mem ∧
    (∀ i : Lane c,active c i → result.bytes (address c B i)=byte (referenceWord c B ops before i) i.2.2) ∧
    (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → result.bytes a=before a) ∧
    (∀ rg∈c.inputs,∀ off,off < A.extent rg → ∀ j : Fin 4,
      result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
    (∀ rg off,¬FPSuccess.writes c (order c) rg off → result.graph.mem rg off=(initial c B before).mem rg off) ∧
    (¬Needed c → result.bytes=before)

theorem whole_bytes (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : Checked.checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (before : Memory) (observed : Observed c B before start sequence prompt request pages)
    (storage : Checked.StorageBinding c v B) (labels : Checked.Labels c)
    (ops : NativeInstructions.ScalarOps) (ctx : Context) : WholeBytes c A B ops ctx before := by
  have guard := (Checked.checkInputs_iff c d v A.extent requestedMax start sequence prompt request pages).mp checked
  have hv := Checked.valid c d A (layoutState B before) start sequence prompt request pages guard.2.1 guard.2.2.2
    (observation c B before start sequence prompt request pages observed) labels
  have metadata := Bytes.metadata c B before start sequence prompt request pages observed guard.2.2.2
  obtain ⟨graph,run,referenceEq⟩ := FPReference.whole c (backend c) (support c) (order c) (initial c B before)
  obtain ⟨framed,frameRun,_,framing⟩ := FPSuccess.grid_success_frame c (backend c) (support c) labels (order c)
    (initial c B before) metadata
  have eq : framed=graph := Option.some.inj (frameRun.symm.trans run)
  subst framed
  refine ⟨⟨graph,stores c B ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩,?_,rfl,rfl,rfl,rfl,run,referenceEq,?_,?_,?_,framing,?_⟩
  · simp only [invoke,run]; rfl
  · intro i hi
    change stores c B ops graph before (address c B i)=_
    rw [readback c B (layoutState B before) hv ops graph before i hi]
    unfold outputWord referenceWord
    rw [referenceEq]
    rfl
  · exact frame c B ops graph before
  · intro rg read off bound j
    by_cases needed : Needed c
    · exact input_frame c A (layoutState B before) B hv
        (Checked.physical_separate c d v B guard.1 guard.2.1 storage needed) ops graph before rg read off bound j
    · apply frame
      rintro ⟨i,hi,_⟩
      exact needed ⟨Formula.batch c.layout (c.launchAt i.1),Host.batch_bound c i.1,Nat.zero_lt_of_lt hi⟩
  · intro empty
    funext a
    apply frame
    rintro ⟨i,hi,_⟩
    exact empty ⟨Formula.batch c.layout (c.launchAt i.1),Host.batch_bound c i.1,Nat.zero_lt_of_lt hi⟩

structure MachineFacts (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (A : FlatAlloc)
    (B : Windows A.extent) (before : Memory) : Prop where
  native : Needed c → Native.GridBridge c A (layoutState B before) Checked.widths (Checked.nativeBytes B)
    (backend c) (initial c B before)
  coverage : ∀ b : Fin c.batches,∀ row,row < c.length b → row < c.blocks*c.layout.BM
  strides : v.q.strides=[c.layout.qRow,c.layout.qHead,c.layout.qDim] ∧
    v.k.strides=[c.layout.kRow,c.layout.kHead,c.layout.kDim] ∧
    v.v.strides=[c.layout.vRow,c.layout.vHead,c.layout.vDim] ∧
    v.o.strides=[c.layout.outRow,c.layout.outHead,c.layout.outDim] ∧
    v.pages.strides=[c.layout.reqRow,c.layout.reqToken]
  alignment : ∀ rg x,(rg,x)∈Checked.roleViews c v → 0 < x.metadata.numel → ∀ off,(B.base rg+4*off)%4=0
  control : Needed c → ∀ (idx : GridIndex c.grid) cur,cur≤Native.seqAt c idx+c.layout.BM →
    (Native.smin (Native.word ((c.launchAt idx).block*c.layout.BM)+Native.word c.layout.BM+
      Native.word (c.prompt (Native.batchAt c idx))) (Native.word (Native.seqAt c idx))).toInt=
      (min ((c.launchAt idx).block*c.layout.BM+c.layout.BM+c.prompt (Native.batchAt c idx)) (Native.seqAt c idx) : Nat) ∧
    (Native.word cur+Native.word c.layout.BM).toInt=(cur+c.layout.BM : Nat)
  pageFormed : Needed c → ∀ (idx : GridIndex c.grid) step (j : Fin c.layout.BN),step < Native.countAt c idx →
    let off := c.layout.reqRow*c.request (Native.batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val)
    (Native.pageOffsetBits c.layout (c.launchAt idx) Checked.widths.request (c.request (Native.batchAt c idx)) step j).toInt=(off : Int) ∧
    (Native.word (4*off)).toInt=(4*off : Nat)
  queryFormed : Needed c → ∀ (idx : GridIndex c.grid) (i : Fin c.layout.BM) (dim : Fin c.layout.D)
    role,role=Native.DataRole.query ∨ role=.output →
    (Native.qOffsetBits c Checked.widths role idx i dim).toInt=(Native.qOffset c role idx i dim : Int) ∧
    (Native.word (4*Native.qOffset c role idx i dim)).toInt=(4*Native.qOffset c role idx i dim : Nat)
  formed : Needed c → ∀ (idx : GridIndex c.grid) step (i : Fin c.layout.BM) (j : Fin c.layout.BN) (dim : Fin c.layout.D),
    step < Native.countAt c idx →
    (∀ role,role=Native.DataRole.query ∨ role=.output →
      (Native.qOffsetBits c Checked.widths role idx i dim).toInt=(Native.qOffset c role idx i dim : Int) ∧
      (Native.word (4*Native.qOffset c role idx i dim)).toInt=(4*Native.qOffset c role idx i dim : Nat)) ∧
    (∀ role,role=Native.DataRole.key ∨ role=.value →
      (Native.kvOffsetBits c Checked.widths role idx step j dim).toInt=(Native.kvOffset c role idx step j dim : Int) ∧
      (Native.word (4*Native.kvOffset c role idx step j dim)).toInt=(4*Native.kvOffset c role idx step j dim : Nat))

theorem machine_facts (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : Checked.checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (before : Memory) (observed : Observed c B before start sequence prompt request pages)
    (storage : Checked.StorageBinding c v B) (labels : Checked.Labels c) : MachineFacts c d v A B before := by
  have guard := (Checked.checkInputs_iff c d v A.extent requestedMax start sequence prompt request pages).mp checked
  refine ⟨Checked.source_native c d v A B requestedMax start sequence prompt request pages checked before observed storage labels,
    ?_,Checked.source_strides c d v A.extent guard.1 guard.2.1,
    Checked.role_alignment c d v B guard.1 storage,
    ?_,
    Checked.page_formed c d v A B requestedMax start sequence prompt request pages guard before observed storage,
    Checked.query_formed c d v A B requestedMax start sequence prompt request pages guard before observed storage,
    Checked.all_formed c d v A B requestedMax start sequence prompt request pages guard before observed storage⟩
  · intro b row hr
    exact Host.full_coverage c b (guard.2.1.2.2.2.2.2.2.1 b).2.1 row hr
  · intro needed idx cur hcur
    have abi := Checked.abi c d v A B requestedMax start sequence prompt request pages guard before observed storage needed
    exact Checked.direct_control _ _ _ _ cur (Native.block_abi c A Checked.widths (Checked.nativeBytes B) abi idx)
      c.blockM (abi.sequence _ (Host.batch_bound c idx)) (by dsimp only [Native.seqAt]; omega) hcur

theorem whole_if_local (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : Checked.checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (before : Memory) (observed : Observed c B before start sequence prompt request pages)
    (storage : Checked.StorageBinding c v B) (labels : Checked.Labels c)
    (actual referenceOps : NativeInstructions.ScalarOps) (laws : NativeInstructions.ScalarLaws actual referenceOps)
    (ctx : Context) : WholeBytes c A B actual ctx before ∧ WholeBytes c A B referenceOps ctx before ∧
      invoke c B actual ctx before=invoke c B referenceOps ctx before := by
  have eq := laws.operations_eq actual referenceOps
  subst referenceOps
  have result := whole_bytes c d v A B requestedMax start sequence prompt request pages checked before observed storage labels actual ctx
  exact ⟨result,result,rfl⟩

#axiomsClean whole_bytes
#axiomsClean machine_facts
#axiomsClean whole_if_local
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Returned

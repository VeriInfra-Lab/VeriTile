import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.StorageBridge
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.CheckedNative

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Returned
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Bytes
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _

structure Context where
  (suppliedOutput ownerDevice callerDevice : Nat)

structure Result where
  graph : State NativePhysical.Expr
  bytes : Memory
  returned : Option Unit
  (suppliedOutput currentDevice : Nat)
  launchDevice : Option Nat

def invoke (c : Call) (r : Regions) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (ctx : Context) (before : Memory) : Option Result := do
  let graph ← FPReference.runGrid c r (backend c) (order c) (initial c r B before)
  return ⟨graph,stores c r B ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩

def WholeBytes (c : Call) (r : Regions) (A : FlatAlloc) (B : Windows A.extent)
    (ops : NativeInstructions.ScalarOps) (ctx : Context) (before : Memory) : Prop :=
  ∃ result,invoke c r B ops ctx before=some result ∧
    result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧ result.currentDevice=ctx.callerDevice ∧
    result.launchDevice=(if Needed c then some ctx.ownerDevice else none) ∧
    FPReference.runGrid c r (backend c) (order c) (initial c r B before)=some result.graph ∧
    result.graph.mem=(reference c r B before).mem ∧
    (∀ i : Lane c,active c i → result.bytes (address c r B i)=byte (referenceWord c r B ops before i) i.2.2) ∧
    (∀ a,(¬∃ i : Lane c,active c i ∧ address c r B i=a) → result.bytes a=before a) ∧
    (∀ rg∈readRegions r,∀ off,off < A.extent rg → ∀ j : Fin 4,
      result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
    (∀ rg off,¬FPReference.Writes c r (order c) rg off → result.graph.mem rg off=(initial c r B before).mem rg off) ∧
    (¬Needed c → result.bytes=before)

/-- Same original bytes supply every input role and the independent reference.
All stores/readback/frames and the normal host observations are constructed. -/
theorem whole_bytes (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start length : Nat → Int)
    (checked : Checked.checkInputs c r v A.extent requestedMax start length=Bool.true)
    (before : Memory) (observed : Observed c r B before start length)
    (storage : Checked.StorageBinding r v B) (labels : Checked.Labels r)
    (ops : NativeInstructions.ScalarOps) (ctx : Context) : WholeBytes c r A B ops ctx before := by
  have guard := (Checked.checkInputs_iff c r v A.extent requestedMax start length).mp checked
  have hm := Checked.memory c r A (layoutState B before) start length guard.2.1 guard.2.2.2
    (observation c r B before start length observed) labels
  have metadata := Bytes.metadata c r B before start length observed guard.2.2.2
  obtain ⟨graph,run,referenceEq⟩ := FPReference.whole c r (backend c) (order c) (initial c r B before)
  have framing := (FPReference.grid_frame c r (backend c) (order c) (initial c r B before) metadata
    labels.2.2.2.1 labels.2.2.2.2).2
  refine ⟨⟨graph,stores c r B ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩,?_,rfl,rfl,rfl,rfl,run,referenceEq,?_,?_,?_,?_,?_⟩
  · simp only [invoke,run]; rfl
  · intro i hi
    change stores c r B ops graph before (address c r B i)=_
    rw [readback c r B (layoutState B before) hm ops graph before i hi]
    unfold outputWord referenceWord
    rw [referenceEq]
    rfl
  · exact frame c r B ops graph before
  · intro rg read off bound j
    by_cases needed : Needed c
    · exact input_frame c r A (layoutState B before) B hm
        (Checked.physical_separate c r v B guard.1 guard.2.1 storage needed) ops graph before rg read off bound j
    · apply frame
      rintro ⟨i,hi,_⟩
      exact needed ⟨(c.launch i.1).batch,launch_batch_lt c i.1,Nat.zero_lt_of_lt hi⟩
  · intro rg off missing
    rw [referenceEq]
    exact framing rg off missing
  · intro empty
    funext a
    apply frame
    rintro ⟨i,hi,_⟩
    exact empty ⟨(c.launch i.1).batch,launch_batch_lt c i.1,Nat.zero_lt_of_lt hi⟩

/-- Guard-derived actual native source control/active addresses. -/
theorem native_projection (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start length : Nat → Int)
    (checked : Checked.checkInputs c r v A.extent requestedMax start length=Bool.true)
    (before : Memory) (observed : Observed c r B before start length)
    (sizes : Checked.AllocationSizes r v A) (labels : Checked.Labels r) :
    ∀ j : HostPlan.Job c,NativeProgram.Projects c r A (layoutState B before) j := by
  have guard := (Checked.checkInputs_iff c r v A.extent requestedMax start length).mp checked
  have hm := Checked.memory c r A (layoutState B before) start length guard.2.1 guard.2.2.2
    (observation c r B before start length observed) labels
  have abi := Checked.abi c r v A requestedMax guard.1 guard.2.2.1 sizes
  have blocks := guard.2.2.1.2.2.2.2.1
  exact NativeProgram.projects hm abi (by omega)

/-- Native control and both active and masked formed addresses, with the
ordinary queried-stride relation and actual byte alignment. -/
structure MachineFacts (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc)
    (B : Windows A.extent) (before : Memory) : Prop where
  projects : ∀ j : HostPlan.Job c,NativeProgram.Projects c r A (layoutState B before) j
  strides : v.q.strides=[c.q.rowStride,c.q.headStride,c.q.dimStride] ∧
    v.k.strides=[c.k.rowStride,c.k.headStride,c.k.dimStride] ∧
    v.v.strides=[c.v.rowStride,c.v.headStride,c.v.dimStride] ∧
    v.o.strides=[c.o.rowStride,c.o.headStride,c.o.dimStride]
  aligned : ∀ rg x,(rg,x)∈Checked.roleViews r v → 0 < x.metadata.numel → ∀ off,
    (B.base rg+4*off)%4=0
  formed : ∀ b block iter lane head dim,b < c.batches → block < c.blocks → iter≤block →
    lane < 128 → head < c.q.heads → dim < c.q.dims →
    (NativeProgram.queryWord c.q (c.start b) block lane head dim).toInt=
      (c.q.offset (c.start b+(block*128+lane)) head dim : Int) ∧
    (NativeProgram.keyWord c.q (c.start b) (iter*128) lane head dim).toInt=
      (c.q.offset (c.start b+iter*128+lane) head dim : Int) ∧
    (NativeProgram.word (4*c.q.offset (c.start b+(block*128+lane)) head dim)).toInt=
      (4*c.q.offset (c.start b+(block*128+lane)) head dim : Nat) ∧
    (NativeProgram.word (4*c.q.offset (c.start b+iter*128+lane) head dim)).toInt=
      (4*c.q.offset (c.start b+iter*128+lane) head dim : Nat)

theorem machine_facts (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start length : Nat → Int)
    (checked : Checked.checkInputs c r v A.extent requestedMax start length=Bool.true)
    (before : Memory) (observed : Observed c r B before start length)
    (storage : Checked.StorageBinding r v B) (sizes : Checked.AllocationSizes r v A)
    (labels : Checked.Labels r) : MachineFacts c r v A B before := by
  have guard := (Checked.checkInputs_iff c r v A.extent requestedMax start length).mp checked
  refine ⟨native_projection c r v A B requestedMax start length checked before observed sizes labels,
    Checked.source_strides c r v A.extent guard.1 guard.2.1,
    Checked.role_alignment c r v B guard.1 storage,?_⟩
  exact Checked.all_formed_exact c r A.extent requestedMax guard.2.1 guard.2.2.1.2.1 v.q
    (guard.2.2.1.2.2.2.2.2.1 v.q (by simp))

theorem whole_if_local (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start length : Nat → Int)
    (checked : Checked.checkInputs c r v A.extent requestedMax start length=Bool.true)
    (before : Memory) (observed : Observed c r B before start length)
    (storage : Checked.StorageBinding r v B) (labels : Checked.Labels r)
    (actual referenceOps : NativeInstructions.ScalarOps) (laws : NativeInstructions.ScalarLaws actual referenceOps)
    (ctx : Context) : WholeBytes c r A B actual ctx before ∧ WholeBytes c r A B referenceOps ctx before ∧
      invoke c r B actual ctx before=invoke c r B referenceOps ctx before := by
  have eq := laws.operations_eq actual referenceOps
  subst referenceOps
  have result := whole_bytes c r v A B requestedMax start length checked before observed storage labels actual ctx
  exact ⟨result,result,rfl⟩

#axiomsClean whole_bytes
#axiomsClean native_projection
#axiomsClean machine_facts
#axiomsClean whole_if_local
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Returned

import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.RawKernel
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelTensor
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Gradient
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Scalar

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Kernel
open VeriTile.Triton FP.Structural Checked KernelTensor
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

/-- Independent entry observations at the actual argument pointers. -/
def data (t : Arguments c U S V runtime) (memory : Role → Nat → α) : Input α :=
  fun r bh time d => atPointer t memory r
    (if r="h0" ∨ r="dht" then stateAddr bh c.D d else rowAddr bh c.T c.D time d)

theorem data_inputs (t : Arguments c U S V runtime) (memory : Role → Nat → α)
    (other : State α) : Execution.Inputs (lift t memory other) (data t memory) c.T c.D := by
  constructor
  · intro r hr bh time d
    rcases hr with rfl|rfl|rfl
    · simpa [data,atPointer,role,name] using lift_read t memory other .x (rowAddr bh c.T c.D time d)
    · simpa [data,atPointer,role,name] using lift_read t memory other .g (rowAddr bh c.T c.D time d)
    · simpa [data,atPointer,role,name] using lift_read t memory other .incoming (rowAddr bh c.T c.D time d)
  · intro r hr bh d
    rcases hr with rfl|rfl
    · simpa [data,atPointer,role,name] using lift_read t memory other .h0 (stateAddr bh c.D d)
    · simpa [data,atPointer,role,name] using lift_read t memory other .dht (stateAddr bh c.D d)

/-- Full source execution, logical Tensor results, phase frames, and checked
coverage/separation. No host return, allocator, or final byteheap assumption. -/
structure Claim (p : Ops α) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (t : Arguments c U S V runtime) (BDF BDB : Nat)
    (memory : Role → Nat → α) (other f b : State α) : Prop where
  forwardExecuted : Execution.gridRun p (forward c.T c.D BDF U S).surfaceBody
    (Execution.programs c.BH c.D BDF) (lift t memory other)=some f
  backwardExecuted : Execution.gridRun p (backward c.T c.D BDB U (S && V)).surfaceBody
    (Execution.programs c.BH c.D BDB) f=some b
  output : ∀ bh time d, bh<c.BH → time<c.T → d<c.D →
    result t b .o (rowAddr bh c.T c.D time d)=state p U (data t memory) bh d (time+1)
  finalState : S=Bool.true → ∀ bh d, bh<c.BH → d<c.D →
    result t b .ht (stateAddr bh c.D d)=state p U (data t memory) bh d c.T
  gradients : ∀ bh time d, bh<c.BH → time<c.T → d<c.D →
    result t b .dx (rowAddr bh c.T c.D time d)=dx p (S && V) (data t memory) c.T bh d time ∧
    result t b .dg (rowAddr bh c.T c.D time d)=dg p U (S && V) (data t memory) c.T bh d time
  forwardFrame : ∀ r a, ¬Execution.ForwardRegion c.BH c.T c.D S r a →
    f.mem r a=(lift t memory other).mem r a
  backwardFrame : ∀ r a, ¬Execution.BackwardRegion c.BH c.T c.D r a → b.mem r a=f.mem r a
  forwardSafety : Safety c U S V runtime BDF
  backwardSafety : Safety c U S V runtime BDB
  separation : ∀ x y, Physical.Active c U S V x → Physical.Active c U S V y →
    written x.role → written y.role → x≠y → (Physical.key runtime x).disjoint (Physical.key runtime y)

theorem complete (p : Ops α) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (BDF BDB : Nat) (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true)
    (memory : Role → Nat → α) (other : State α) :
    ∃ f b, Claim p c U S V runtime t BDF BDB memory other f b := by
  obtain ⟨f,b,raw⟩ := RawKernel.complete p (data t memory) c.BH c.T c.D BDF BDB
    (tile_bounds BDF tileF).1 (tile_bounds BDB tileB).1 U S (S && V)
    (lift t memory other) (data_inputs t memory other)
  have hf := raw.forwardExecuted
  have hb := raw.backwardExecuted
  have hO := raw.output
  have hH := raw.finalState
  have hG := raw.gradients
  have hff := raw.forwardFrame
  have hbf := raw.backwardFrame
  refine ⟨f,b,hf,hb,?_,?_,?_,hff,hbf,
    checked_safety c U S V runtime accepted BDF tileF,
    checked_safety c U S V runtime accepted BDB tileB,
    Physical.key_disjoint c U S V runtime accepted⟩
  · intro bh time d hh ht hd
    rw [result_eq c U S V runtime accepted t b .o rfl _ (row_bound c.BH c.T c.D bh time d hh ht hd)]
    simp only [name,Execution.read]
    rw [hO bh hh time ht d hd]
    rfl
  · intro hs bh d hh hd
    rw [result_eq c U S V runtime accepted t b .ht hs _ (state_bound c.BH c.D bh d hh hd)]
    simp only [name,Execution.read]
    rw [hH hs bh hh d hd]
    rfl
  · intro bh time d hh ht hd
    have bound := row_bound c.BH c.T c.D bh time d hh ht hd
    obtain ⟨hx,hg⟩ := hG bh hh time ht d hd
    rw [result_eq c U S V runtime accepted t b .dx rfl _ bound,
      result_eq c U S V runtime accepted t b .dg rfl _ bound]
    simp only [name,Execution.read]
    rw [hx,hg]
    exact ⟨rfl,rfl⟩

def Derivative (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (t : Arguments c U S V runtime) (memory : Role → Nat → ℝ) (b : State ℝ) : Prop :=
  ∀ bh d, bh<c.BH → d<c.D → ∀ vx vg : Nat → ℝ,
    HasDerivAt (fun e =>
      (∑ time∈Finset.range c.T,data t memory "do" bh time d*
        pathState (data t memory) U bh d vx vg e (time+1))+
      terminal realOps (S && V) (data t memory) bh d*
        pathState (data t memory) U bh d vx vg e c.T)
      (∑ time∈Finset.range c.T,
        (result t b .dx (rowAddr bh c.T c.D time d)*vx time+
         result t b .dg (rowAddr bh c.T c.D time d)*vg time)) 0

theorem real_complete (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (BDF BDB : Nat) (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true)
    (memory : Role → Nat → ℝ) (other : State ℝ) :
    ∃ f b, Claim realOps c U S V runtime t BDF BDB memory other f b ∧
      Derivative c U S V runtime t memory b := by
  obtain ⟨f,b,h⟩ := complete realOps c U S V runtime accepted t BDF BDB tileF tileB memory other
  refine ⟨f,b,h,?_⟩
  intro bh d hh hd vx vg
  have equal : (∑ time∈Finset.range c.T,
      (result t b .dx (rowAddr bh c.T c.D time d)*vx time+
       result t b .dg (rowAddr bh c.T c.D time d)*vg time))=
    ∑ time∈Finset.range c.T,(dx realOps (S && V) (data t memory) c.T bh d time*vx time+
      dg realOps U (S && V) (data t memory) c.T bh d time*vg time) := by
    apply Finset.sum_congr rfl
    intro time ht
    rw [(h.gradients bh time d hh (Finset.mem_range.mp ht) hd).1,
      (h.gradients bh time d hh (Finset.mem_range.mp ht) hd).2]
  rw [equal]
  exact loss_derivative (data t memory) U (S && V) c.T bh d vx vg

def terms (words : Role → Nat → BitVec 32) : Role → Nat → Native.Term :=
  fun r address => .leaf (name r) address (words r address)

def gateNode : Native.Shape → Bool
  | .leaf r _ => r=="g"
  | _ => Bool.false

/-- Local schedules observed in the pinned short-sequence and looped SM80
profiles. The compiler boundary selects the specialization's schedule; this
definition reads topology only and has no input-word or result callback. -/
def profileSchedule (reverseUnrolled : Bool) : Native.Schedule where
  swapAdd := fun lhs _ => match lhs with | .zero => Bool.true | _ => Bool.false
  swapMul := fun _ _ => Bool.false
  contract := fun lhs _ _ => gateNode lhs || reverseUnrolled
  swapFma := fun lhs rhs _ => match rhs with | .zero => Bool.false | _ => gateNode lhs

/-- Numerical observation of the independently defined source graph. A saved
forward state occurs only as a multiplication operand in dg, so its evaluation
is completed before that instruction: no contraction crosses the saved load. -/
def NativeValues (p : Native.Instructions) (schedule : Native.Schedule)
    (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (t : Arguments c U S V runtime) (words : Role → Nat → BitVec 32) (b : State Native.Term) : Prop :=
  (∀ bh time d, bh<c.BH → time<c.T → d<c.D →
    Native.eval p schedule (result t b .o (rowAddr bh c.T c.D time d))=
      Native.eval Native.software schedule (state Native.sourceOps U (data t (terms words)) bh d (time+1)) ∧
    Native.eval p schedule (result t b .dx (rowAddr bh c.T c.D time d))=
      Native.eval Native.software schedule (dx Native.sourceOps (S && V) (data t (terms words)) c.T bh d time) ∧
    Native.eval p schedule (result t b .dg (rowAddr bh c.T c.D time d))=
      Native.eval Native.software schedule (dg Native.sourceOps U (S && V) (data t (terms words)) c.T bh d time)) ∧
  (S=Bool.true → ∀ bh d, bh<c.BH → d<c.D →
    Native.eval p schedule (result t b .ht (stateAddr bh c.D d))=
      Native.eval Native.software schedule (state Native.sourceOps U (data t (terms words)) bh d c.T))

def NativeTensorResult (p : Native.Instructions) (schedule : Native.Schedule)
    (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (t : Arguments c U S V runtime) (b : State Native.Term) : Prop :=
  ∀ r, present U S V r=Bool.true → ∀ i, i<extent c r →
    VeriTile.Torch.FP32.logicalValue (resultWords (Native.eval p schedule) t b r)
      (t.tensor r) (t.laws r) i=
      (⟨Native.eval p schedule (result t b r i)⟩ : VeriTile.Torch.FP32.Word)

theorem native_complete (p : Native.Instructions) (laws : Native.LocalConformance p)
    (schedule : Native.Schedule) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (BDF BDB : Nat) (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true)
    (words : Role → Nat → BitVec 32) (other : State Native.Term) :
    ∃ f b, Claim Native.sourceOps c U S V runtime t BDF BDB (terms words) other f b ∧
      NativeValues p schedule c U S V runtime t words b ∧
      NativeTensorResult p schedule c U S V runtime t b := by
  obtain ⟨f,b,h⟩ := complete Native.sourceOps c U S V runtime accepted t BDF BDB tileF tileB (terms words) other
  refine ⟨f,b,h,⟨?_,?_⟩,?_⟩
  · intro bh time d hh ht hd
    rw [h.output bh time d hh ht hd,(h.gradients bh time d hh ht hd).1,
      (h.gradients bh time d hh ht hd).2,Native.software_eq p laws]
    exact ⟨rfl,rfl,rfl⟩
  · intro hs bh d hh hd
    rw [h.finalState hs bh d hh hd,Native.software_eq p laws]
  · intro r active i inside
    exact result_logical c U S V runtime accepted (Native.eval p schedule) t b r active i inside

theorem saved_rounding_boundary (p : Native.Instructions) (schedule : Native.Schedule)
    (gradient saved : Native.Term) :
    Native.eval p schedule (.mul gradient saved)=
      if schedule.swapMul (Native.shape gradient) (Native.shape saved) then
        p.mulRN (Native.eval p schedule saved) (Native.eval p schedule gradient)
      else p.mulRN (Native.eval p schedule gradient) (Native.eval p schedule saved) := rfl

theorem saved_word_reload (p : Native.Instructions) (reverseUnrolled : Bool)
    (gradient saved : Native.Term) (address : Nat) :
    Native.eval p (profileSchedule reverseUnrolled) (.mul gradient saved)=
      Native.eval p (profileSchedule reverseUnrolled)
        (.mul gradient (.leaf "o" address (Native.eval p (profileSchedule reverseUnrolled) saved))) := by
  simp [Native.eval,profileSchedule]

#specNonCircular data avoiding [Execution.gridRun,FP.Structural.exec]
#specNonCircular profileSchedule avoiding [Execution.gridRun,FP.Structural.exec,data,terms,Native.eval]
#axiomsClean complete
#axiomsClean real_complete
#axiomsClean native_complete
#axiomsClean saved_rounding_boundary
#axiomsClean saved_word_reload
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Kernel

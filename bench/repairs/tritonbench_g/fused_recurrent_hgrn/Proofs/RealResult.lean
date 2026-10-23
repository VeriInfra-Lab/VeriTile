import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.StorageBinding
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Safety

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Public
open VeriTile.Triton FP.Structural
open Checked Physical Wrapper
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 2000
open scoped BigOperators

def realData {M β : Type} (api : API M ℝ β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (initial incoming : M) : Input ℝ :=
  Execution.hostInput realOps (Closed.input (memory api 0 runtime initial) c.T c.D) c.T c.D hasDo
    (fun a => api.read incoming (key runtime ⟨.incoming,a⟩))
    (fun a => api.read incoming (key runtime ⟨.dht,a⟩))

structure RealClaim {M β : Type} (api : API M ℝ β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (BDF BDB : Nat) (initial incoming : M)
    (out : Wrapper.Forward M) (grad : Wrapper.Backward M) : Prop where
  returned : out.output=runtime.tensor .o ∧ out.finalState=(if S then some (runtime.tensor .ht) else none) ∧
    out.saved=(runtime.tensor .g,runtime.tensor .o,if U then some (runtime.tensor .h0) else none) ∧
    grad.dx=runtime.tensor .dx ∧ grad.dg=runtime.tensor .dg ∧ grad.initialGradient=none ∧ grad.flagGradient=none
  forwardSafety : Safety c U S V runtime BDF
  backwardSafety : Safety c U S V runtime BDB
  output : ∀ bh t d,bh<c.BH → t<c.T → d<c.D →
    api.read out.memory (key runtime ⟨.o,rowAddr bh c.T c.D t d⟩)=
      state realOps U (Closed.input (memory api 0 runtime initial) c.T c.D) bh d (t+1)
  finalState : S=Bool.true → ∀ bh d,bh<c.BH → d<c.D →
    api.read out.memory (key runtime ⟨.ht,stateAddr bh c.D d⟩)=
      state realOps U (Closed.input (memory api 0 runtime initial) c.T c.D) bh d c.T
  gradients : ∀ bh t d,bh<c.BH → t<c.T → d<c.D →
    api.read grad.memory (key runtime ⟨.dx,rowAddr bh c.T c.D t d⟩)=
      dx realOps (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t ∧
    api.read grad.memory (key runtime ⟨.dg,rowAddr bh c.T c.D t d⟩)=
      dg realOps U (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t
  derivative : ∀ bh d,bh<c.BH → d<c.D → ∀ vx vg : Nat → ℝ,
    HasDerivAt (fun e =>
      (∑ t∈Finset.range c.T,realData api c U S V hasDo runtime initial incoming "do" bh t d*
        pathState (realData api c U S V hasDo runtime initial incoming) U bh d vx vg e (t+1))+
      terminal realOps (S && V) (realData api c U S V hasDo runtime initial incoming) bh d*
        pathState (realData api c U S V hasDo runtime initial incoming) U bh d vx vg e c.T)
      (∑ t∈Finset.range c.T,
        (api.read grad.memory (key runtime ⟨.dx,rowAddr bh c.T c.D t d⟩)*vx t+
        api.read grad.memory (key runtime ⟨.dg,rowAddr bh c.T c.D t d⟩)*vg t)) 0
  savedBindings : ∀ r,r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht → present U S V r=Bool.true → ∀ a,a<extent c r →
    api.read incoming (key runtime ⟨r,a⟩)=
      Execution.read (Closed.first realOps c BDF U S (memory api 0 runtime initial)) (name r) a
  cotangentBinding : ∀ a,a<c.rows → api.read incoming (key runtime ⟨.incoming,a⟩)=
    if hasDo then api.read incoming (key runtime ⟨.incoming,a⟩) else 0
  savedOutputs : ∀ x,x∈cells c S Bool.true →
    api.read grad.memory (key runtime x)=api.read out.memory (key runtime x)
  forwardFrame : ∀ owner address,Outside c U S V runtime Bool.true owner address →
    api.observe out.memory owner address=api.observe initial owner address
  backwardFrame : ∀ owner address,Outside c U S V runtime Bool.false owner address →
    api.observe grad.memory owner address=api.observe incoming owner address
  callerFrames : ∀ phase r,Readonly phase r → ∀ address,
    (runtime.tensor r).allocationBase ≤ address →
    address < (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes →
    api.observe (if phase then out.memory else grad.memory) (runtime.tensor r).device address=
      api.observe (if phase then initial else incoming) (runtime.tensor r).device address

theorem real_complete {M β : Type} (api : API M ℝ β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true) (BDF BDB : Nat)
    (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true) (initial incoming : M)
    (saved : SavedStorage api runtime (mathematicalForward api c U S V runtime BDF initial).memory incoming)
    (zero : MissingZero api 0 c U S V hasDo runtime incoming) :
    RealClaim api c U S V hasDo runtime BDF BDB initial incoming
      (mathematicalForward api c U S V runtime BDF initial)
      (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming) := by
  have first := Closed.first_complete realOps c BDF (tile_bounds BDF tileF).1 U S (memory api 0 runtime initial)
  have second := Closed.complete realOps c BDF BDB (tile_bounds BDF tileF).1 (tile_bounds BDB tileB).1
    U S hasDo V (memory api 0 runtime initial)
    (fun a => api.read incoming (key runtime ⟨.incoming,a⟩))
    (fun a => api.read incoming (key runtime ⟨.dht,a⟩))
  have grads (bh t d : Nat) (hb : bh<c.BH) (ht : t<c.T) (hd : d<c.D) :
      api.read (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming).memory
        (key runtime ⟨.dx,rowAddr bh c.T c.D t d⟩)=
          dx realOps (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t ∧
      api.read (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming).memory
        (key runtime ⟨.dg,rowAddr bh c.T c.D t d⟩)=
          dg realOps U (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t := by
    have values := second.2.2.1 bh hb t ht d hd
    constructor
    · dsimp only [mathematicalBackward,returnedBackward]
      rw [readback api c U S V runtime accepted Bool.false _ incoming _
        (backward_row_member c S .dx (Or.inl rfl) bh t d hb ht hd)]
      change Execution.read _ "dx" _=_
      rw [Execution.read,values.1]
      rfl
    · dsimp only [mathematicalBackward,returnedBackward]
      rw [readback api c U S V runtime accepted Bool.false _ incoming _
        (backward_row_member c S .dg (Or.inr rfl) bh t d hb ht hd)]
      change Execution.read _ "dg" _=_
      rw [Execution.read,values.2]
      rfl
  refine ⟨⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩,checked_safety c U S V runtime accepted BDF tileF,
    checked_safety c U S V runtime accepted BDB tileB,?_,?_,grads,?_,?_,?_,?_,?_,?_,?_⟩
  · intro bh t d hb ht hd
    dsimp only [mathematicalForward,returnedForward]
    rw [readback api c U S V runtime accepted Bool.true _ initial _ (forward_row_member c S bh t d hb ht hd)]
    change Execution.read _ "o" _=_
    rw [Execution.read,first.2.1 bh hb t ht d hd]
    rfl
  · intro hs bh d hb hd
    dsimp only [mathematicalForward,returnedForward]
    rw [readback api c U S V runtime accepted Bool.true _ initial _ (forward_slot_member c S hs bh d hb hd)]
    change Execution.read _ "ht" _=_
    rw [Execution.read,first.2.2.1 hs bh hb d hd]
    rfl
  · intro bh d hb hd vx vg
    have sum : (∑ t∈Finset.range c.T,
        (api.read (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming).memory
          (key runtime ⟨.dx,rowAddr bh c.T c.D t d⟩)*vx t+
        api.read (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming).memory
          (key runtime ⟨.dg,rowAddr bh c.T c.D t d⟩)*vg t))=
      ∑ t∈Finset.range c.T,(dx realOps (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t*vx t+
        dg realOps U (S && V) (realData api c U S V hasDo runtime initial incoming) c.T bh d t*vg t) := by
      apply Finset.sum_congr rfl
      intro t ht
      rw [(grads bh t d hb (Finset.mem_range.mp ht) hd).1,(grads bh t d hb (Finset.mem_range.mp ht) hd).2]
    rw [sum]
    exact loss_derivative _ U (S && V) c.T bh d vx vg
  · exact mathematical_saved api c U S V runtime accepted BDF (tile_bounds BDF tileF).1 initial incoming saved
  · intro a ha
    cases hdo : hasDo
    · exact zero hdo a ha
    · rfl
  · intro x hx
    have active := (cells_active c U S V Bool.true x hx).2
    have writes := (cells_mem c S Bool.true x).mp hx
    have role : x.role=.o ∨ x.role=.ht := by
      rcases writes with h|h
      · exact Or.inl h.1
      · exact Or.inr h.2.1
    dsimp only [mathematicalBackward,returnedBackward]
    rw [backward_keeps_outputs api c U S V runtime accepted _ incoming x active role]
    exact saved.read x.role (by rcases role with h|h <;> simp [h]) active.1 x.offset active.2
  · dsimp only [mathematicalForward,returnedForward]
    exact frame api c U S V runtime Bool.true _ initial
  · dsimp only [mathematicalBackward,returnedBackward]
    exact frame api c U S V runtime Bool.false _ incoming

  · intro phase r hr address lo hi
    cases phase
    · simp only [Bool.false_eq_true,↓reduceIte]
      dsimp only [mathematicalBackward,returnedBackward]
      exact caller_frame api c U S V runtime accepted Bool.false _ incoming r hr address lo hi
    · simp only [↓reduceIte]
      dsimp only [mathematicalForward,returnedForward]
      exact caller_frame api c U S V runtime accepted Bool.true _ initial r hr address lo hi

#axiomsClean real_complete
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Public

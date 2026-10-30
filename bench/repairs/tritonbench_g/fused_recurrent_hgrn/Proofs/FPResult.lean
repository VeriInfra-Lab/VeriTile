import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.StorageBinding
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Safety
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Scalar
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Physical

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Public
open VeriTile.Triton FP.Structural
open Checked Physical Wrapper
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 2000
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev encode := VeriTile.Bench.Repairs.RotaryEmb.Physical.encode

structure Codec {M : Type} (api : API M Word Byte) : Prop where
  read : ∀ m (key : VeriTile.Bench.Repairs.SwigluTriton.Memory.Key),key.address%4=0 → ∀ i : Fin 4,
    api.observe m key.owner (key.address+i.val)=encode (api.read m key) i

def wordData {M β : Type} (api : API M Word β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (initial incoming : M) : Input Native.Term :=
  Execution.hostInput Native.sourceOps (Closed.input (terms api runtime initial) c.T c.D) c.T c.D hasDo
    (fun a => .leaf "do" a (api.read incoming (key runtime ⟨.incoming,a⟩)))
    (fun a => .leaf "dht" a (api.read incoming (key runtime ⟨.dht,a⟩)))

structure FPClaim {M : Type} (api : API M Word Byte) (p : Native.Instructions) (schedule : Native.Schedule)
    (c : Shape) (U S V hasDo : Bool) (runtime : Runtime c U S V) (BDF BDB : Nat) (initial incoming : M)
    (out : Wrapper.Forward M) (grad : Wrapper.Backward M) : Prop where
  returned : out.output=runtime.tensor .o ∧ out.finalState=(if S then some (runtime.tensor .ht) else none) ∧
    out.saved=(runtime.tensor .g,runtime.tensor .o,if U then some (runtime.tensor .h0) else none) ∧
    grad.dx=runtime.tensor .dx ∧ grad.dg=runtime.tensor .dg ∧ grad.initialGradient=none ∧ grad.flagGradient=none
  forwardSafety : Safety c U S V runtime BDF
  backwardSafety : Safety c U S V runtime BDB
  output : ∀ bh t d,bh<c.BH → t<c.T → d<c.D →
    api.read out.memory (key runtime ⟨.o,rowAddr bh c.T c.D t d⟩)=Native.eval Native.software schedule
      (state Native.sourceOps U (Closed.input (terms api runtime initial) c.T c.D) bh d (t+1))
  finalState : S=Bool.true → ∀ bh d,bh<c.BH → d<c.D →
    api.read out.memory (key runtime ⟨.ht,stateAddr bh c.D d⟩)=Native.eval Native.software schedule
      (state Native.sourceOps U (Closed.input (terms api runtime initial) c.T c.D) bh d c.T)
  gradients : ∀ bh t d,bh<c.BH → t<c.T → d<c.D →
    api.read grad.memory (key runtime ⟨.dx,rowAddr bh c.T c.D t d⟩)=Native.eval Native.software schedule
      (dx Native.sourceOps (S && V) (wordData api c U S V hasDo runtime initial incoming) c.T bh d t) ∧
    api.read grad.memory (key runtime ⟨.dg,rowAddr bh c.T c.D t d⟩)=Native.eval Native.software schedule
      (dg Native.sourceOps U (S && V) (wordData api c U S V hasDo runtime initial incoming) c.T bh d t)
  savedBindings : ∀ r,r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht → present U S V r=Bool.true → ∀ a,a<extent c r →
    api.read incoming (key runtime ⟨r,a⟩)=Native.observe p schedule
      (Closed.first Native.sourceOps c BDF U S (terms api runtime initial)) (name r) a
  cotangentBinding : ∀ a,a<c.rows → api.read incoming (key runtime ⟨.incoming,a⟩)=
    if hasDo then api.read incoming (key runtime ⟨.incoming,a⟩) else 0
  outputBytes : ∀ x,x∈cells c S Bool.true → ∀ i : Fin 4,
    api.observe out.memory (key runtime x).owner ((key runtime x).address+i.val)=encode
      (Native.observe Native.software schedule
        (Closed.first Native.sourceOps c BDF U S (terms api runtime initial)) (name x.role) x.offset) i
  gradientBytes : ∀ x,x∈cells c S Bool.false → ∀ i : Fin 4,
    api.observe grad.memory (key runtime x).owner ((key runtime x).address+i.val)=encode
      (Native.observe Native.software schedule
        (Closed.second Native.sourceOps c BDF BDB U S hasDo V (terms api runtime initial)
          (fun a => .leaf "do" a (api.read incoming (key runtime ⟨.incoming,a⟩)))
          (fun a => .leaf "dht" a (api.read incoming (key runtime ⟨.dht,a⟩)))) (name x.role) x.offset) i
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

theorem fp_complete {M : Type} (api : API M Word Byte) (codec : Codec api) (p : Native.Instructions)
    (laws : Native.LocalConformance p) (schedule : Native.Schedule) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true) (BDF BDB : Nat)
    (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true) (initial incoming : M)
    (saved : SavedStorage api runtime (nativeForward api p schedule c U S V runtime BDF initial).memory incoming)
    (zero : MissingZero api 0 c U S V hasDo runtime incoming) :
    FPClaim api p schedule c U S V hasDo runtime BDF BDB initial incoming
      (nativeForward api p schedule c U S V runtime BDF initial)
      (nativeBackward api p schedule c U S V hasDo runtime BDF BDB initial incoming) := by
  have first := Closed.first_complete Native.sourceOps c BDF (tile_bounds BDF tileF).1 U S (terms api runtime initial)
  have second := Closed.complete Native.sourceOps c BDF BDB (tile_bounds BDF tileF).1 (tile_bounds BDB tileB).1
    U S hasDo V (terms api runtime initial)
    (fun a => .leaf "do" a (api.read incoming (key runtime ⟨.incoming,a⟩)))
    (fun a => .leaf "dht" a (api.read incoming (key runtime ⟨.dht,a⟩)))
  refine ⟨⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩,checked_safety c U S V runtime accepted BDF tileF,
    checked_safety c U S V runtime accepted BDB tileB,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro bh t d hb ht hd
    dsimp only [nativeForward,returnedForward]
    rw [readback api c U S V runtime accepted Bool.true _ initial _ (forward_row_member c S bh t d hb ht hd)]
    change Native.observe p schedule _ "o" _=_
    rw [Native.observe,first.2.1 bh hb t ht d hd]
    exact Native.eval_software p laws schedule _
  · intro hs bh d hb hd
    dsimp only [nativeForward,returnedForward]
    rw [readback api c U S V runtime accepted Bool.true _ initial _ (forward_slot_member c S hs bh d hb hd)]
    change Native.observe p schedule _ "ht" _=_
    rw [Native.observe,first.2.2.1 hs bh hb d hd]
    exact Native.eval_software p laws schedule _
  · intro bh t d hb ht hd
    have values := second.2.2.1 bh hb t ht d hd
    constructor
    · dsimp only [nativeBackward,returnedBackward]
      rw [readback api c U S V runtime accepted Bool.false _ incoming _
        (backward_row_member c S .dx (Or.inl rfl) bh t d hb ht hd)]
      change Native.observe p schedule _ "dx" _=_
      rw [Native.observe,values.1]
      exact Native.eval_software p laws schedule _
    · dsimp only [nativeBackward,returnedBackward]
      rw [readback api c U S V runtime accepted Bool.false _ incoming _
        (backward_row_member c S .dg (Or.inr rfl) bh t d hb ht hd)]
      change Native.observe p schedule _ "dg" _=_
      rw [Native.observe,values.2]
      exact Native.eval_software p laws schedule _
  · exact native_saved api p schedule c U S V runtime accepted BDF (tile_bounds BDF tileF).1 initial incoming saved
  · intro a ha
    cases hdo : hasDo
    · exact zero hdo a ha
    · rfl
  · intro x hx i
    have active := (cells_active c U S V Bool.true x hx).2
    rw [codec.read _ _ (key_bounds c U S V runtime accepted x active).2.2.2]
    dsimp only [nativeForward,nativeBackward,returnedForward,returnedBackward]
    rw [readback api c U S V runtime accepted Bool.true _ initial x hx]
    exact congrArg (fun w => encode w i) (Native.eval_software p laws schedule _)
  · intro x hx i
    have active := (cells_active c U S V Bool.false x hx).2
    rw [codec.read _ _ (key_bounds c U S V runtime accepted x active).2.2.2]
    dsimp only [nativeForward,nativeBackward,returnedForward,returnedBackward]
    rw [readback api c U S V runtime accepted Bool.false _ incoming x hx]
    exact congrArg (fun w => encode w i) (Native.eval_software p laws schedule _)
  · intro x hx
    have active := (cells_active c U S V Bool.true x hx).2
    have writes := (cells_mem c S Bool.true x).mp hx
    have role : x.role=.o ∨ x.role=.ht := by
      rcases writes with h|h
      · exact Or.inl h.1
      · exact Or.inr h.2.1
    dsimp only [nativeBackward,returnedBackward]
    rw [backward_keeps_outputs api c U S V runtime accepted _ incoming x active role]
    exact saved.read x.role (by rcases role with h|h <;> simp [h]) active.1 x.offset active.2
  · dsimp only [nativeForward,returnedForward]
    exact frame api c U S V runtime Bool.true _ initial
  · dsimp only [nativeBackward,returnedBackward]
    exact frame api c U S V runtime Bool.false _ incoming

  · intro phase r hr address lo hi
    cases phase
    · simp only [Bool.false_eq_true,↓reduceIte]
      dsimp only [nativeBackward,returnedBackward]
      exact caller_frame api c U S V runtime accepted Bool.false _ incoming r hr address lo hi
    · simp only [↓reduceIte]
      dsimp only [nativeForward,returnedForward]
      exact caller_frame api c U S V runtime accepted Bool.true _ initial r hr address lo hi

#axiomsClean fp_complete
#specNonCircular Codec avoiding [Execution.gridRun,Closed.first,Closed.second,state,dx,dg]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Public

import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Host

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
open Checked Binding Host
noncomputable section
open scoped BigOperators

def tileTarget (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool)
    (fwd bwd : Inputs α) (p t : Nat) : α :=
  if t+1<c.N then Reduction.product ops (trees.tile p t)
    (fun j => (Reference.reverse ops c bwd p (c.N-1-t) j,Reference.forward ops c fwd seed p t j))
  else ops.zero

def gateTarget (ops : Ops α) (c : Shape) (trees : Trees c) (seed : Bool)
    (fwd bwd : Inputs α) (h t : Nat) : α :=
  Reduction.eval ops (trees.host h t) (fun q => tileTarget ops c trees seed fwd bwd (h*c.tiles+q) t)

theorem saved_product {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c)
    (call : Invocation api ops c seed runtime) (p t : Nat) (hp : p<c.programs) (ht : t<c.N) :
    Source.savedProduct ops c trees (mirror api c seed runtime call.backwardPrepared) p t=
      tileTarget ops c trees seed call.forwardInputs call.backwardInputs p t := by
  unfold Source.savedProduct tileTarget
  split_ifs with h
  · unfold Source.tileValue
    congr 1
    funext j
    rw [invocation_saved api ops c seed runtime accepted call p t j hp ht j.isLt,
      Preparation.reverse_reads api ops c seed runtime call.backwardCaller call.backwardBefore
        call.backwardPrepared call.backwardCopies p (c.N-1-t) j hp (by omega) j.isLt]
    rfl
  · rfl

structure Claim {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (call : Invocation api ops c seed runtime)
    (out : Forward M) (grad : Backward M) : Prop where
  returned : out.output=runtime.tensor .output ∧ out.savedOutput=out.output ∧
    out.savedGate=runtime.tensor .gate ∧ grad.dx=runtime.tensor .dx ∧ grad.dgate=runtime.tensor .dgate ∧
    grad.dseed=(if seed then some (runtime.tensor .dseed) else none)
  launches : out.launched=decide (0<c.N) ∧ grad.launched=decide (0<c.N)
  machine : Machine.Domain c seed runtime.abi
  gateUnchanged : ∀ h t,h<c.heads → t<c.N → call.backwardInputs.gate h t=call.forwardInputs.gate h t
  output : ∀ p t j,p<c.programs → t<c.N → j<1024 →
    api.read out.memory (key c seed runtime ⟨.output,p,t,j⟩)=Reference.forward ops c call.forwardInputs seed p t j
  savedOutput : ∀ p t j,p<c.programs → t<c.N → j<1024 →
    api.read grad.memory (key c seed runtime ⟨.output,p,t,j⟩)=Reference.forward ops c call.forwardInputs seed p t j
  dx : ∀ p t j,p<c.programs → t<c.N → j<1024 →
    api.read grad.memory (key c seed runtime ⟨.dx,p,t,j⟩)=Reference.dx ops c call.backwardInputs p t j
  dgate : ∀ h t,h<c.heads → t<c.N →
    api.read grad.memory (key c seed runtime ⟨.dgate,h,t,0⟩)=
      gateTarget ops c trees seed call.forwardInputs call.backwardInputs h t
  dseed : ∀ p j,p<c.programs → j<1024 → (0<c.N ∨ seed=Bool.true) →
    api.read grad.memory (key c seed runtime ⟨.dseed,p,0,j⟩)=Reference.reverse ops c call.backwardInputs p c.N j
  forwardFrame : ∀ owner address,StageOutside c seed runtime Bool.true owner address →
    api.observe out.memory owner address=api.observe call.forwardPrepared owner address
  backwardFrame : ∀ owner address,StageOutside c seed runtime Bool.false owner address →
    api.observe grad.memory owner address=api.observe call.backwardPrepared owner address

theorem complete {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c)
    (call : Invocation api ops c seed runtime) :
    Claim api ops c seed runtime trees call
      (forward api ops c seed runtime call.forwardPrepared)
      (backward api ops c seed runtime trees call.backwardPrepared) := by
  refine ⟨⟨rfl,rfl,rfl,rfl,rfl,rfl⟩,⟨rfl,rfl⟩,machine_domain c seed runtime accepted,
    invocation_gate api ops c seed runtime accepted call,?_,?_,?_,?_,?_,?_,?_⟩
  · intro p t j hp ht hj
    rw [forward_read api ops c seed runtime accepted call.forwardPrepared p t j hp ht hj]
    exact Preparation.forward_reads api ops c seed runtime call.forwardCaller call.forwardBefore
      call.forwardPrepared call.forwardCopies p t j hp ht hj
  · intro p t j hp ht hj
    change api.read (stageCommit _ _ _ _ _ _ _) _=_
    rw [backward_keeps_output api c seed runtime accepted _ call.backwardPrepared p t j hp ht hj]
    exact invocation_saved api ops c seed runtime accepted call p t j hp ht hj
  · intro p t j hp ht hj
    rw [backward_read api ops c seed runtime accepted trees call.backwardPrepared ⟨.dx,p,t,j⟩
      (Or.inr (Or.inl ⟨rfl,hp,ht,hj⟩)) (by simp)]
    simp only [Source.backwardTarget,true_and,hp,ht,hj,↓reduceIte]
    unfold Reference.dx
    split_ifs with h
    · exact Preparation.reverse_reads api ops c seed runtime call.backwardCaller call.backwardBefore
        call.backwardPrepared call.backwardCopies p (c.N-1-t) j hp (by omega) hj
    · rfl
  · intro h t hh ht
    rw [backward_read api ops c seed runtime accepted trees call.backwardPrepared ⟨.dgate,h,t,0⟩
      (Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl,hh,ht,rfl⟩)))) (by simp)]
    simp only [Source.backwardTarget,reduceCtorEq,false_and,true_and,hh,ht,↓reduceIte]
    unfold gateTarget
    congr 1
    funext q
    exact saved_product api ops c seed runtime accepted trees call _ t (c.program_bound h q hh q.isLt) ht
  · intro p j hp hj hs
    rw [backward_read api ops c seed runtime accepted trees call.backwardPrepared ⟨.dseed,p,0,j⟩
      (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl,hp,rfl,hj,hs⟩)))) (by simp)]
    simp only [Source.backwardTarget,reduceCtorEq,false_and,true_and,hp,hj,hs,↓reduceIte]
    exact Preparation.reverse_reads api ops c seed runtime call.backwardCaller call.backwardBefore
      call.backwardPrepared call.backwardCopies p c.N j hp (by omega) hj
  · exact stage_frame api c seed runtime Bool.true _ call.forwardPrepared
  · exact stage_frame api c seed runtime Bool.false _ call.backwardPrepared

theorem gate_real (c : Shape) (trees : Trees c) (seed : Bool) (fwd bwd : Inputs ℝ)
    (h t : Nat) (ht : t<c.N) : gateTarget realOps c trees seed fwd bwd h t=
    ∑ q : Fin c.tiles,∑ j : Fin 1024,MathTarget.reverse c bwd (h*c.tiles+q) (t+1) j*
      MathTarget.forward c fwd seed (h*c.tiles+q) t j := by
  rw [gateTarget,Reduction.real]
  apply Finset.sum_congr rfl
  intro q _
  unfold tileTarget
  split_ifs with active
  · rw [Reduction.product_real]
    apply Finset.sum_congr rfl
    intro j _
    rw [RealProof.reverse c bwd _ _ j (by omega),show c.N-(c.N-1-t)=t+1 by omega,RealProof.forward]
  · simp [show t+1=c.N by omega,realOps]

#axiomsClean complete
#axiomsClean gate_real
#specNonCircular gateTarget avoiding [Source.run,Source.backwardRun,Host.backward]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Public

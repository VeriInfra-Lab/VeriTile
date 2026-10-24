import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Results

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
open Checked Binding Host
noncomputable section
open scoped BigOperators

/-- An arbitrary-Real interpretation of both actual wrapper phases. The
caller values are independent Reals, never decoded or rounded FP32 words. -/
structure RealClaim {M β : Type} (api : API M ℝ β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (call : Invocation api realOps c seed runtime)
    (out : Forward M) (grad : Backward M) : Prop extends Claim api realOps c seed runtime trees call out grad where
  realOutput : ∀ p t j,p<c.programs → t<c.N → j<1024 →
    api.read out.memory (key c seed runtime ⟨.output,p,t,j⟩)=MathTarget.forward c call.forwardInputs seed p t j
  realDX : ∀ p t j,p<c.programs → t<c.N → j<1024 →
    api.read grad.memory (key c seed runtime ⟨.dx,p,t,j⟩)=MathTarget.reverse c call.backwardInputs p (t+1) j
  realDGate : ∀ h t,h<c.heads → t<c.N →
    api.read grad.memory (key c seed runtime ⟨.dgate,h,t,0⟩)=
      ∑ q : Fin c.tiles,∑ j : Fin 1024,MathTarget.reverse c call.backwardInputs (h*c.tiles+q) (t+1) j*
        MathTarget.forward c call.forwardInputs seed (h*c.tiles+q) t j
  realDSeed : ∀ p j,p<c.programs → j<1024 → (0<c.N ∨ seed=Bool.true) →
    api.read grad.memory (key c seed runtime ⟨.dseed,p,0,j⟩)=MathTarget.reverse c call.backwardInputs p 0 j

theorem real_complete {M β : Type} (api : API M ℝ β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c)
    (call : Invocation api realOps c seed runtime) :
    RealClaim api c seed runtime trees call (forward api realOps c seed runtime call.forwardPrepared)
      (backward api realOps c seed runtime trees call.backwardPrepared) := by
  have source := complete api realOps c seed runtime accepted trees call
  refine ⟨source,?_,?_,?_,?_⟩
  · intro p t j hp ht hj
    rw [source.output p t j hp ht hj,RealProof.forward]
  · intro p t j hp ht hj
    rw [source.dx p t j hp ht hj,RealProof.dx c call.backwardInputs p t j ht]
  · intro h t hh ht
    rw [source.dgate h t hh ht,gate_real c trees seed call.forwardInputs call.backwardInputs h t ht]
  · intro p j hp hj hs
    rw [source.dseed p j hp hj hs,RealProof.reverse c call.backwardInputs p c.N j (by omega)]
    simp

#axiomsClean real_complete
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Public

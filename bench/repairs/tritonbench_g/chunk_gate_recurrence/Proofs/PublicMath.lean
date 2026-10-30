import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Whole
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Math
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Safety
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Kernels
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Correct
noncomputable section

/-- Independent whole operation mathematics. The exclusive prefix, suffix
adjoint, complete feature contraction and all return/side-output frame cells
are explicit. The FP32 bit theorem is separate in FPEquiv.lean. -/
structure Result (c : Shape) (seed : Bool) (initial final : Memory ℝ) : Prop where
  output : ∀ p t j, p<c.programs → t<c.N → j<1024 →
    final .output p t j=MathTarget.forward c (inputsOf initial) seed p t j
  inputGradient : ∀ p t j, p<c.programs → t<c.N → j<1024 →
    final .dx p t j=MathTarget.reverse c (inputsOf initial) p (t+1) j
  gateGradient : ∀ h t, h<c.heads → t<c.N →
    final .dgate h t 0=MathTarget.dgate c (inputsOf initial) seed h t
  seedGradient : ∀ p j, p<c.programs → j<1024 → (c.N>0 ∨ seed) →
    final .dseed p 0 j=MathTarget.reverse c (inputsOf initial) p 0 j
  frame : ∀ r p t j, ¬Source.writes c seed r p t j → final r p t j=initial r p t j

/-- Both full grids, complete source loops, DL postlude, saved output flow,
last-slice host zeroing and host contraction implement the independent target.
No supplied carry, saved-output equation, reduction answer or FP atom occurs
in this theorem. N=0 and N=1 are included. -/
theorem real_complete (c : Shape) (trees : Trees c) (seed : Bool) (initial : Memory ℝ) :
    Result c seed initial (Source.run realOps c trees seed initial) := by
  constructor
  · intro p t j hp ht hj
    rw [Source.run_complete]
    simp only [Source.completeImage,reduceCtorEq,true_and,hp,ht,hj,↓reduceIte]
    exact RealProof.forward c (inputsOf initial) seed p t j
  · intro p t j hp ht hj
    rw [Source.run_complete]
    simp only [Source.completeImage,reduceCtorEq,false_and,true_and,hp,ht,hj,↓reduceIte]
    exact RealProof.dx c (inputsOf initial) p t j ht
  · intro h t hh ht
    rw [Source.run_complete]
    simp only [Source.completeImage,reduceCtorEq,false_and,true_and,hh,ht,↓reduceIte]
    exact RealProof.dgate c trees (inputsOf initial) seed h t ht
  · intro p j hp hj hs
    rw [Source.run_complete]
    simp only [Source.completeImage,reduceCtorEq,false_and,true_and,hp,hj,hs,↓reduceIte]
    simpa using RealProof.reverse c (inputsOf initial) p c.N j (by omega)
  · exact Source.run_frame realOps c trees seed initial

theorem real_access_safe (c : Shape) (trees : Trees c) (seed : Bool) (initial : Memory ℝ)
    (abi : Machine.ABI) (machine : Machine.Domain c seed abi) :
    Result c seed initial (Source.run realOps c trees seed initial) ∧
    (∀ r p t j, Safety.Access c seed r p t j → (abi.allocation r).Contains (Machine.offset c abi r p t j)) :=
  ⟨real_complete c trees seed initial,
    fun r p t j access => (Safety.all_accesses_safe c seed abi machine r p t j access).1⟩

#axiomsClean real_complete
#axiomsClean real_access_safe
#specNonCircular MathTarget.forward avoiding [Source.run,Result]
#specNonCircular MathTarget.reverse avoiding [Source.run,Result]
#specNonCircular MathTarget.dgate avoiding [Source.run,Result]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Correct

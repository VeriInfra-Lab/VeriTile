import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Runtime
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelBinding
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Correct
open VeriTile Triton FP.Structural Binding
noncomputable section

def idealResult (r : Request) (x : Tensor) (memory : Nat → ℝ) (k : Nat) : ℝ :=
  Spec.result r (x.read memory) (batchIndex r k) (rowIndex r k) (columnIndex r k)

/-- Complete selected DSL success, independent real rotation on the same input
Tensor, all output elements, all physical words, and the full region frame. -/
specification kernel_correct (r : Request) (x y : Tensor) (input : Input x r) (out : Output x y)
    (memory : Nat → ℝ) (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    VeriTile.Spec.Real (Contract.checkInputs x.actual.metadata ⟨Bool.true,r.position⟩=.ok () ∧
      ∃ final, Runtime.run ideal r "X" "Y" qs (initial x y memory)=some final ∧
      (∀ k,k<y.size → y.read (commit y memory final) k=idealResult r x memory k) ∧
      commit y memory final=reference y memory (idealResult r x memory) ∧
      (∀ a,¬footprint y a → commit y memory final a=memory a) ∧
      (∀ (region : RegionName) (o : Nat),region≠"Y" ∨ r.size≤o → final.mem region o=(initial x y memory).mem region o)) := by
  refine ⟨(Contract.checkInputs_iff _ _).mpr input.admitted,?_⟩
  obtain ⟨final,ran,values,frame⟩ := Runtime.whole ideal r "X" "Y" qs all (initial x y memory) (by decide)
  have flat : ∀ k,k<y.size → (final.mem "Y" k).read .fp32=idealResult r x memory k := by
    intro k hk
    have coords := coordinates r k (by rw [←input.size,←out.size]; exact hk)
    have value := values (batchIndex r k) (rowIndex r k) (columnIndex r k) coords.1 coords.2.1 coords.2.2.1
    rw [coords.2.2.2,scalar_logical ideal r x y input memory _ _ _ coords.1 coords.2.1 coords.2.2.1,
      ideal_result] at value
    exact value
  refine ⟨final,ran,?_,commit_reference y memory final _ flat,commit_frame y memory final,frame⟩
  intro k hk
  rw [y.read_view _ k hk,commit_value y memory final k hk]
  exact flat k hk

#guard_msgs (drop info) in
#specNonCircular idealResult avoiding [Source.source,Selected.source,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Correct

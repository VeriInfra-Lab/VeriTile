import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelInterface
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.RmsnormFused.RawNative
open VeriTile Triton FP.Structural Ordered
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
noncomputable section

theorem kernel_native (shape : Shape) (valid : shape.Valid)
    (I : Instructions) (P : Policy) (R : Reductions) (epsilonValue : ℝ)
    (epsilonWord : Word) (_abi : I.scalarABI epsilonValue=epsilonWord)
    (memory : RegionName → Nat → Word) (initial : State (Term Word))
    (input : ∀ i,Runtime.read initial "X" i=.input (memory "X" i))
    (weight : ∀ i,Runtime.read initial "W" i=.input (memory "W" i))
    (schedule : List Nat) (all : schedule.Perm shape.grid) :
    ∃ final, Runtime.run graphAlgebra "X" "Y" "W" shape.rows shape.width shape.block
      (.input epsilonWord) schedule initial=some final ∧
      (∀ row col,row<shape.rows → col<shape.width →
        interpret I P R col (Runtime.read final "Y" (row*shape.width+col))=
          Ordered.reference I P R shape.width shape.block epsilonWord
            (fun j => memory "X" (row*shape.width+j)) (memory "W") col) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,values,frame⟩ := KernelInterface.complete graphAlgebra shape valid
    (.input epsilonWord) schedule all initial
  refine ⟨final,ran,?_,frame⟩
  intro row col hr hc
  rw [values row col hr hc]
  exact source_lowering I P R memory shape.width shape.block row col epsilonWord initial input weight hc

#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.RawNative

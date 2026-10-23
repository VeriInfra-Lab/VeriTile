import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Real
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Host

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.HostCorrect
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
noncomputable section

/-- Independent last-axis softmax of input plus the optional additive mask. -/
def reference : Formula ℝ := fun n _ xs mask c =>
  let logits := fun j => xs j+(mask.getD (fun _ => 0)) j
  Real.exp (logits c)/∑ j ∈ Finset.range n,Real.exp (logits j)

def memory : Memory RealModel.access where
  replace := fun s cells => {s with mem := cells}
  cells := fun _ _ => rfl

specification python_checks_iff_preconditions (x : ArgumentMetadata) (mask : Option ArgumentMetadata) (dim : Int) :
    checkInput x mask dim=.ok () ↔ AdmittedArguments x mask dim := checkInput_iff x mask dim

specification softmax_wrapper_correct (layout : Layout) (x : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (before : World BlockState) (valid : Valid layout before (descriptor x))
    (maskValid : ∀ m,mask=some m → Valid layout before (descriptor m))
    (allocation : Allocation RealModel.access layout x before) :
    ∃ final,runWrapper RealModel.access memory layout RealModel.evaluate x mask dim before allocation=.ok allocation.output final ∧
      Preserves RealModel.access before final ∧ Valid layout final (descriptor allocation.output) ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue RealModel.access Neg.neg final.program allocation.output r c=
          reference (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue RealModel.access Neg.neg before.program x r)
            (maskValues RealModel.access Neg.neg before.program mask r) c) :=
  returned_tensor RealModel.access memory Neg.neg layout RealModel.evaluate reference RealModel.source
    RealModel.terminates x mask dim h before valid maskValid allocation

#specNonCircular reference avoiding [softmax_kernel,runWrapper,RealModel.evaluate]
#axiomsClean python_checks_iff_preconditions
#axiomsClean softmax_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.HostCorrect

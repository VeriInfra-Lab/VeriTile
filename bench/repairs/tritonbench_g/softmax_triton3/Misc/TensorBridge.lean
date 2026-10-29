import bench.repairs.tritonbench_g.softmax_triton3.Misc.NativeCorrect
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_triton3.Correct
import bench.repairs.tritonbench_g.softmax_triton3.FPEquiv
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.KernelBinding
import VeriTile.Meta.StatementAudit

/-! Optional correspondence between the Python guard and its input contract. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
open VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
noncomputable section

def realMemory : Memory RealModel.access where
  replace := fun s cells => {s with mem := cells}
  cells := fun _ _ => rfl

def nativeMemory : Memory (@Floating.access Word inferInstance) where
  replace := fun s cells => {s with mem := cells}
  cells := fun _ _ => rfl


theorem python_checks_iff_preconditions (x : ArgumentMetadata) (mask : Option ArgumentMetadata) (dim : Int) :
    checkInput x mask dim=.ok () ↔ AdmittedArguments x mask dim := checkInput_iff x mask dim

/-- Every valid program ID successfully executes the complete source and writes
the independent Real formula. The source loads are bound to the actual Tensor
storage, offsets and logical indices; all exterior memory cells are preserved.
No host execution, allocation or desired-output hypothesis is required. -/
theorem softmax_kernel_correct (x y : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (output : TensorContract y.metadata) (shape : y.metadata.shape=x.metadata.shape)
    (read : Nat → ℝ) (s : BlockState) (input : InputMemory RealModel.access read x s)
    (masks : ∀ m,mask=some m → InputMemory RealModel.access read m s)
    (p : Nat) (hp : p<rows x.metadata) :
    Spec.Real (∃ t,physicalEvaluate realMemory RealModel.evaluate x y mask
        (blockSize (cols x.metadata)) p s=some t ∧
      (∀ c,c<cols x.metadata →
        RealModel.access.read t y.region (y.tensor.physicalIndex (p*cols y.metadata+c))=
          Correct.reference (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue Neg.neg read x p) (logicalMask Neg.neg read mask p) c) ∧
      FrameRows RealModel.access (descriptor y) [p] s t) :=
  checked_row RealModel.access realMemory Neg.neg RealModel.evaluate Correct.reference RealModel.source
    RealModel.terminates x y mask dim h output shape read s input masks p hp


/-- Complete successful FP32 source execution agrees with the independent
native graph at every active output word. The same physical-address reader
binds input and mask storage, including overlapping views. Every cell outside
the row's output write set retains its full raw Cell and tag. -/
theorem softmax_kernel_fp32 (inst : Instructions) (profile : Profile)
    (x y : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (output : TensorContract y.metadata) (shape : y.metadata.shape=x.metadata.shape)
    (read : Nat → Word) (s : State Word) (input : InputMemory Floating.access read x s)
    (masks : ∀ m,mask=some m → InputMemory Floating.access read m s)
    (p : Nat) (hp : p<rows x.metadata) :
    ∃ t,physicalEvaluate nativeMemory (NativeGraph.evaluate inst profile.vector) x y mask
        (blockSize (cols x.metadata)) p s=some t ∧
      (∀ c,c<cols x.metadata →
        Floating.access.read t y.region (y.tensor.physicalIndex (p*cols y.metadata+c))=
          FPEquiv.reference inst profile (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue FPEquiv.negateWord read x p) (logicalMask FPEquiv.negateWord read mask p) c) ∧
      FrameRows Floating.access (descriptor y) [p] s t :=
  checked_row Floating.access nativeMemory FPEquiv.negateWord (NativeGraph.evaluate inst profile.vector)
    (FPEquiv.reference inst profile) (NativeGraph.source inst profile.vector) (NativeGraph.terminates inst profile.vector)
    x y mask dim h output shape read s input masks p hp


#axiomsClean softmax_kernel_correct
#axiomsClean softmax_kernel_fp32
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.TensorBridge

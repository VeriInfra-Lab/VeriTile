import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelContract
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

/-! Native observation of the actual source term graph. Reduction trees,
FP32 FMA contraction and multiplication operand order are selected compiler
metadata. They never inspect numerical words or assert split/FMA equality. -/
namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelProfile
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

namespace Compiler
abbrev Term := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Term
abbrev Reductions := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Reductions
abbrev Instructions := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Instructions
abbrev Schedule := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Schedule
abbrev algebra := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.algebra
abbrev eval := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.eval
abbrev shape := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.shape
end Compiler

/-- Runtime scalar leaves preserve their raw ABI words. Their structural
identities, rather than numerical payloads, are visible to compiler selection. -/
def arguments (scalars : RuntimeArgs.Values Word) : RuntimeArgs.Values Compiler.Term :=
  ⟨⟨.leaf "abi.smoothing" 0 scalars.smoothing,
   .leaf "abi.lse_square_scale" 0 scalars.squareScale⟩,.leaf "abi.logit_scale" 0 scalars.logitScale⟩

/-- Actual Tensor words and sign-extended int32 labels seed the source term graph.
The three runtime registers are ordinary ABI inputs, not real literals. -/
def entry (p : Parameters) (scalars : RuntimeArgs.Values Word) (words : State Word) : State Compiler.Term :=
  let s : State Compiler.Term := {
    mem := fun r i => if r=Region.cast p.labels then .mk .int ((words.mem r i).read .int)
      else .mk .real (.leaf r i ((words.mem r i).read .real))
    regs := fun _ _ _ => none
    pids := words.pids
    numPids := words.numPids
    undef := fun d _ _ => defaultValue d }
  let s := s.setReg "smoothing" .real [] (fun _ => (arguments scalars).smoothing)
  let s := s.setReg "lse_square_scale" .real [] (fun _ => (arguments scalars).squareScale)
  let s := s.setReg "logit_scale" .real [] (fun _ => (arguments scalars).logitScale)
  s.setReg "total_classes" .int [] (fun _ => (p.totalClasses : Int))

theorem entry_binding (p : Parameters) (scalars : RuntimeArgs.Values Word) (words : State Word) :
    RuntimeArgs.Bound p (entry p scalars words) (arguments scalars) := by
  constructor
  · constructor <;> simp [entry,State.setReg]
  · simp [entry,State.setReg]

/-- Independent ordered expressions; no definition executes the source. -/
def forwardLSE (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) : Compiler.Term :=
  KernelNative.lseValue (Compiler.algebra R) p (entry p scalars words)

def forwardLoss (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) : Compiler.Term :=
  KernelNative.lossValue (Compiler.algebra R) p (entry p scalars words)

def forwardZ (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) : Compiler.Term :=
  KernelNative.zValue (Compiler.algebra R) p (entry p scalars words)

def backwardLane (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) (lane : Fin p.block) : Compiler.Term :=
  KernelNative.gradientValue (Compiler.algebra R) p (entry p scalars words) lane

/-- The source's LSE-square accumulation is observed as its selected FP32
FMA, without any comparison with separately rounded addition/multiplication. -/
theorem forward_square_fused (I : Compiler.Instructions) (S : Compiler.Schedule)
    (loss scale lse : Compiler.Term)
    (selected : S.fusion (.binary (some .fp32) .real .add (Compiler.shape loss)
      (.binary (some .fp32) .real .mul
        (.binary (some .fp32) .real .mul (Compiler.shape scale) (Compiler.shape lse))
        (Compiler.shape lse)))=.right) :
    Compiler.eval I S (.binary (some .fp32) .real .add loss
      (.binary (some .fp32) .real .mul (.binary (some .fp32) .real .mul scale lse) lse))=
      I.fmaRN (Compiler.eval I S (.binary (some .fp32) .real .mul scale lse))
        (Compiler.eval I S lse) (Compiler.eval I S loss) :=
  VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.right_fused I S _ _ _ selected

/-- The actual backward correction `probability + coefficient * probability`
retains the compiler's fused instruction when that site is selected. -/
theorem backward_correction_fused (I : Compiler.Instructions) (S : Compiler.Schedule)
    (coefficient probability : Compiler.Term)
    (selected : S.fusion (.binary (some .fp32) .real .add (Compiler.shape probability)
      (.binary (some .fp32) .real .mul (Compiler.shape coefficient) (Compiler.shape probability)))=.right) :
    Compiler.eval I S (.binary (some .fp32) .real .add probability
      (.binary (some .fp32) .real .mul coefficient probability))=
      I.fmaRN (Compiler.eval I S coefficient) (Compiler.eval I S probability)
        (Compiler.eval I S probability) :=
  VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.right_fused I S _ _ _ selected

#axiomsClean forward_square_fused
#axiomsClean backward_correction_fused
#specNonCircular forwardLSE avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular forwardLoss avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular forwardZ avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular backwardLane avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular Compiler.eval avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelProfile

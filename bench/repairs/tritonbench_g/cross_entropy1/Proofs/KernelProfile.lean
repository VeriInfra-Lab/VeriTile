import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelContract
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

/-! Native observation of the actual source term graph. Reduction trees,
FP32 FMA contraction and multiplication operand order are selected compiler
metadata. They never inspect numerical words or assert split/FMA equality. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelProfile
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
  ⟨.leaf "abi.smoothing" 0 scalars.smoothing,
   .leaf "abi.lse_square_scale" 0 scalars.squareScale⟩

/-- Actual Tensor words and signed-int64 labels seed the source term graph.
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
  s.setReg "total_classes" .int [] (fun _ => (p.totalClasses : Int))

theorem entry_binding (p : Parameters) (scalars : RuntimeArgs.Values Word) (words : State Word) :
    RuntimeArgs.Bound p (entry p scalars words) (arguments scalars) := by
  constructor <;> simp [entry,State.setReg]

/-- Independent ordered expressions; no definition executes the source. -/
def forwardLSE (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) : Compiler.Term :=
  KernelNative.lseValue (Compiler.algebra R) p (entry p scalars words)

def forwardLoss (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) : Compiler.Term :=
  KernelNative.lossValue (Compiler.algebra R) p (entry p scalars words)

def backwardLane (R : Compiler.Reductions) (p : Parameters)
    (scalars : RuntimeArgs.Values Word) (words : State Word) (lane : Fin p.block) : Compiler.Term :=
  KernelNative.gradientValue (Compiler.algebra R) p (entry p scalars words) lane

/-- Complete successful typed source execution, exact native output words,
and the complete tagged memory frame. The selected schedule retains enabled
FMA lowering and operand order without equating it to separately rounded ops. -/
theorem forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := entry p inputs.scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S (fpValue final views.lse (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardLSE R p inputs.scalars words) ∧
      Compiler.eval I S (fpValue final views.loss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardLoss R p inputs.scalars words) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    (Compiler.algebra R) p (entry p inputs.scalars words) inputs.parameters.geometry.widthPositive
    views.lseLogits views.distinctOutputs (arguments inputs.scalars) (entry_binding p inputs.scalars words)
  simp only [entry,State.setReg,pids.1,pids.2] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · apply congrArg (Compiler.eval I S)
    simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,
      views.lseView.regionEq,forwardLSE,entry,State.setReg] using lseRead
  · apply congrArg (Compiler.eval I S)
    simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,
      views.lossView.regionEq,forwardLoss,entry,State.setReg] using lossRead

/-- Native observation of every active backward lane, including ignored
literal zeros, int64 labels, masked tails, runtime signed zeros and in-place
storage. The full memory frame follows from the actual source stores. -/
theorem backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := entry p inputs.scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        Compiler.eval I S (fpValue final views.gradient (launch.row*p.cols+(launch.block*p.block+lane.val)))=
          Compiler.eval I S (backwardLane R p inputs.scalars words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs
    (Compiler.algebra R) p (entry p inputs.scalars words)
    (arguments inputs.scalars) (entry_binding p inputs.scalars words)
  simp only [KernelNative.gradientOffset,entry,State.setReg,pids.1,pids.2] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  apply congrArg (Compiler.eval I S)
  simpa only [fpValue,matrix_index _ _ _ _ _ _ _ _ views.gradientView launch.rowBound active,
    views.gradientView.regionEq,backwardLane,entry,State.setReg] using reads lane active

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

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_square_fused
#axiomsClean backward_correction_fused
#specNonCircular forwardLSE avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular forwardLoss avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular backwardLane avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#specNonCircular Compiler.eval avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelProfile

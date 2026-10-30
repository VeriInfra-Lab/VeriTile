import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ScalarBackend

/-! Complete three-input pipeline observations in the existing software FP32
word model, for all configurations and every word pattern. Remaining numerical
primitives are supplied explicitly; no laws about them are assumed. The view
keeps successful executions, complete output cells and both exact memory frames.
It supplies no numerical rewrite certificate and makes no hardware claim. -/

namespace VeriTile.Triton.FP.Software
open Structural

abbrev pipelineWordInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
attribute [local instance] pipelineWordInhabited

/-- The remaining-callback space is inhabited, so the universal software model
quantification is not empty. This witness only supplies unconstrained remaining
callbacks; it is not a transcendental or mixed-format reference implementation.
`scalarFP32` still supplies the actual scalar FP32 arithmetic. -/
theorem remaining_callbacks_nonempty : Nonempty (Algebra (FP.Value .fp32)) := by
  exact ⟨{
    literal := fun _ _ _ => FP.Value.zero .fp32
    negInf := FP.Value.infinity .fp32 true
    binary := fun _ _ _ _ _ => FP.Value.zero .fp32
    unary := fun _ _ _ => FP.Value.zero .fp32
    cast := fun _ _ _ x => x
    fromNat := fun _ _ => FP.Value.zero .fp32
    fromInt := fun _ _ => FP.Value.zero .fp32
    fp32Bits := fun bits => ⟨bits.bits⟩
    fp32Load := id
    reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
    reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32 }⟩

structure PipelineIO₃ where
  io : KernelIO₃

def PipelineEquivalent₃ (lhs rhs : PipelineIO₃) : Prop :=
  PrivateScratch lhs.io ∧ PrivateScratch rhs.io ∧
  ∀ (cfg : FP.Config) (rest : Algebra (FP.Value .fp32)) (s : State (FP.Value .fp32)),
    ∃ a b, Structural.exec (scalarFP32 rest cfg) lhs.io.kernel s = some a ∧
      Structural.exec (scalarFP32 rest cfg) rhs.io.kernel s = some b ∧
      (∀ i : Fin lhs.io.Bout,
        a.mem lhs.io.out (lhs.io.write (s.pids 0) + i.val) =
          b.mem rhs.io.out (rhs.io.write (s.pids 0) + i.val)) ∧
      Frame lhs.io s a ∧ Frame rhs.io s b

/-- No arithmetic rewrite certificate is provided by this view. Its
structural relation fixes the existing FP32 software word interpretation. -/
abbrev pipelineTheory : Spec.StatementTheory ComputeStmt where
  modelID := "VeriTile.Software.scalarFP32/all-configs/opaque-other-primitives"
  Certified := fun _ _ => False

instance : Spec.ProgramSyntax PipelineIO₃ where
  Statement := ComputeStmt
  theory := pipelineTheory
  Signature := IO₃Signature
  signature := fun p => ioSignature p.io
  body := fun p => p.io.kernel.surfaceBody
  structural := some PipelineEquivalent₃
  sameContext := fun lhs rhs => lhs.io.scratch = rhs.io.scratch
end VeriTile.Triton.FP.Software

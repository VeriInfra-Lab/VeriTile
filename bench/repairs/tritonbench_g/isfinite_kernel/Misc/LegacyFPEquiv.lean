import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Equivalence

/-! Independent FP reader. Exact raw source bytes, FP32/FP64 precision,
signed/zero input strides, supplied output identity, finite indices, actual
byte pointers and complete memory frames are structural results. The explicit
scalar classifier interface is a primitive-model boundary, not a numerical
transformation deferral. This host theorem is optional historical work. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv
open Checked
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

theorem checked_fp (x y : Operand)
    (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ())
    (native : Bytes.Word x.metadata.precision.width → Bool)
    (localClassifier : Bytes.LocalClassifier x.metadata.precision native)
    (before : Bytes.Heap) (savedDevice : Nat) (ready : Bool)
    (launched : x.metadata.n=0 ∨ ready=Bool.true) :
    InputContract x.metadata y.metadata ∧
    ∃ result,Checked.run (Bytes.primitives x.metadata.precision.width) native
        x y before savedDevice ready=.ok result ∧
      result.output=y ∧ result.currentDevice=savedDevice ∧
      (∀ i,i<x.metadata.n → result.memory (y.metadata.outputView.address i)=
        Memory.boolByte (Bytes.reference x.metadata.precision
          (Bytes.read x.metadata.precision.width before (x.metadata.view.address i)))) ∧
      (∀ q,(∀ i,i<x.metadata.n → q≠y.metadata.outputView.address i) → result.memory q=before q) ∧
      (∀ i k,i<x.metadata.n → k<x.metadata.width →
        result.memory (x.metadata.view.address i+k)=before (x.metadata.view.address i+k)) ∧
      result.memory=Lowering.allChunks (Bytes.primitives x.metadata.precision.width)
        (Bytes.reference x.metadata.precision) x.metadata.view y.metadata.outputView x.metadata.n
        (List.range (chunkCount x.metadata.n)) before ∧
      Checked.Calls (Bytes.primitives x.metadata.precision.width) native x.metadata y.metadata := by
  have h : InputContract x.metadata y.metadata := (checkInputs_iff _ _).mp checked
  have wordWidth := input_width x.metadata h.input
  exact Checked.equivalent (Bytes.primitives x.metadata.precision.width) native (Bytes.reference x.metadata.precision)
    localClassifier x y checked wordWidth before savedDevice ready launched

theorem active_pointer (x y : ViewMetadata)
    (checked : checkInputs (.view x) (.view y)=.ok ()) (c pid lane : Nat)
    (launched : pid<Finite.grid (chunkLength x.n c)) (inTile : lane<1024)
    (active : pid*1024+lane<chunkLength x.n c) :
    Finite.pointer (x.view.shift (chunkStart c)) pid lane=x.view.address (chunkStart c+(pid*1024+lane)) ∧
    Finite.pointer (y.outputView.shift (chunkStart c)) pid lane=y.outputView.address (chunkStart c+(pid*1024+lane)) := by
  have h : InputContract x y := (checkInputs_iff _ _).mp checked
  have inside := Finite.active x.n c _ active
  refine ⟨Finite.checked_pointer Bool.false x h.input x.n _ _ pid lane rfl
    (Finite.small x.n c) launched inTile inside,?_⟩
  simpa only [Finite.output_view y h.output] using Finite.checked_pointer Bool.true y h.output x.n _ _ pid lane
    h.shape (Finite.small x.n c) launched inTile inside

theorem guard_domain (x y : ArgumentMetadata) :
    checkInputs x y=.ok () ↔ AdmittedArguments x y := checkInputs_iff x y

theorem accessed_alignment (x y : ViewMetadata)
    (checked : checkInputs (.view x) (.view y)=.ok ()) (i : Nat) (hi : i<x.n) :
    x.view.address i % x.precision.width=0 ∧ y.outputView.address i % 1=0 := by
  have h : InputContract x y := (checkInputs_iff _ _).mp checked
  rw [input_width x h.input]
  exact ⟨Finite.alignment Bool.false x h.input i hi,Nat.mod_one _⟩

theorem wrapper_outcomes (x y : Operand) (native : Bytes.Word x.metadata.precision.width → Bool)
    (before : Bytes.Heap) (savedDevice : Nat) (ready : Bool) :
    (∃ e,Checked.run (Bytes.primitives x.metadata.precision.width) native x y before savedDevice ready=.error (.input e)) ∨
    (Checked.run (Bytes.primitives x.metadata.precision.width) native x y before savedDevice ready=.error .launch ∧
      InputContract x.metadata y.metadata ∧ x.metadata.n≠0 ∧ ready=Bool.false) ∨
    (∃ result,Checked.run (Bytes.primitives x.metadata.precision.width) native x y before savedDevice ready=.ok result ∧
      InputContract x.metadata y.metadata ∧ result.output=y ∧ result.currentDevice=savedDevice) :=
  Checked.outcomes _ native x y before savedDevice ready

#axiomsClean checked_fp
#axiomsClean active_pointer
#axiomsClean guard_domain
#axiomsClean accessed_alignment
#axiomsClean wrapper_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv

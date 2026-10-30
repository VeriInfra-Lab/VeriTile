import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Host

/-! Independent mathematical Real reader for the checked direct repair.
All Real inputs are finite. The Real carrier is never encoded as an FP word;
native classification is addressed separately by FPEquiv.lean. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.Correct
open Checked
noncomputable section

theorem checked_real (x y : Operand)
    (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ())
    (before : RealStorage.Heap) (savedDevice : Nat) (ready : Bool)
    (launched : x.metadata.n=0 ∨ ready=Bool.true) :
    InputContract x.metadata y.metadata ∧
    ∃ result,Checked.run (RealStorage.primitives x.metadata.width (width_pos x.metadata))
        RealStorage.reference x y before savedDevice ready=.ok result ∧
      result.output=y ∧ result.currentDevice=savedDevice ∧
      (∀ i,i<x.metadata.n → result.memory.bytes (y.metadata.outputView.address i)=1) ∧
      (∀ q,(∀ i,i<x.metadata.n → q≠y.metadata.outputView.address i) → result.memory.bytes q=before.bytes q) ∧
      (∀ i,i<x.metadata.n → result.memory.reals (x.metadata.view.address i)=before.reals (x.metadata.view.address i)) ∧
      Checked.Calls (RealStorage.primitives x.metadata.width (width_pos x.metadata))
        RealStorage.reference x.metadata y.metadata := by
  simpa only [RealStorage.primitives,RealStorage.reference,Memory.boolByte,↓reduceIte] using
    Checked.complete (RealStorage.primitives x.metadata.width (width_pos x.metadata)) RealStorage.reference
      x y checked rfl before savedDevice ready launched

theorem successful_return_metadata (x y : Operand)
    (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ()) :
    y.metadata.shape=x.metadata.shape ∧ y.metadata.dtype=.other "bool" ∧
    y.metadata.device=x.metadata.device ∧ y.metadata.device.kind="cuda" := by
  have h : InputContract x.metadata y.metadata := (checkInputs_iff _ _).mp checked
  exact ⟨by rw [h.input.rank.1,h.output.rank.1,h.shape],h.output.dtype,h.device.symm,h.output.cuda.1⟩

theorem guard_domain (x y : ArgumentMetadata) :
    checkInputs x y=.ok () ↔ AdmittedArguments x y := checkInputs_iff x y

theorem wrapper_outcomes (x y : Operand) (before : RealStorage.Heap) (savedDevice : Nat) (ready : Bool) :
    (∃ e,Checked.run (RealStorage.primitives x.metadata.width (width_pos x.metadata)) RealStorage.reference
      x y before savedDevice ready=.error (.input e)) ∨
    (Checked.run (RealStorage.primitives x.metadata.width (width_pos x.metadata)) RealStorage.reference
      x y before savedDevice ready=.error .launch ∧ InputContract x.metadata y.metadata ∧
      x.metadata.n≠0 ∧ ready=Bool.false) ∨
    (∃ result,Checked.run (RealStorage.primitives x.metadata.width (width_pos x.metadata)) RealStorage.reference
      x y before savedDevice ready=.ok result ∧ InputContract x.metadata y.metadata ∧
      result.output=y ∧ result.currentDevice=savedDevice) :=
  Checked.outcomes _ RealStorage.reference x y before savedDevice ready

#axiomsClean checked_real
#axiomsClean successful_return_metadata
#axiomsClean guard_domain
#axiomsClean wrapper_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.Correct

import bench.repairs.tritonbench_g.attention_fwd_triton1.Contract
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked.Correct
open Whole001
noncomputable section

def idealScale (c : Shape) : ℝ := 1 / Real.sqrt c.D

def runReal (c : Shape) (args : Host.Arguments) (receipt : Host.AllocationReceipt c args)
    (pointers : Pointers receipt.abi) (store ifcond : Bool) (initial : Memory ℝ) :
    Option (Host.Tensor × Host.Tensor × Memory ℝ) :=
  if metadataChecks c args then
    if args.store=some store ∧ args.ifcond=some ifcond then
      if setupChecks c pointers then
        some (Host.outputTensor c args receipt.abi,Host.historyTensor c args receipt.abi,
          Source.run realOps c receipt.abi (idealScale c) store ifcond initial)
      else none
    else none
  else none

/-- Whole precision-erased call: public O, internal pre-update H, all logical
values and complete frames. No exact finite-word representation of 1/sqrt(D)
is needed or asserted by this independent Real endpoint. -/
specification checked_real (c : Shape) (args : Host.Arguments)
    (receipt : Host.AllocationReceipt c args) (pointers : Pointers receipt.abi)
    (store ifcond : Bool) (initial : Memory ℝ)
    (inputs : metadataChecks c args=Bool.true)
    (hst : args.store=some store) (hic : args.ifcond=some ifcond)
    (setup : setupChecks c pointers=Bool.true) :
    let final := Source.run realOps c receipt.abi (idealScale c) store ifcond initial
    VeriTile.Spec.Real (
      runReal c args receipt pointers store ifcond initial=
        some (Host.outputTensor c args receipt.abi,Host.historyTensor c args receipt.abi,final) ∧
      (∀ head row j, head<c.heads → row<c.T → j<c.D →
        final (receipt.abi.o.base+c.mainOffset head row j)=
          MathTarget.output c (inputsOf c receipt.abi initial) (idealScale c) head row j) ∧
      (store=Bool.true → ∀ head chunk i j,head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
        final (receipt.abi.h.base+c.historyOffset head chunk i j)=
          MathTarget.history c (inputsOf c receipt.abi initial) head chunk i j) ∧
      (store=Bool.false → ∀ point,receipt.abi.h.Contains point → final point=initial point) ∧
      ReadOnly c receipt.abi final initial ∧
      (∀ point,¬receipt.abi.o.Contains point → (store=Bool.false ∨ ¬receipt.abi.h.Contains point) →
        final point=initial point) ∧ Machine.Closed c receipt.abi) := by
  refine ⟨by simp [runReal,inputs,hst,hic,setup],?_,?_,?_,?_,?_,?_⟩
  · exact RealProof.full_output c receipt.abi (idealScale c) store ifcond initial
  · exact RealProof.full_history c receipt.abi (idealScale c) store ifcond initial
  · exact Source.run_history_disabled realOps c receipt.abi (idealScale c) store ifcond initial
  · exact Source.run_readOnly realOps c receipt.abi (idealScale c) store ifcond initial
  · exact Source.run_frame realOps c receipt.abi (idealScale c) store ifcond initial
  · exact Machine.closed c receipt.abi (admitted_machine c receipt.abi pointers setup)

#axiomsClean checked_real
#specNonCircular MathTarget.output avoiding [Source.run,runReal]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked.Correct

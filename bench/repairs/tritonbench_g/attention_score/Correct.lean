import bench.repairs.tritonbench_g.attention_score.Proofs.AcceptedReal
import bench.repairs.tritonbench_g.attention_score.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionScore.Correct
open VeriTile Triton Geometry Physical Whole002
noncomputable section
abbrev target := RealPhysical.logicalTarget

/-- Full real projection of the direct-input checked wrapper, including its
fresh returned Tensor, both persistent block globals and the exterior frame. -/
specification score_real (c : Shape) (dispatch : Dispatch c) (abi : ABI c)
    (allocation : Host.Allocation c abi) (physical : Nat → ℝ) (xs : LogicalInputs ℝ c)
    (direct : Normalized c abi physical xs) (observed : Contract.Observations)
    (offset size : Int) (window complement : Bool)
    (admitted : Contract.InputContract c observed offset size window)
    (pointers : Contract.PointerBinding c abi observed) :
    Spec.Real (Machine.Closed c abi offset size window ∧ ∃ result,
      Host.realReturn c dispatch abi allocation physical offset size window complement=some result ∧
      result.tensor=allocation.output ∧ result.tensor.shape=[c.Z,c.H,c.NK] ∧
      result.tensor.device=allocation.inputDevice ∧ result.tensor.fp32=Bool.true ∧
      result.blockM=c.B ∧ result.blockN=c.B ∧
      (∀ z h j,result.memory ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
        target c xs offset size (LogicalFormula.hostScale c.D) window complement z h j) ∧
      (∀ a,¬(abi.alloc "out").Contains a → result.memory a=physical a)) := by
  obtain ⟨closed,block,memory,he,hv,_,hf⟩ := Whole002.Correct.repaired_return c dispatch abi physical xs direct
    offset size window complement (Contract.domain c abi observed offset size window admitted pointers)
  refine ⟨closed,⟨allocation.output,memory,persistentBlock dispatch,persistentBlock dispatch⟩,?_,
    rfl,allocation.shape,allocation.device,allocation.precision,block,block,hv,hf⟩
  simp [Host.realReturn,he]

#specNonCircular target avoiding [Host.realReturn,FP.Structural.exec]
#axiomsClean score_real
end
end VeriTile.Bench.Repairs.AttentionScore.Correct

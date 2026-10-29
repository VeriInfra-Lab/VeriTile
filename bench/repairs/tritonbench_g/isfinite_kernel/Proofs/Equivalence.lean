import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Host

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Checked
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

theorem equivalent {M A : Type} [Inhabited A] {width : Nat} (ops : Memory.Primitives M A width)
    (compute reference : A → Bool) (localLaw : ∀ a,compute a=reference a)
    (x y : Operand) (checked : checkInputs (.view x.metadata) (.view y.metadata)=.ok ())
    (wordWidth : width=x.metadata.width) (before : M) (savedDevice : Nat) (ready : Bool)
    (launched : x.metadata.n=0 ∨ ready=Bool.true) :
    InputContract x.metadata y.metadata ∧
    ∃ result,Checked.run ops compute x y before savedDevice ready=.ok result ∧
      result.output=y ∧ result.currentDevice=savedDevice ∧
      (∀ i,i<x.metadata.n → ops.byte result.memory (y.metadata.outputView.address i)=
        Memory.boolByte (reference (ops.read before (x.metadata.view.address i)))) ∧
      (∀ q,(∀ i,i<x.metadata.n → q≠y.metadata.outputView.address i) → ops.byte result.memory q=ops.byte before q) ∧
      (∀ i k,i<x.metadata.n → k<x.metadata.width →
        ops.byte result.memory (x.metadata.view.address i+k)=ops.byte before (x.metadata.view.address i+k)) ∧
      result.memory=Lowering.allChunks ops reference x.metadata.view y.metadata.outputView x.metadata.n
        (List.range (chunkCount x.metadata.n)) before ∧ Calls ops compute x.metadata y.metadata := by
  have h : InputContract x.metadata y.metadata := (checkInputs_iff _ _).mp checked
  have resultValues := memory_complete ops compute x.metadata y.metadata h wordWidth before
  refine ⟨h,_,success ops compute x y before savedDevice checked ready launched,rfl,rfl,?_,?_,?_,?_,
    calls ops compute x.metadata y.metadata h⟩
  · intro i hi
    exact (resultValues.1 i hi).trans (congrArg Memory.boolByte (localLaw _))
  · exact resultValues.2.1
  · intro i k hi hk
    exact input_byte_frame ops compute x.metadata y.metadata h wordWidth before i k hi hk
  · exact congrArg (fun classify => Lowering.allChunks ops classify x.metadata.view y.metadata.outputView
      x.metadata.n (List.range (chunkCount x.metadata.n)) before) (funext localLaw)

#axiomsClean equivalent
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Checked

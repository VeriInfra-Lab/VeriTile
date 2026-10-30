import bench.repairs.tritonbench_g.cross_entropy2.Proofs.RawNative

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelProfile
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

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
      Compiler.eval I S (fpValue final views.zLoss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardZ R p inputs.scalars words) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := RawNative.forward R I S p (entry p inputs.scalars words) inputs.parameters.geometry.widthPositive
    views.lseLogits ⟨views.distinctOutputs,views.zDistinct⟩ (arguments inputs.scalars) (entry_binding p inputs.scalars words)
  simp only [entry,State.setReg,pids.1,pids.2] at lseRead lossRead zRead frame
  refine ⟨final,execution,?_,?_,?_,frame⟩
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,
      views.lseView.regionEq,forwardLSE,entry,State.setReg] using lseRead
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,
      views.lossView.regionEq,forwardLoss,entry,State.setReg] using lossRead
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.zLossView launch.rowBound launch.blockBound,
      views.zLossView.regionEq,forwardZ,entry,State.setReg] using zRead

/-- Native observation of every active backward lane, including ignored
literal zeros, sign-extended int32 labels, masked tails and runtime signed zeros.
The gradient uses fresh output storage. The full memory frame follows from the actual source stores. -/
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
  obtain ⟨final,execution,reads,frame⟩ := RawNative.backward R I S p (entry p inputs.scalars words)
    (arguments inputs.scalars) (entry_binding p inputs.scalars words)
  simp only [KernelNative.gradientOffset,entry,State.setReg,pids.1,pids.2] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  simpa only [fpValue,matrix_index _ _ _ _ _ _ _ _ views.gradientView launch.rowBound active,
    views.gradientView.regionEq,backwardLane,entry,State.setReg] using reads lane active

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelProfile

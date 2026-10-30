import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeValues
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Recompute s f bh iv t h u) :
    (KernelBackwardStep.recompute M s f scale bh iv t h u regs).val.mem=
      (KernelBackwardMemory.keyStore M s "dq" (plane s bh iv) t
        (KernelRecomputeValues.dq M s f scale bh iv t h u)
        (KernelBackwardMemory.keyStore M s "dk" (plane s bh iv) t
          (KernelRecomputeValues.dk M s bh iv t h u) u)).mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp only [KernelBackwardStep.recompute,State.setReg_mem]
    simp only [KernelRecomputeValues.dq,KernelRecomputeValues.dk,KernelRecomputeValues.next,
      KernelBackwardReference.dq,KernelBackwardReference.correction,KernelBackwardReference.input,
      KernelBackwardReference.valueSum,KernelReference.add,KernelReference.sub,KernelReference.mul,
      KernelReference.read,KernelReference.cast,KernelReference.zero,KernelReference.scalar,
      KernelReference.stored,KernelBackwardMemory.keyStore,keyPtr,valPtr,betaPtr,keyMask,valMask,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      Nat.add_zero,Bool.false_eq_true,ite_false,ite_true,KernelRecomputeValues.eta,
      KernelRecomputeValues.scatter_other "dk" "k" _ _ _ _ _ _ (by decide),
      KernelRecomputeValues.scatter_other "dk" "v" _ _ _ _ _ _ (by decide),
      KernelRecomputeValues.scatter_other "dk" "do" _ _ _ _ _ _ (by decide),
      KernelRecomputeValues.scatter_other "dk" "beta" _ _ _ _ _ _ (by decide)]
    apply KernelStore.masked_mem
    try simp only [State.setReg_mem]
    apply KernelStore.masked_mem
    rfl

#axiomsClean memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeMemory

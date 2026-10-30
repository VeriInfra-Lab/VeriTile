import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardValues
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelStore
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelMemory KernelStore KernelForwardValues
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u) :
    (KernelForwardStep.execute M s f scale bh iv t h u regs).val.mem=
      (KernelMemory.rowStore M s "o" bh iv s.FBV t (output M s f scale bh iv t h u)
        (KernelMemory.rowStore M s "v" bh iv s.FBV t (residual M s f bh iv t h u) u)).mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp only [KernelForwardStep.execute,State.setReg_mem]
    simp only [KernelForwardValues.output,KernelForwardValues.residual,KernelForwardValues.input,
      KernelReference.output,KernelReference.residual,KernelReference.forward,
      KernelReference.forwardInput,KernelReference.keySum,KernelReference.add,KernelReference.mul,
      KernelReference.sub,KernelReference.read,KernelReference.cast,KernelReference.zero,KernelReference.scalar,
      KernelReference.stored,KernelMemory.rowStore,keyPtr,valPtr,betaPtr,keyMask,valMask,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      Nat.add_zero,Bool.false_eq_true,ite_false,ite_true,eta]
    apply masked_mem
    simp only [State.setReg_mem]
    apply masked_mem
    rfl

#axiomsClean memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardMemory

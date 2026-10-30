import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelPrefix
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelInitial
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (u : State α) :
    (KernelPrefix.forward M s f scale u).val.mem=u.mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> rfl
theorem forward_pids (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (u : State α) :
    (KernelPrefix.forward M s f scale u).val.pids=u.pids := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> rfl
theorem backward_memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (u : State α) :
    (KernelPrefix.backward M s f scale u).val.mem=u.mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases tg <;> rfl

theorem forward_registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv : Nat) (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    KernelRegisters.Forward s f bh iv 0 (KernelReference.forwardInput M s f bh iv u).seed
      (KernelPrefix.forward M s f scale u).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial
  all_goals
    constructor
    all_goals
      simp [KernelPrefix.forward,KernelReference.forwardInput,KernelReference.add,KernelReference.zero,
        KernelReference.read,KernelReference.cast,hv,hk,hh,keyPtr,valPtr,betaPtr,keyMask,valMask,
        keyAddress,valAddress,betaAddress,stateAddress,State.setReg,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Nat.add_assoc]
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,keyMask,valMask,
    keyAddress,valAddress,betaAddress,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]

#axiomsClean forward_memory
#axiomsClean forward_registers
#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelInitial

import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelInitial
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardStep
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStatic
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def mask (s : Shape) (iv : Nat) : Values α .bool [s.FBV,s.BK] :=
  fun ji => decide (ji.2.1.val<s.K) && decide (iv*s.FBV+ji.1.val<s.V)
structure Forward (s : Shape) (bh iv : Nat) (u : State α) : Prop where
  head : u.regs .nat [] "i_bh"=some (fun _ => bh)
  key : u.regs .nat [] "i_k"=some (fun _ => 0)
  value : u.regs .nat [] "i_v"=some (fun _ => iv)
  stateMask : u.regs .bool [s.FBV,s.BK] "mask_kv"=some (mask s iv)

theorem initial (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    Forward s bh iv (KernelPrefix.forward M s f scale u).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial
  all_goals
    constructor
    all_goals simp [KernelPrefix.forward,hv,hk,hh,mask,State.setReg,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals funext ji
  all_goals simp [mask,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  all_goals rfl

theorem step (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u)
    (static : Forward s bh iv u) :
    Forward s bh iv (KernelForwardStep.execute M s f scale bh iv t h u regs).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    rcases static with ⟨head,key,value,stateMask⟩
    constructor
    all_goals simp [KernelForwardStep.execute,State.setReg,keyPtr,valPtr,betaPtr,
      head,key,value,stateMask]

#axiomsClean initial
#axiomsClean step
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStatic

import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelInitial
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardValues
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardInitial
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Static (s : Shape) (bh iv : Nat) (u : State α) : Prop where
  head : u.regs .nat [] "i_bh"=some (fun _ => bh)
  key : u.regs .nat [] "i_k"=some (fun _ => 0)
  value : u.regs .nat [] "i_v"=some (fun _ => iv)
  maskKey : u.regs .bool [s.BK] "mask_bk"=some (keyMask s)
  maskValue : u.regs .bool [s.BBV] "mask_bv"=some (valMask s iv s.BBV)

theorem registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv : Nat) (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    Reverse s f bh iv (s.T-1) (KernelBackwardReference.input M s f bh iv u).terminal
      (KernelPrefix.backward M s f scale u).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases tg
  all_goals
    constructor
    all_goals
      simp [KernelPrefix.backward,KernelBackwardReference.input,KernelReference.add,KernelReference.zero,
        KernelReference.read,KernelReference.cast,hv,hk,hh,keyPtr,valPtr,betaPtr,dbetaPtr,plane,keyMask,valMask,
        keyAddress,valAddress,betaAddress,stateAddress,State.setReg,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Nat.add_assoc]
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,dbetaPtr,plane,keyMask,valMask,
    keyAddress,valAddress,betaAddress,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]

theorem static (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    Static s bh iv (KernelPrefix.backward M s f scale u).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases tg
  all_goals
    constructor
    all_goals simp [KernelPrefix.backward,hv,hk,hh,keyMask,valMask,State.setReg,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyMask,valMask,Broadcast.leftIndex,Broadcast.rightIndex]

theorem reverse_static (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Reverse s f bh iv t h u) (static : Static s bh iv u) :
    Static s bh iv (KernelBackwardStep.reverse M s f scale bh iv t h u regs).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    rcases static with ⟨head,key,value,mk,mv⟩
    constructor
    all_goals simp [KernelBackwardStep.reverse,State.setReg,keyPtr,valPtr,betaPtr,dbetaPtr,
      head,key,value,mk,mv]

theorem recompute_static (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Recompute s f bh iv t h u) (static : Static s bh iv u) :
    Static s bh iv (KernelBackwardStep.recompute M s f scale bh iv t h u regs).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    rcases static with ⟨head,key,value,mk,mv⟩
    constructor
    all_goals simp [KernelBackwardStep.recompute,State.setReg,keyPtr,valPtr,betaPtr,
      head,key,value,mk,mv]

#axiomsClean registers
#axiomsClean static
#axiomsClean reverse_static
#axiomsClean recompute_static
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardInitial

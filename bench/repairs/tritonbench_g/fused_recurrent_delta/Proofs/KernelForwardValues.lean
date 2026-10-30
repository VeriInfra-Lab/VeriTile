import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardStep
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardValues
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 18000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def input (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) : KernelReference.ForwardInput s α :=
  let a := KernelReference.forwardInput M s f bh iv u
  ⟨fun n => a.key (t+n),fun n => a.query (t+n),fun n => a.value (t+n),fun n => a.beta (t+n),h⟩
def next (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) : ForwardTile s α :=
  KernelReference.forward M s (input M s f bh iv t h u) 1
def residual (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) : Fin s.FBV → α :=
  KernelReference.residual M s (input M s f bh iv t h u) 0
def output (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) : Fin s.FBV → α :=
  KernelReference.output M s (input M s f bh iv t h u) scale 0

@[simp] theorem eta (ji : TileIndex [a,b]) : (ji.1,ji.2.1,PUnit.unit)=ji := by
  rcases ji with ⟨j,i,⟨⟩⟩
  rfl

theorem carry (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u) :
    (KernelForwardStep.execute M s f scale bh iv t h u regs).val.regs .real [s.FBV,s.BK] "h"=
      some (next M s f bh iv t h u) := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp [KernelForwardStep.execute,next,input,KernelReference.forward,
      KernelReference.forwardInput,KernelReference.keySum,KernelReference.add,KernelReference.mul,
      KernelReference.sub,KernelReference.read,KernelReference.cast,KernelReference.zero,
      keyPtr,valPtr,betaPtr,keyMask,valMask,State.setReg,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals try rfl

#axiomsClean carry

theorem registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u) :
    KernelRegisters.Forward s f bh iv (t+1) (next M s f bh iv t h u)
      (KernelForwardStep.execute M s f scale bh iv t h u regs).val := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,carry M s f scale bh iv t h u regs⟩
  all_goals
    rcases f with ⟨vector,initial,final,ig,tg,default⟩
    cases vector <;> cases regs
    all_goals simp [KernelForwardStep.execute,State.setReg,keyPtr,valPtr,betaPtr]
  all_goals try assumption
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,keyAddress,valAddress,betaAddress,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

theorem pids (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u) :
    (KernelForwardStep.execute M s f scale bh iv t h u regs).val.pids=u.pids := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs <;>
    simp [KernelForwardStep.execute,keyPtr,valPtr,betaPtr]

#axiomsClean registers
#axiomsClean pids
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardValues

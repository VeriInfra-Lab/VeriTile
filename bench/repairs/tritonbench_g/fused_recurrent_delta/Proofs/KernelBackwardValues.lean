import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardStep
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardReference
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardValues
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def enhanced (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.enhance M s scale (KernelBackwardReference.input M s f bh iv u) t h
def next (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.retreat M s (KernelBackwardReference.input M s f bh iv u) t
    (enhanced M s f scale bh iv t h u)
def dk (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.dk M s (KernelBackwardReference.input M s f bh iv u) t
    (enhanced M s f scale bh iv t h u)
def dv (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.dv M s (KernelBackwardReference.input M s f bh iv u) t
    (enhanced M s f scale bh iv t h u)
def dbeta (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.dbeta M s (KernelBackwardReference.input M s f bh iv u) t
    (enhanced M s f scale bh iv t h u)

@[simp] theorem eta (ij : TileIndex [a,b]) : (ij.1,ij.2.1,PUnit.unit)=ij := by
  rcases ij with ⟨i,j,⟨⟩⟩
  rfl

theorem carry (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : KernelBackwardRegisters.Reverse s f bh iv t h u) :
    (KernelBackwardStep.reverse M s f scale bh iv t h u regs).val.regs .real [s.BK,s.BBV] "d_h"=
      some (next M s f scale bh iv t h u) := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp [KernelBackwardStep.reverse,next,enhanced,KernelBackwardReference.retreat,
      KernelBackwardReference.enhance,KernelBackwardReference.rawDv,KernelBackwardReference.dv,
      KernelBackwardReference.input,KernelBackwardReference.keySum,KernelReference.add,KernelReference.mul,
      KernelReference.sub,KernelReference.read,KernelReference.cast,KernelReference.zero,KernelReference.scalar,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,State.setReg,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals try rfl

theorem registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : KernelBackwardRegisters.Reverse s f bh iv (t+1) h u) :
    KernelBackwardRegisters.Reverse s f bh iv t (next M s f scale bh iv (t+1) h u)
      (KernelBackwardStep.reverse M s f scale bh iv (t+1) h u regs).val := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,carry M s f scale bh iv (t+1) h u regs⟩
  all_goals
    rcases f with ⟨vector,initial,final,ig,tg,default⟩
    cases vector <;> cases regs
    all_goals simp [KernelBackwardStep.reverse,State.setReg,keyPtr,valPtr,betaPtr,dbetaPtr]
  all_goals try assumption
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,dbetaPtr,keyAddress,valAddress,betaAddress,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]
  all_goals omega

theorem pids (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : KernelBackwardRegisters.Reverse s f bh iv t h u) :
    (KernelBackwardStep.reverse M s f scale bh iv t h u regs).val.pids=u.pids := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs <;>
    simp [KernelBackwardStep.reverse,keyPtr,valPtr,betaPtr,dbetaPtr]

#axiomsClean carry
#axiomsClean registers
#axiomsClean pids
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardValues

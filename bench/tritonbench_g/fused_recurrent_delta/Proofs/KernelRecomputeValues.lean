import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardStep
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardReference
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelStore
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeValues
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def next (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (h : Tile s α) (u : State α) : Tile s α :=
  let a := KernelBackwardReference.input M s f bh iv u
  fun ij => KernelReference.add M (h ij)
    (KernelReference.mul M (a.key t ij.1) (KernelReference.mul M (a.value t ij.2.1) (a.beta t ij.2.1)))
def dk (M : Algebra α) (s : Shape) (bh iv t : Nat) (h : Tile s α) (u : State α) :=
  KernelBackwardReference.correction M s h
    (fun i => KernelReference.read M u "dk" (keyAddress s (plane s bh iv) t i.val) (decide (i.val<s.K)))
    (fun j => KernelReference.read M u "dv" (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V)))
def dq (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) :=
  KernelBackwardReference.dq M s scale (KernelBackwardReference.input M s f bh iv u) t
    (next M s f bh iv t h u)

theorem scatter_other {ι : Type} (r r' : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (xs : List ι) (u : State α) (offset : Nat) (different : r'≠r) :
    (xs.foldl (fun v i => if active i then v.write r (off i) (value i) else v) u).mem r' offset=u.mem r' offset :=
  State.masked_scatter_frame r off value active xs r' offset (Or.inl different) u

@[simp] theorem eta (ij : TileIndex [a,b]) : (ij.1,ij.2.1,PUnit.unit)=ij := by
  rcases ij with ⟨i,j,⟨⟩⟩
  rfl

theorem carry (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Recompute s f bh iv t h u) :
    (KernelBackwardStep.recompute M s f scale bh iv t h u regs).val.regs .real [s.BK,s.BBV] "h"=
      some (next M s f bh iv t h u) := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp [KernelBackwardStep.recompute,next,KernelBackwardReference.input,
      KernelReference.add,KernelReference.mul,KernelReference.sub,KernelReference.read,
      KernelReference.cast,KernelReference.zero,keyPtr,valPtr,betaPtr,keyMask,valMask,
      State.setReg,bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,scatter_other]
  all_goals try rfl

theorem registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Recompute s f bh iv t h u) :
    Recompute s f bh iv (t+1) (next M s f bh iv t h u)
      (KernelBackwardStep.recompute M s f scale bh iv t h u regs).val := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,carry M s f scale bh iv t h u regs⟩
  all_goals
    rcases f with ⟨vector,initial,final,ig,tg,default⟩
    cases vector <;> cases regs
    all_goals simp [KernelBackwardStep.recompute,State.setReg,keyPtr,valPtr,betaPtr]
  all_goals try assumption
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,keyAddress,valAddress,betaAddress,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

#axiomsClean scatter_other
#axiomsClean carry
#axiomsClean registers
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRecomputeValues

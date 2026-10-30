import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardStateMemory
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardInitial
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRecomputeValues
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardMiddle
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem scatter_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (xs : List ι) (u : State α) :
    (xs.foldl (fun v i => if active i then v.write r (off i) (value i) else v) u).regs=u.regs := by
  induction xs generalizing u with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

theorem scatter_mem {ι : Type} (r : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (xs : List ι) (u v : State α) (same : u.mem=v.mem) :
    (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w) u).mem=
      (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w) v).mem := by
  induction xs generalizing u v with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

def execute (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : Tile s α) (u : State α) (carry : u.regs .real [s.BK,s.BBV] "d_h"=some h)
    (static : KernelBackwardInitial.Static s bh iv u) :
    {v : State α // run (engine M) (second s f scale).before u=some v} := by
  obtain ⟨head,key,value,mk,mv⟩ := static
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> cases ig
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [second,first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,head,key,value,mk,mv,carry]
    try rfl

theorem memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : Tile s α) (u : State α) (carry : u.regs .real [s.BK,s.BBV] "d_h"=some h)
    (static : KernelBackwardInitial.Static s bh iv u) :
    (execute M s f scale bh iv h u carry static).val.mem=
      (if f.initialGrad then KernelBackwardStateMemory.store M s "dh0" bh iv h u else u).mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> cases ig <;> cases static
  all_goals simp only [execute,KernelBackwardStateMemory.store,KernelReference.stored,KernelReference.cast,
    KernelBackwardStateMemory.mask,keyMask,valMask,stateAddress,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    Nat.zero_mul,Nat.zero_add,Nat.add_assoc,Bool.false_eq_true,ite_false,ite_true,State.setReg_mem,Bool.and_eq_true,decide_eq_true_eq]
  all_goals try rfl
  all_goals apply scatter_mem
  all_goals rfl

theorem registers (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : Tile s α) (u : State α) (carry : u.regs .real [s.BK,s.BBV] "d_h"=some h)
    (static : KernelBackwardInitial.Static s bh iv u) :
    Recompute s f bh iv 0 (KernelBackwardReference.input M s f bh iv u).initial
      (execute M s f scale bh iv h u carry static).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> cases ig <;> cases static
  all_goals
    constructor
    all_goals
      simp [execute,KernelBackwardReference.input,KernelReference.add,KernelReference.zero,
        KernelReference.read,KernelReference.cast,keyPtr,valPtr,betaPtr,plane,keyMask,valMask,
        keyAddress,valAddress,betaAddress,stateAddress,State.setReg,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Nat.add_assoc,
        KernelRecomputeValues.scatter_other]
  all_goals try assumption
  all_goals funext i
  all_goals simp [FP.Structural.bop,keyPtr,valPtr,betaPtr,plane,keyMask,valMask,
    keyAddress,valAddress,betaAddress,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]

theorem static (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : Tile s α) (u : State α) (carry : u.regs .real [s.BK,s.BBV] "d_h"=some h)
    (static : KernelBackwardInitial.Static s bh iv u) :
    KernelBackwardInitial.Static s bh iv (execute M s f scale bh iv h u carry static).val := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases initial <;> cases ig
  all_goals
    rcases static with ⟨head,key,value,mk,mv⟩
    constructor
    all_goals simp [execute,State.setReg,head,key,value,mk,mv]

#axiomsClean execute
#axiomsClean memory
#axiomsClean registers
#axiomsClean static
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardMiddle

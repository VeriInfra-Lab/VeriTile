import bench.tritonbench_g.matmul_kernel.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem min_form (a b : Nat) : min a b=if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

structure Context {α : Type} (p : Config) (initial : State α) (k : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  pm : current.regs .nat [] "pid_m"=some (fun _ => pidM p initial)
  pn : current.regs .nat [] "pid_n"=some (fun _ => pidN p initial)
  offk : current.regs .nat [BK p] "offs_k"=some (fun i => i.1.val)
  ap : current.regs .ptr [BM p,BK p] "a_ptrs"=some (fun i => (("A" : RegionName),aOffset p initial k i))
  bp : current.regs .ptr [BK p,BN p] "b_ptrs"=some (fun i => (("B" : RegionName),bOffset p initial k i))

def HasAcc {α : Type} (p : Config) (a : CIndex p → α) (s : State α) : Prop :=
  s.regs .real [BM p,BN p] "accumulator"=some a

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (s : State α) :
    {t : State α // FP.Structural.run (backend M D) (prelude p) s=some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude,source,typed,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,VeriTile.Triton.Region.cast,Option.bind,integralFloorDiv,integralMod]
  rfl

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (s : State α) :
    Context p s 0 (preludeExecution M D p s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p s).val := by
  constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,pidM,pidN,row,col,aRow,bCol,aOffset,bOffset,HasAcc,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    {t : State α // FP.Structural.run (backend M D) (body p)
      (current.setReg "k" .nat [] (fun _ => k))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [body,source,typed,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,VeriTile.Triton.Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.offk,ctx.ap,ctx.bp,ha,backend]
  rfl

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p initial current k a ctx ha).val
    Context p initial (k+1) t ∧ HasAcc p (update M D p initial k a) t := by
  dsimp only
  all_goals constructor
  all_goals try constructor
  all_goals simp [HasAcc,bodyExecution,ctx.memory,ctx.pm,ctx.pn,ctx.offk]
  · funext i
    simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,aOffset,
      Nat.add_mul,Nat.one_mul,Nat.mul_one,Nat.add_assoc]
  · funext i
    simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,bOffset,
      Nat.add_mul,Nat.one_mul,Nat.mul_one,Nat.add_assoc]
  · norm_num only [update]
    congr 1 <;> funext i <;> norm_num [aValue,bValue,read,BK]

#axiomsClean preludeExecution
#axiomsClean preludeContext
#axiomsClean bodyExecution
#axiomsClean bodyContext
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource

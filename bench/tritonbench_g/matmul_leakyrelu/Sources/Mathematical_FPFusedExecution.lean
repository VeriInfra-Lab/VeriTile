import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_FPFusedReference
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem min_form (a b : Nat) : min a b=if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

structure Context {α : Type} (p : Parameters) (initial : State α) (k : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  pm : current.regs .nat [] "pid_m"=some (fun _ => pidM p initial)
  pn : current.regs .nat [] "pid_n"=some (fun _ => pidN p initial)
  offk : current.regs .nat [BK p] "offs_k"=some (fun i => i.1.val)
  ap : current.regs .ptr [BM p,BK p] "a_ptrs"=some (fun i => (p.A,aOffset p initial k i))
  bp : current.regs .ptr [BK p,BN p] "b_ptrs"=some (fun i => (p.B,bOffset p initial k i))

def HasAcc {α : Type} (p : Parameters) (a : CIndex p → α) (s : State α) : Prop :=
  s.regs .real [BM p,BN p] "accumulator"=some a

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    {t : State α // run (backend M D) (prelude p st) s=some t} := by
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [prelude,source,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
     rfl)

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    Context p s 0 (preludeExecution M D p st s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p st s).val := by
  cases st <;> constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,blocksM,blocksN,perGroup,firstM,
    groupSize,pidM,pidN,row,col,aRow,bCol,aOffset,bOffset,min_form,HasAcc,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p st)
      (current.setReg "k" .nat [] (fun _ => k))=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [body,source,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.offk,ctx.ap,ctx.bp,ha,backend]
     rfl)

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p st initial current k a ctx ha).val
    Context p initial (k+1) t ∧ HasAcc p (update M D p st initial k a) t := by
  cases st <;> dsimp only
  all_goals constructor
  all_goals try constructor
  all_goals simp [HasAcc,bodyExecution,ctx.memory,ctx.pm,ctx.pn,ctx.offk,
    aOffset,bOffset,Nat.add_mul,Nat.add_assoc,update,aValue,bValue,read,dtype,backend,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl
  all_goals funext i
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,aOffset,bOffset,
    Nat.add_mul,Nat.add_assoc]
  all_goals unfold aValue bValue
  all_goals simp [aOffset,bOffset,read,dtype,Nat.add_assoc]

#axiomsClean preludeExecution
#axiomsClean preludeContext
#axiomsClean bodyExecution
#axiomsClean bodyContext
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused

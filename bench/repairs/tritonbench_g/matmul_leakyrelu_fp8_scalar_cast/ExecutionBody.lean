import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.ExecutionContext

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def safeK (p : Problem) (k lane : Nat) : Int :=
  if (k : Int)*64+(lane : Int) < (p.K : Int) then (k : Int)*64+(lane : Int) else 0

def aOffset {α : Type} (p : Problem) (b : Buffers) (s : State α) (k : Nat) (i : AIndex) : Nat :=
  ((qMod (row p s i.1) (p.M : Int))*(b.A.rowStride : Int) +
    safeK p k i.2.1.val*(b.A.columnStride : Int)).toNat

def bOffset {α : Type} (p : Problem) (b : Buffers) (s : State α) (k : Nat) (i : BIndex) : Nat :=
  (safeK p k i.1.val*(b.B.rowStride : Int) +
    (qMod (col p s i.2.1) (p.N : Int))*(b.B.columnStride : Int)).toNat

def aValue {α : Type} [Inhabited α] (M : Algebra α) (p : Problem) (b : Buffers)
    (s : State α) (k : Nat) (i : AIndex) : α :=
  M.fp32Load (if (k : Int)*64+(i.2.1.val : Int) < (p.K : Int) then
    (s.mem "A" (aOffset p b s k i)).read .real
  else M.literal (some .fp32) .real 0.0)

def bValue {α : Type} [Inhabited α] (M : Algebra α) (p : Problem) (b : Buffers)
    (s : State α) (k : Nat) (i : BIndex) : α :=
  M.fp32Load (if (k : Int)*64+(i.1.val : Int) < (p.K : Int) then
    (s.mem "B" (bOffset p b s k i)).read .real
  else M.literal (some .fp32) .real 0.0)

def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) (k : Nat) (acc : CIndex → α) : CIndex → α :=
  D.apply (some .fp32) .real .real .real {} (batch := [])
    (aValue M p b s k) (bValue M p b s k) acc

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial current : State α) (k : Nat) (a : CIndex → α)
    (ctx : Context p initial current) (ha : HasAcc a current) :
    {t : State α // run (backend M D) (body p b)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [body,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    ctx.memory,ctx.lk,ctx.am,ctx.bn,ha,backend]
  rfl

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial current : State α) (k : Nat) (a : CIndex → α)
    (ctx : Context p initial current) (ha : HasAcc a current) :
    let t := (bodyExecution M D p b initial current k a ctx ha).val
    Context p initial t ∧ HasAcc (update M D p b initial k a) t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.pm,ctx.pn,ctx.lm,ctx.ln,ctx.lk,
      ctx.am,ctx.bn,ctx.kt,backend]
  · simp [HasAcc,bodyExecution,ctx.memory,update,aValue,bValue,aOffset,bOffset,safeK,ite_mul,backend,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    congr 1 <;> funext j <;> simp [aValue,bValue,aOffset,bOffset,safeK,ite_mul]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution

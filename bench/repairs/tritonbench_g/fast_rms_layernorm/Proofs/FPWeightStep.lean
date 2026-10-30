import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightProgram

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPWeightProgram
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def columns (K pid : Nat) : Values α .nat [K] := fun i => pid*K+i.1.val
def masks (h : Call) (K pid : Nat) : Values α .bool [K] :=
  fun i => decide (pid*K+i.1.val<h.cols)
def raw (M : Algebra α) (h : Call) (K pid : Nat) (u : State α)
    (role : RegionName) (row : Nat) (i : TileIndex [K]) : α :=
  if pid*K+i.1.val<h.cols then (u.mem role (row*h.cols+(pid*K+i.1.val))).read .real else FPReference.literal M 0
def saved (u : State α) (row : Nat) : α := (u.mem "r" row).read .real

def initial (M : Algebra α) (h : Call) (K : Nat) (u : State α) : State α :=
  ((u.setReg "col" .nat [K] (columns K (u.pids 0))).setReg "mask" .bool [K] (masks h K (u.pids 0))).setReg
    "acc" .real [K] (fun _ => FPReference.literal M 0)

theorem prefix_executes (M : Algebra α) (h : Call) (K : Nat) (u : State α) :
    run (engine M) (prefixCode h K) u=some (initial M h K u) := by
  simp [prefixCode,Kernels.weightBackward,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
    columns,masks,initial,FPReference.literal]
  rfl

def nextAcc (M : Algebra α) (h : Call) (K pid row : Nat) (u : State α) (acc : Values α .real [K]) :=
  fun i => FPReference.add M (acc i) (FPWeightReference.term M h.gemma
    (raw M h K pid u "x" row i) (raw M h K pid u "dx" row i) (saved u row))

theorem iteration (M : Algebra α) (h : Call) (K pid row : Nat) (u : State α)
    (acc : Values α .real [K])
    (hcol : u.regs .nat [K] "col"=some (columns K pid))
    (hmask : u.regs .bool [K] "mask"=some (masks h K pid))
    (hacc : u.regs .real [K] "acc"=some acc) :
    ∃ v, run (engine M) (loopBody h K) (u.setReg "row" .nat [] (fun _ => row))=some v ∧
      v.mem=u.mem ∧ v.pids=u.pids ∧
      v.regs .nat [K] "col"=some (columns K pid) ∧
      v.regs .bool [K] "mask"=some (masks h K pid) ∧
      v.regs .real [K] "acc"=some (nextAcc M h K pid row u acc) := by
  cases mode : h.gemma
  all_goals
    simp [loopBody,Kernels.weightBackward,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,hcol,hmask,hacc,
      nextAcc,columns,masks,raw,saved,FPWeightReference.term,FPWeightReference.normed,
      FPReference.literal,FPReference.cast,FPReference.add,FPReference.mul,mode]
    <;> funext i
    <;> simp [nextAcc,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ofFloat,toFloat,raw,saved,FPWeightReference.term,FPWeightReference.normed,
      FPReference.literal,FPReference.cast,FPReference.add,FPReference.mul,mode]

#axiomsClean prefix_executes
#axiomsClean iteration
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightStep

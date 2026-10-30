import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelReference
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelRows
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open KernelMemory
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def raw (u : State α) (r : RegionName) (stride B : Nat) : Fin B → α :=
  fun j => (u.mem r (u.pids 0*stride+j.val)).read .real

def forwardHeap (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool) (u : State α) :=
  commit "y" (fun i : TileIndex [B] => u.pids 0*N+i.1.val)
    (fun i => KernelReference.output M N eps bias (raw u "x" N B) (raw u "w" 0 B) (raw u "b" 0 B) i.1)
    (fun i => i.1.val<N) (TileShape.allIndices [B])
    (update "mu" (u.pids 0) (KernelReference.savedMean M N (raw u "x" N B))
      (update "r" (u.pids 0) (KernelReference.savedInverse M N eps (raw u "x" N B)) u.mem))

theorem forward_memory (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=forwardHeap M N B eps bias u := by
  cases bias
  all_goals
    simp [forwardKernel,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,FP.Structural.store,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      forwardHeap,raw,KernelReference.output,KernelReference.savedMean,KernelReference.savedInverse,
      KernelReference.centered,KernelReference.mean,KernelReference.inverse,KernelReference.reduce,
      KernelReference.input,KernelReference.literal,KernelReference.cast,KernelReference.add,
      KernelReference.sub,KernelReference.mul,KernelReference.div,fold_mem,
      ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex]
    <;> dsimp only [State.write,update]
    <;> norm_num
    <;> rfl

def backwardHeap (M : Algebra α) (N B : Nat) (u : State α) :=
  commit "dx" (fun i : TileIndex [B] => u.pids 0*N+i.1.val)
    (fun i => KernelReference.gradient M N (raw u "x" N B) (raw u "dy" N B) (raw u "w" 0 B)
      ((u.mem "mu" (u.pids 0)).read .real) ((u.mem "r" (u.pids 0)).read .real) i.1)
    (fun i => i.1.val<N) (TileShape.allIndices [B]) u.mem

theorem backward_memory (M : Algebra α) (N B : Nat) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u=some v ∧
      v.mem=backwardHeap M N B u := by
  simp [backwardKernel,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,FP.Structural.store,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    backwardHeap,raw,KernelReference.gradient,KernelReference.normalized,KernelReference.weighted,
    KernelReference.product,KernelReference.reduce,KernelReference.input,KernelReference.literal,
    KernelReference.cast,KernelReference.add,KernelReference.sub,KernelReference.mul,KernelReference.div,
    fold_mem,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex]
  <;> (try dsimp only [State.write,update])
  <;> norm_num
  <;> rfl

#axiomsClean forward_memory
#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelRows

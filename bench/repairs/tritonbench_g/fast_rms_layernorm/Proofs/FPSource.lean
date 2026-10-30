import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPReference
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Complete row execution uses total opaque FP primitive callbacks. No success
or source-result oracle appears in the premise. Output formulas and frames are
proved separately before this is composed into a public host theorem. -/
theorem forward_executes (M : Algebra α) (h : Call) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.forward h) u=some v := by
  unfold Kernels.forward
  split
  all_goals
    simp [FP.Structural.exec,Kernels.regularSource,Kernels.gemmaSource,Original.forward,Original.gemma,
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource.forward,
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource.gemma,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex]

def update (r : RegionName) (o : Nat) (v : Cell α) (heap : RegionName → Nat → Cell α) :=
  fun q n => if q=r ∧ n=o then v else heap q n

def commit {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (heap : RegionName → Nat → Cell α) :=
  indices.foldl (fun mem i => if active i then update r (off i) (val i) mem else mem) heap

theorem fold_mem {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (u : State α) :
    (indices.foldl (fun s i => if active i then s.write r (off i) (val i) else s) u).mem=
      commit r off val active indices u.mem := by
  induction indices generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,hi,ite_true,ite_false]
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl

def forwardHeap (M : Algebra α) (h : Call) (u : State α) : RegionName → Nat → Cell α :=
  let x:=fun j : Fin h.block => (u.mem "x" (u.pids 0*h.cols+j.val)).read .real
  let w:=fun j : Fin h.block => (u.mem "w" j.val).read .real
  commit "y" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
    (fun i => FPReference.output M h x w i.1) (fun i => i.1.val<h.cols) (TileShape.allIndices [h.block])
    (update "r" (u.pids 0) (FPReference.savedInv M h x) u.mem)

theorem forward_memory (M : Algebra α) (h : Call) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.forward h) u=some v ∧ v.mem=forwardHeap M h u := by
  unfold Kernels.forward
  split <;> rename_i branch
  all_goals
    simp [branch,ofFloat,toFloat,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,FP.Structural.exec,Kernels.regularSource,Kernels.gemmaSource,Original.forward,Original.gemma,
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource.forward,
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource.gemma,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      forwardHeap,FPReference.output,FPReference.savedInv,FPReference.normalized,FPReference.inv,
      FPReference.reduce,FPReference.input,FPReference.padded,FPReference.literal,FPReference.count,FPReference.cast,
      FPReference.add,FPReference.mul,FPReference.div,fold_mem]
    <;> dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write,update]
    <;> norm_num
    <;> rfl

#axiomsClean forward_memory

#axiomsClean forward_executes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPSource

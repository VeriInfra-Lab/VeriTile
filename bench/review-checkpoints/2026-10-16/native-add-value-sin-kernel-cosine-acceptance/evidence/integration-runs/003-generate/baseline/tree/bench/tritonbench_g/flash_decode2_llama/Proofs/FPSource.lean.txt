import VeriTile.Triton
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.StructuralIO
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages

/-! Floating-value versions of the three already accepted stage cuts.
`Acc`, `SumExp`, previous maximum and `Final` remain supplied stage inputs.
The complete online-softmax loop is not asserted by these stage contracts.

The two program coordinates, all original strides, dimension and aliasing
are retained. The fp32 compute tags survive the `.real` carrier erasure used
by `ComputeKernel.mk`; that carrier contains opaque floating values here.

The scalar maximum uses the original `tl.maximum` expression. The frontend
emits `maximumFloat` with the fp32 context inferred from the load types; no
same-format cast or task-local AST replacement is introduced. Its independent
maximum callback keeps NaN and signed-zero behavior explicit. The supplied
stage inputs and erased `.real` storage binding remain as before.

-/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Row where
  batchStride : Nat
  headStride : Nat
  laneStride : Nat

/-- Scalar stage inputs do not have a lane stride. -/
structure Scalar where
  batchStride : Nat
  headStride : Nat

def address {α : Type} (s : State α) (r : Row) {D : Nat}
    (i : TileIndex [D]) : Nat :=
  s.pids 0 * r.batchStride + s.pids 1 * r.headStride + i.1.val * r.laneStride

def scalarAddress {α : Type} (s : State α) (r : Scalar) : Nat :=
  s.pids 0 * r.batchStride + s.pids 1 * r.headStride

def normalization (Acc SumExp O : RegionName) (a : Row) (z : Scalar)
    (o : Row) (D : Nat) : ComputeKernel :=
  ComputeKernel.mk [Acc, SumExp] [O] (triton {
    tl.region Acc = tl.float32, SumExp = tl.float32, O = tl.float32
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, $(D))
    acc = tl.load(Acc + cur_batch * $(a.batchStride) +
      cur_head * $(a.headStride) + offs_d * $(a.laneStride))
    sum_exp = tl.load(SumExp + cur_batch * $(z.batchStride) + cur_head * $(z.headStride))
    tl.store(O + cur_batch * $(o.batchStride) + cur_head * $(o.headStride) +
      offs_d * $(o.laneStride), acc / sum_exp)
  }).surfaceBody

def copy (Final O : RegionName) (a o : Row) (D : Nat) : ComputeKernel :=
  ComputeKernel.mk [Final] [O] (triton {
    tl.region Final = tl.float32, O = tl.float32
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, $(D))
    final = tl.load(Final + cur_batch * $(a.batchStride) +
      cur_head * $(a.headStride) + offs_d * $(a.laneStride))
    tl.store(O + cur_batch * $(o.batchStride) + cur_head * $(o.headStride) +
      offs_d * $(o.laneStride), final)
  }).surfaceBody

/-- Direct maximum primitive at fp32 compute precision; no comparison rewrite. -/
def maximum (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) : ComputeKernel :=
  ComputeKernel.mk [LogExp, Prev] [Next] (triton {
    tl.region LogExp = tl.float32, Prev = tl.float32, Next = tl.float32
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    tlogic = tl.load(LogExp + cur_batch * $(l.batchStride) + cur_head * $(l.headStride) + $(block))
    max_logic = tl.load(Prev + cur_batch * $(m.batchStride) + cur_head * $(m.headStride))
    new_max_logic = tl.maximum(tlogic, max_logic)
    tl.store(Next + cur_batch * $(m.batchStride) + cur_head * $(m.headStride), new_max_logic)
  }).surfaceBody

def normalizationValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Acc SumExp : RegionName) (a : Row) (z : Scalar) (D : Nat) : TileIndex [D] → α :=
  fun i => M.binary (some .fp32) .real .div
    (M.fp32Load ((s.mem Acc (address s a i)).read .real))
    (M.fp32Load ((s.mem SumExp (scalarAddress s z)).read .real))

def copyValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Final : RegionName) (a : Row) (D : Nat) : TileIndex [D] → α :=
  fun i => M.fp32Load ((s.mem Final (address s a i)).read .real)

def maximumValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (LogExp Prev : RegionName) (block : Nat) (l m : Scalar) : α :=
  M.binary (some .fp32) .real .max
    (M.fp32Load ((s.mem LogExp (scalarAddress s l + block)).read .real))
    (M.fp32Load ((s.mem Prev (scalarAddress s m)).read .real))

theorem normalization_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Acc SumExp O : RegionName) (a : Row) (z : Scalar) (o : Row) (D : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (normalization Acc SumExp O a z o D) s = some t ∧
      t.mem = (scatter s O D (address s o) (fun _ => True)
        (normalizationValues M s Acc SumExp a z D)).mem := by
  simp [normalization, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, normalizationValues, address,
    scalarAddress, scatter]
  apply scatter_memory_congr (active := fun _ => True)
  rfl

theorem copy_memory {α : Type} [Inhabited α] (M : Algebra α)
    (Final O : RegionName) (a o : Row) (D : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (copy Final O a o D) s = some t ∧
      t.mem = (scatter s O D (address s o) (fun _ => True) (copyValues M s Final a D)).mem := by
  simp [copy, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, copyValues, address, scatter]
  apply scatter_memory_congr (active := fun _ => True)
  rfl

theorem maximum_memory {α : Type} [Inhabited α] (M : Algebra α)
    (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) (s : State α) :
    ∃ t, FP.Structural.exec M (maximum LogExp Prev Next block l m) s = some t ∧
      t.mem = (s.write Next (scalarAddress s m)
        (.mk .real (maximumValue M s LogExp Prev block l m))).mem := by
  simp [maximum, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, Region.cast, maximumValue, scalarAddress, State.write, State.setReg,
    ofFloat, toFloat]

#axiomsClean normalization_memory
#axiomsClean copy_memory
#axiomsClean maximum_memory
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages

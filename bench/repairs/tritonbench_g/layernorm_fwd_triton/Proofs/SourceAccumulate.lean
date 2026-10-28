import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceModel
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SignedRange
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure MeanReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (a : TileIndex [p.block] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  epsilon : t.regs .real [] "eps" = s.regs .real [] "eps"
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  nValue : t.regs .int [] "N" = some (fun _ => (p.columns : Int))
  zeroValue : t.regs .int [] "zero" = some (fun _ => 0)
  stepValue : t.regs .int [] "step" = some (fun _ => (p.block : Int))
  lanes : t.regs .int [p.block] "lanes" = some (fun i => (i.1.val : Int))
  xStride : t.regs .int [] "stride_x_hd" = some (fun _ => (p.xDimStride : Int))
  yStride : t.regs .int [] "stride_y_hd" = some (fun _ => (p.yDimStride : Int))
  wStride : t.regs .int [] "stride_w_hd" = some (fun _ => (p.weightDimStride : Int))
  accumulator : t.regs .real [p.block] "_mean" = some a

structure VarReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (a : TileIndex [p.block] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  epsilon : t.regs .real [] "eps" = s.regs .real [] "eps"
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  nValue : t.regs .int [] "N" = some (fun _ => (p.columns : Int))
  zeroValue : t.regs .int [] "zero" = some (fun _ => 0)
  stepValue : t.regs .int [] "step" = some (fun _ => (p.block : Int))
  lanes : t.regs .int [p.block] "lanes" = some (fun i => (i.1.val : Int))
  xStride : t.regs .int [] "stride_x_hd" = some (fun _ => (p.xDimStride : Int))
  yStride : t.regs .int [] "stride_y_hd" = some (fun _ => (p.yDimStride : Int))
  wStride : t.regs .int [] "stride_w_hd" = some (fun _ => (p.weightDimStride : Int))
  accumulator : t.regs .real [p.block] "_var" = some a
  mean : t.regs .real [] "mean" = some (fun _ => meanValue M p s)

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, run M (prelude p) s = some t ∧ MeanReady M p s t (initial M p) := by
  simp [prelude, kernel, runtimeSource, ComputeKernel.surfaceBody, Random.i32ArangeSize, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, Region.cast]
  constructor <;> simp [State.setReg, initial, xBase, yBase, wBase, bop, numeric]
  all_goals try rfl
  all_goals funext i
  all_goals simp [bop, Broadcast.leftIndex, Broadcast.rightIndex, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add]

theorem mean_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (a : TileIndex [p.block] → α) (h : MeanReady M p s t a) :
    ∃ u, run M (loopBody p 18) (t.setReg "off" .int [] (fun _ => (off : Int))) = some u ∧
      MeanReady M p s u (meanStep M p s off a) := by
  simp [loopBody, kernel, runtimeSource, ComputeKernel.surfaceBody, Random.i32ArangeSize, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, Int.toNat_add, Int.toNat_mul, Int.natCast_nonneg, Int.toNat_of_nonpos, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w, h.accumulator, meanStep, xValue, centered,
    bop, numeric, ofFloat, toFloat]
  all_goals try rfl
  all_goals funext i
  all_goals simp [meanStep, varStep, xValue, centered, bop, numeric,
    Broadcast.leftIndex, Broadcast.rightIndex, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add]

theorem mean_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (a : TileIndex [p.block] → α)
    (h : MeanReady M p s t a) :
    ∃ u, FP.Structural.rangeInt M "off" (start : Int) (stop : Int) p.block (loopBody p 18) t = some u ∧
      MeanReady M p s u (foldRange (meanStep M p s) start stop p.block a) := by
  apply SignedRange.values M "off" (loopBody p 18) (meanStep M p s)
    (fun _ a t => MeanReady M p s t a) (fun a t => MeanReady M p s t a)
    start stop p.block a t
  · intro n a u hu _
    exact mean_body_run M p s u n a hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem var_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (a : TileIndex [p.block] → α) (h : VarReady M p s t a) :
    ∃ u, run M (loopBody p 21) (t.setReg "off" .int [] (fun _ => (off : Int))) = some u ∧
      VarReady M p s u (varStep M p s off a) := by
  simp [loopBody, kernel, runtimeSource, ComputeKernel.surfaceBody, Random.i32ArangeSize, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, Int.toNat_add, Int.toNat_mul, Int.natCast_nonneg, Int.toNat_of_nonpos, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w, h.accumulator, h.mean]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w, h.accumulator, h.mean, varStep, xValue, centered,
    bop, numeric, ofFloat, toFloat]
  all_goals try rfl
  all_goals funext i
  all_goals simp [meanStep, varStep, xValue, centered, bop, numeric,
    Broadcast.leftIndex, Broadcast.rightIndex, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add]

theorem var_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (a : TileIndex [p.block] → α)
    (h : VarReady M p s t a) :
    ∃ u, FP.Structural.rangeInt M "off" (start : Int) (stop : Int) p.block (loopBody p 21) t = some u ∧
      VarReady M p s u (foldRange (varStep M p s) start stop p.block a) := by
  apply SignedRange.values M "off" (loopBody p 21) (varStep M p s)
    (fun _ a t => VarReady M p s t a) (fun a t => VarReady M p s t a)
    start stop p.block a t
  · intro n a u hu _
    exact var_body_run M p s u n a hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem mean_post_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α)
    (h : MeanReady M p s t (foldRange (meanStep M p s) 0 p.columns p.block (initial M p))) :
    ∃ u, run M (meanPost p) t = some u ∧ VarReady M p s u (initial M p) := by
  simp [meanPost, kernel, runtimeSource, ComputeKernel.surfaceBody, Random.i32ArangeSize, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, Region.cast, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.epsilon, h.x, h.nValue, h.zeroValue, h.stepValue, h.lanes, h.xStride, h.yStride, h.wStride, h.y, h.w,
    meanValue, initial, bop, numeric]
  all_goals try rfl

#axiomsClean prelude_run
#axiomsClean mean_body_run
#axiomsClean mean_range_run
#axiomsClean var_body_run
#axiomsClean var_range_run
#axiomsClean mean_post_run
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source

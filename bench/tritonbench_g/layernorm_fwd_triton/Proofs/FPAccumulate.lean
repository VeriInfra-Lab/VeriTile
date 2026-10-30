import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPModel
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
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
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  accumulator : t.regs .real [p.block] "_mean" = some a

structure VarReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (a : TileIndex [p.block] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  accumulator : t.regs .real [p.block] "_var" = some a
  mean : t.regs .real [] "mean" = some (fun _ => meanValue M p s)

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, run M (prelude p) s = some t ∧ MeanReady M p s t (initial M p) := by
  simp [prelude, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, Region.cast]
  constructor <;> simp [State.setReg, initial, xBase, yBase, wBase, bop, numeric]
  all_goals try rfl
  all_goals funext i
  all_goals simp [bop, Broadcast.leftIndex, Broadcast.rightIndex]

theorem mean_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (a : TileIndex [p.block] → α) (h : MeanReady M p s t a) :
    ∃ u, run M (loopBody p 6) (t.setReg "off" .nat [] (fun _ => off)) = some u ∧
      MeanReady M p s u (meanStep M p s off a) := by
  simp [loopBody, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator, meanStep, xValue, centered,
    bop, numeric, ofFloat, toFloat]
  all_goals try rfl

theorem mean_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (a : TileIndex [p.block] → α)
    (h : MeanReady M p s t a) :
    ∃ u, FP.Structural.range M "off" start stop p.block (loopBody p 6) t = some u ∧
      MeanReady M p s u (foldRange (meanStep M p s) start stop p.block a) := by
  apply range_values M "off" (loopBody p 6) (meanStep M p s)
    (fun _ a t => MeanReady M p s t a) (fun a t => MeanReady M p s t a)
    start stop p.block a t
  · intro n a u hu _
    exact mean_body_run M p s u n a hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem var_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (a : TileIndex [p.block] → α) (h : VarReady M p s t a) :
    ∃ u, run M (loopBody p 9) (t.setReg "off" .nat [] (fun _ => off)) = some u ∧
      VarReady M p s u (varStep M p s off a) := by
  simp [loopBody, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator, h.mean]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator, h.mean, varStep, xValue, centered,
    bop, numeric, ofFloat, toFloat]
  all_goals try rfl

theorem var_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (a : TileIndex [p.block] → α)
    (h : VarReady M p s t a) :
    ∃ u, FP.Structural.range M "off" start stop p.block (loopBody p 9) t = some u ∧
      VarReady M p s u (foldRange (varStep M p s) start stop p.block a) := by
  apply range_values M "off" (loopBody p 9) (varStep M p s)
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
  simp [meanPost, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, Region.cast, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.x, h.y, h.w,
    meanValue, initial, bop, numeric]
  all_goals try rfl

#axiomsClean prelude_run
#axiomsClean mean_body_run
#axiomsClean mean_range_run
#axiomsClean var_body_run
#axiomsClean var_range_run
#axiomsClean mean_post_run
end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages

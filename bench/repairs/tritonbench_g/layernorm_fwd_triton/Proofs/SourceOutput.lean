import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceAccumulate
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

@[simp] theorem fold_writes_undef {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : ι → RegionName) (o : ι → Nat)
    (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (r i) (o i) (v i) else t) s).undef = s.undef := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

def scatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (off : Nat) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i => if off+i.1.val < p.columns then
    t.write p.y (yBase p s+(off+i.1.val)*p.yDimStride) (.mk .fp32 (yValue M p s (off+i.1.val))) else t) t

theorem scatter_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) : (scatter M p s t off).mem = outputStep M p s off t.mem := by
  exact fold_writes_mem _ _ _ _ _ _

theorem outputStep_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (mem : RegionName → Nat → Cell α) (off : Nat) (r : RegionName) (o : Nat)
    (h : Outside p s r o) : outputStep M p s off mem r o = mem r o := by
  have hm := scatter_memory M p s {s with mem := mem} off
  rw [← hm]
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h (off+i.1.val) hi)

structure OutputReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (mem : RegionName → Nat → Cell α) : Prop where
  memory : t.mem = mem
  readX : t.mem p.x = s.mem p.x
  readW : t.mem p.weight = s.mem p.weight
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  mean : t.regs .real [] "mean" = some (fun _ => meanValue M p s)
  rstd : t.regs .real [] "rstd" = some (fun _ => rstdValue M p s)

theorem var_post_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α)
    (epsilon : α) (heps : s.regs .real [] "eps_arg"=some (fun _ => epsilon))
    (h : VarReady M p s t (foldRange (varStep M p s) 0 p.columns p.block (initial M p))) :
    ∃ u, run M (varPost p) t = some u ∧ OutputReady M p s u s.mem := by
  simp [varPost, kernel, runtimeSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, Region.cast, h.epsilon, heps, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.x, h.y, h.w, h.mean,
    varianceValue, rstdValue, epsilonValue, heps, bop, numeric]
  all_goals try rfl

theorem output_body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (off : Nat) (mem : RegionName → Nat → Cell α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) (h : OutputReady M p s t mem) :
    ∃ u, run M (loopBody p 12) (t.setReg "off" .nat [] (fun _ => off)) = some u ∧
      OutputReady M p s u (outputStep M p s off mem) := by
  simp [loopBody, kernel, runtimeSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast,
    h.readX, h.readW, h.pids, h.undef, h.x, h.y, h.w, h.mean, h.rstd]
  constructor
  · simp only [fold_writes_mem, State.setReg_mem, h.memory]
    rfl
  · funext o
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hx)]
    simp [h.readX]
  · funext o
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hw)]
    simp [h.readW]
  · simp [h.pids]
  · simp only [fold_writes_undef, State.setReg_undef, h.undef]
  · simp [fold_writes_regs, h.x]
  · simp [fold_writes_regs, h.y]
  · simp [fold_writes_regs, h.w]
  · simp [fold_writes_regs, h.mean]
  · simp [fold_writes_regs, h.rstd]

#axiomsClean scatter_memory
#axiomsClean outputStep_frame
#axiomsClean var_post_run
#axiomsClean output_body_run
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source

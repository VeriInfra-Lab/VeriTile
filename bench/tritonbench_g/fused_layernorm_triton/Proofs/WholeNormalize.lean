import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeWelford

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- One second-loop iteration. The unit row axis is flattened by pure
register remaps, as in the retained normalization proof. No arithmetic or
load is introduced by these remaps. The final cast expands the original
BF16 pointer store conversion. -/
def normalizeChunk (X W Bias Y : RegionName) (D B k : Nat) : List ComputeStmt :=
  [ ComputeStmt.assign .nat [] "xrow" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .nat [1,1] "x0"))),
    ComputeStmt.assign .real [] "rowMean" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .real [1,1] "tmp3"))),
    ComputeStmt.assign .real [] "rowInverse" (.alg (.remap (shape := [1,1]) []
      (fun _ => ⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩) (.ref .real [1,1] "tmp10"))) ] ++
  ((triton {
    tl.region X = tl.float32, W = tl.float32, Bias = tl.float32, Y = tl.bfloat16
    xrow = tl.program_id(0)
    rowMean = 0.0
    rowInverse = 0.0
    roffset = $(k * B)
    rindex = roffset + tl.arange(0, $(B))
    rmask = rindex < $(D)
    tmp11 = tl.load(X + (rindex + ($(D) * xrow)), mask=rmask).to(tl.float32)
    tmp15 = tl.load(W + rindex, mask=rmask).to(tl.float32)
    tmp18 = tl.load(Bias + rindex, mask=rmask).to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - rowMean
    tmp14 = tmp13 * rowInverse
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    stored = (tmp21).to(tl.bfloat16)
    tl.store(Y + (rindex + ($(D) * xrow)), stored, mask=rmask)
  } : ComputeKernel).surfaceBody).drop 3

def normalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 ((s.mem X (col + D * row)).read .real))
  let w := cast32 (cast32 ((s.mem W col).read .real))
  let b := cast32 (cast32 ((s.mem Bias col).read .real))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def maskedNormalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 (if col < D then (s.mem X (col + D * row)).read .real else s.undef .real X (col + D * row)))
  let w := cast32 (cast32 (if col < D then (s.mem W col).read .real else s.undef .real W col))
  let b := cast32 (cast32 (if col < D then (s.mem Bias col).read .real else s.undef .real Bias col))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def outputChunk {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias Y : RegionName) (D B k row : Nat) (mean inverse : α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if k * B + i.1.val < D then
      t.write Y (k * B + i.1.val + D * row)
        (.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k * B + i.1.val) mean inverse))
    else t) s

structure NormalizeRegisters {α : Type} (s t : State α) (row : Nat)
    (mean inverse : α) : Prop where
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  mean : t.regs .real [1,1] "tmp3" = some (fun _ => mean)
  rstd : t.regs .real [1,1] "tmp10" = some (fun _ => inverse)
  row : t.regs .nat [1,1] "x0" = some (fun _ => row)

theorem typed_scatter_memory_congr {α ι : Type} (l : List ι) (s t : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) (h : s.mem = t.mem) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split_ifs with hi
    · exact ih _ _ (by simp only [State.write, h])
    · exact ih s t h

theorem typed_scatter_regs {α ι : Type} (l : List ι) (s : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).regs = s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons]; split_ifs <;> simp [ih]

theorem typed_scatter_undef {α ι : Type} (l : List ι) (s : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).undef = s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons]; split_ifs <;> simp [ih]

theorem normalize_chunk_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (D B k row : Nat) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeChunk X W Bias Y D B k) s = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse := by
  simp [normalizeChunk, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
    numericLt, withEquality, Region.cast, ofFloat, toFloat, hm, hr, hx, store,
    maskedNormalizeValue, outputChunk]
  constructor
  · apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor <;> simp [typed_scatter_regs, typed_scatter_undef, hm, hr, hx]

#axiomsClean normalize_chunk_run
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole

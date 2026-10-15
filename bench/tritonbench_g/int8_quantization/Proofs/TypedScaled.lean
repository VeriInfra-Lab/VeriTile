import bench.tritonbench_g.int8_quantization.Int8Quantization
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Parameters where
  x : RegionName
  suppliedScale : RegionName
  out : RegionName
  scaleOut : RegionName
  rows : Nat
  columns : Nat
  blockRows : Nat
  scaleStride : Nat
  preScale : ℝ

/-- The accepted supplied-scale, pre-rounding stage. Its two outputs are fp32;
no final bias/cast or scale producer is asserted by this stage contract. -/
def source (X ScalePre XInt8 Scale : RegionName) (p : Parameters) : ComputeKernel := triton {
  tl.region X = tl.float32, ScalePre = tl.float32, XInt8 = tl.float32, Scale = tl.float32
  off_blk = tl.program_id(0)
  off_b = tl.program_id(1)
  x_offset = off_b * $(p.rows) * $(p.columns)
  offs_m = off_blk * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_k = tl.arange(0, $(p.columns))
  x_ptrs = X + x_offset + offs_m[:, None] * $(p.columns) + offs_k[None, :]
  x_int8_ptrs = XInt8 + x_offset + offs_m[:, None] * $(p.columns) + offs_k[None, :]
  scale = tl.load(ScalePre + off_b * $(p.scaleStride) + off_blk)
  x = tl.load(x_ptrs, mask=offs_m[:, None] < $(p.rows))
  x_int8 = ($(p.preScale) * x) / scale
  tl.store(x_int8_ptrs, x_int8, mask=offs_m[:, None] < $(p.rows))
  tl.store(Scale + off_b * $(p.scaleStride) + off_blk, scale)
}

def kernel (p : Parameters) := source p.x p.suppliedScale p.out p.scaleOut p

def scaleOffset {α : Type} (p : Parameters) (s : State α) := s.pids 1 * p.scaleStride + s.pids 0

def row {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.columns]) :=
  s.pids 0 * p.blockRows + i.1.val

def offset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.blockRows,p.columns]) :=
  s.pids 1 * p.rows * p.columns + row p s i * p.columns + i.2.1.val

def scale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.fp32Load ((s.mem p.suppliedScale (scaleOffset p s)).read .real)

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.blockRows,p.columns]) : α :=
  M.binary (some .fp32) .real .div
    (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real p.preScale)
      (M.fp32Load ((s.mem p.x (offset p s i)).read .real))) (scale M p s)

def rowWritten {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.columns]).foldl (fun t i =>
    if row p s i < p.rows then t.write p.out (offset p s i) (.mk .real (value M p s i)) else t) s

def reference {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (rowWritten M p s).write p.scaleOut (scaleOffset p s) (.mk .real (scale M p s))

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem fold_expanded {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then
      { t with mem := fun r' o => if r' = r ∧ o = off i then value i else t.mem r' o }
      else t) s).mem =
    l.foldl (fun mem i => fun r' o => if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem :=
  fold_memory l r off value active s

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

theorem source_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧ t.mem = (reference M p s).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [kernel, source, ComputeKernel.surfaceBody,
    FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
    State.setReg, reference, rowWritten, value, scale, scaleOffset, offset, row,
    ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
    numericLt, natLt, Option.bind, fold_regs, fold_pids]
  try simp only [fold_memory]
  try simp +contextual (config := { maxSteps := 1000000 }) [State.write, fold_memory, fold_expanded]
  funext r o
  split
  · rfl
  · apply congrArg (fun (mem : RegionName → Nat → Cell α) => mem r o)
    congr 1
    funext mem i r2 o2
    split
    · rename_i h
      simp [h.1]
    · rfl

theorem offset_injective {α : Type} (p : Parameters) (s : State α) :
    Function.Injective (offset p s) := by
  intro i j h
  unfold offset at h
  rw [Nat.add_assoc, Nat.add_assoc] at h
  have hbase := Nat.add_left_cancel h
  have hrow : row p s i = row p s j := by
    have hi := i.2.1.isLt
    have hj := j.2.1.isLt
    have hC : 0 < p.columns := Nat.zero_lt_of_lt hi
    have hh := congrArg (fun n => n / p.columns) hbase
    simpa [Nat.mul_comm, Nat.mul_add_div hC, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj] using hh
  have hr : i.1 = j.1 := Fin.ext (Nat.add_left_cancel hrow)
  have hc : i.2.1 = j.2.1 := Fin.ext (by simpa [hrow] using hbase)
  exact Prod.ext hr (Prod.ext hc (Subsingleton.elim _ _))

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (r ≠ p.scaleOut ∨ o ≠ scaleOffset p s) ∧
    (r ≠ p.out ∨ ∀ i, row p s i < p.rows → o ≠ offset p s i)

theorem source_correct {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hregions : p.out ≠ p.scaleOut) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ i, row p s i < p.rows → t.mem p.out (offset p s i) = .mk .real (value M p s i)) ∧
      t.mem p.scaleOut (scaleOffset p s) = .mk .real (scale M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run M p s
  refine ⟨t, he, ?_, ?_, ?_⟩
  · intro i hi
    rw [hm]
    simp only [reference, State.write, hregions, false_and, if_false]
    exact State.masked_scatter_readback p.out (offset p s) _ (fun i => row p s i < p.rows)
      s i hi (offset_injective p s)
  · rw [hm]
    simp [reference, State.write]
  · intro r o h
    rw [hm]
    have hnot : ¬ (r = p.scaleOut ∧ o = scaleOffset p s) := by
      rcases h.1 with hr | ho
      · exact fun hx => hr hx.1
      · exact fun hx => ho hx.2
    simp only [reference, State.write, hnot, if_false]
    unfold rowWritten
    apply State.masked_scatter_frame
    rcases h.2 with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)
#axiomsClean source_run
#axiomsClean offset_injective
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled

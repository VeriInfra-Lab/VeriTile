import bench.tritonbench_g.swiglu_fwd.Proofs.Real
import bench.tritonbench_g.swiglu_fwd.Proofs.Math
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

structure Layout where
  X : RegionName
  Y : RegionName
  OUT : RegionName
  rows : Nat
  cols : Nat
  sx : Nat
  sy : Nat
  so : Nat

def Layout.source (c : Layout) (B : Nat) : ComputeKernel :=
  typedKernel .fp32 .fp32 .fp32 Bool.false c.X c.Y c.OUT c.sx c.sy c.so c.cols B

/-- FP32 is represented by the existing typed source's real carrier plus the
explicit some fp32 computation tag. Neither load cast is eliminated. -/
def scalar (M : Algebra α) (x y : α) : α :=
  let x := M.cast (some .fp32) .real .real x
  let y := M.cast (some .fp32) .real .real y
  M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul x (M.unary (some .fp32) .sigmoid x)) y

def raw {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (r b : Nat) (s : State α) (region : RegionName) (stride : Nat)
    (i : TileIndex [B]) : α :=
  if b * B + i.1.val < c.cols then
    (s.mem region (r * stride + (b * B + i.1.val))).read .real
  else M.literal (some .fp32) .real 0.0

def tileState {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (r b : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then r else if ax = 1 then b else 0 }

/-- Constructed by reducing all ten installed source statements. -/
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (r b : Nat) (s : State α) :
    { t : State α // run M (c.source B).surfaceBody (tileState M c B r b s) = some t } := by
  refine ⟨?_, ?_⟩
  swap
  simp (maxSteps := 500000) [Layout.source, typedKernel, typed_fp32_fp32,
    ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas, store,
    Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
    tileState, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
  rfl

def writeCell (mem : RegionName → Nat → Cell α) (r : RegionName)
    (a : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

def tileMemory {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [B]).foldl (fun mem i =>
    if b * B + i.1.val < c.cols then
      writeCell mem c.OUT (r * c.so + (b * B + i.1.val))
        (.mk .real (scalar M (raw M c B r b s c.X c.sx i)
          (raw M c B r b s c.Y c.sy i)))
    else mem) s.mem

theorem scatter_mem {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      lanes.foldl (fun mem i => if active i then writeCell mem r (off i) (value i) else mem) s.mem := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi, if_true, State.write, writeCell] using ih (s.write r (off i) (value i))
    · simpa only [hi, if_false] using ih s

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) :
    (sourceRun M c B r b s).val.mem = tileMemory M c B r b s := by
  simp only [sourceRun, store]
  rw [scatter_mem]
  simp only [tileMemory, scalar, raw]

private theorem fold_written {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (target : ι) (unique : ∀ i, active i → off i = off target → i = target)
    (lanes : List ι) (mem : RegionName → Nat → Cell α)
    (h : mem r (off target) = value target) :
    (lanes.foldl (fun current i => if active i then writeCell current r (off i) (value i)
      else current) mem) r (off target) = value target := by
  induction lanes generalizing mem with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hi : active i
    · simp only [hi, if_true]
      by_cases ho : off i = off target
      · have hit := unique i hi ho
        subst i
        simp [writeCell]
      · simp [writeCell, Ne.symm ho, h]
    · simpa only [hi, if_false] using h

theorem tile_readback {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) (target : TileIndex [B])
    (active : b * B + target.1.val < c.cols) :
    (sourceRun M c B r b s).val.mem c.OUT (r * c.so + (b * B + target.1.val)) =
      .mk .real (scalar M
        ((s.mem c.X (r * c.sx + (b * B + target.1.val))).read .real)
        ((s.mem c.Y (r * c.sy + (b * B + target.1.val))).read .real)) := by
  rw [sourceRun_mem]
  have unique : ∀ i : TileIndex [B], b * B + i.1.val < c.cols →
      r * c.so + (b * B + i.1.val) = r * c.so + (b * B + target.1.val) → i = target := by
    intro i hi heq
    apply Prod.ext
    · apply Fin.ext
      omega
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ := List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold tileMemory
  rw [hall, List.foldl_append, List.foldl_cons]
  simp only [active, if_true]
  suffices h : _ = Cell.mk .real (scalar M (raw M c B r b s c.X c.sx target)
      (raw M c B r b s c.Y c.sy target)) by
    simpa only [raw, active, if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell, raw, active]

def Outside (c : Layout) (B r b : Nat) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.OUT ∨ ∀ i : TileIndex [B], b * B + i.1.val < c.cols →
    a ≠ r * c.so + (b * B + i.1.val)

private theorem fold_frame {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (mem : RegionName → Nat → Cell α) (r : RegionName) (a : Nat)
    (outside : r ≠ region ∨ ∀ i, active i → a ≠ off i) :
    (lanes.foldl (fun current i => if active i then writeCell current region (off i) (value i)
      else current) mem) r a = mem r a := by
  induction lanes generalizing mem with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : active i
    · simp only [hi, if_true, writeCell]
      rw [if_neg]
      rcases outside with hr | ha
      · exact fun h => hr h.1
      · exact fun h => ha i hi h.2
    · simp only [hi, if_false]

theorem tile_frame {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) (region : RegionName) (a : Nat)
    (outside : Outside c B r b region a) :
    (sourceRun M c B r b s).val.mem region a = s.mem region a := by
  rw [sourceRun_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole

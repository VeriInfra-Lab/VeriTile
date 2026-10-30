import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_Real
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_Math
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

inductive Channel where
  | dx | dy | out
  deriving DecidableEq, Repr

structure Layout where
  X : RegionName
  Y : RegionName
  DOUT : RegionName
  DX : RegionName
  DY : RegionName
  OUT : RegionName
  rows : Nat
  cols : Nat
  sx : Nat
  sy : Nat
  sd : Nat
  sdx : Nat
  sdy : Nat
  so : Nat

def Layout.output (c : Layout) : Channel → RegionName
  | .dx => c.DX | .dy => c.DY | .out => c.OUT

def Layout.stride (c : Layout) : Channel → Nat
  | .dx => c.sdx | .dy => c.sdy | .out => c.so

def enabled (R : Bool) (ch : Channel) : Prop := ch ≠ .out ∨ R = Bool.true
instance (R : Bool) (ch : Channel) : Decidable (enabled R ch) := inferInstanceAs (Decidable (ch ≠ .out ∨ R = Bool.true))

def Layout.source (c : Layout) (B : Nat) (R : Bool) : ComputeKernel :=
  typedKernel .fp32 .fp32 .fp32 .fp32 Bool.false c.X c.Y c.DOUT c.OUT c.DX c.DY
    c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R

/-- Source FP32 graph. Three explicit input casts and left-associated products
are retained; the independent mathematical specification lives in Math. -/
def scalar (M : Algebra α) (ch : Channel) (x y d : α) : α :=
  let x := M.cast (some .fp32) .real .real x
  let y := M.cast (some .fp32) .real .real y
  let d := M.cast (some .fp32) .real .real d
  let sig := M.unary (some .fp32) .sigmoid x
  let one := M.literal (some .fp32) .real 1
  let mul := M.binary (some .fp32) .real .mul
  match ch with
  | .dx => mul (mul (mul sig (M.binary (some .fp32) .real .add one
      (mul x (M.binary (some .fp32) .real .sub one sig)))) y) d
  | .dy => mul (mul x sig) d
  | .out => mul (mul x sig) y

def raw {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (r b : Nat) (s : State α) (region : RegionName) (stride : Nat)
    (i : TileIndex [B]) : α :=
  if b * B + i.1.val < c.cols then
    (s.mem region (r * stride + (b * B + i.1.val))).read .real
  else M.literal (some .fp32) .real 0.0

def tileState {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (r b : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then r else if ax = 1 then b else 0 }

@[simp] theorem scatter_regs {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write region (off i) (value i) else t) s).regs = s.regs := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi,if_true,State.write] using ih (s.write region (off i) (value i))
    · simpa only [hi,if_false] using ih s

/-- Reduction of every statement of the installed typed original source,
including both conditional pointer/store blocks. -/
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (R : Bool) (r b : Nat) (s : State α) :
    { t : State α // run M (c.source B R).surfaceBody (tileState M c B r b s) = some t } := by
  cases R
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 900000) [Layout.source, typedKernel, template, typedLoad, typedStore,
      loadAddress, storeAddress, recomputedValue, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas, store,
      Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
      tileState, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
    rfl

def writeCell (mem : RegionName → Nat → Cell α) (r : RegionName)
    (a : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

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

theorem fold_written {α ι : Type} (r : RegionName) (off : ι → Nat)
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

theorem fold_frame {α ι : Type} (region : RegionName) (off : ι → Nat)
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

def scatter {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [B]).foldl (fun mem i =>
    if b*B+i.1.val < c.cols then
      writeCell mem (c.output ch) (r*c.stride ch+(b*B+i.1.val))
        (.mk .real (scalar M ch (raw M c B r b s c.X c.sx i)
          (raw M c B r b s c.Y c.sy i) (raw M c B r b s c.DOUT c.sd i)))
    else mem) mem

def tileMemory {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (R : Bool) (r b : Nat) (s : State α) : RegionName → Nat → Cell α :=
  let dx := scatter M c B r b s .dx s.mem
  let dy := scatter M c B r b s .dy dx
  if R then scatter M c B r b s .out dy else dy

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (R : Bool) (r b : Nat) (s : State α) :
    (sourceRun M c B R r b s).val.mem = tileMemory M c B R r b s := by
  cases R <;> simp only [sourceRun, store, scatter_mem]
  all_goals rfl

theorem scatter_readback {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α)
    (target : TileIndex [B]) (active : b*B+target.1.val < c.cols) :
    scatter M c B r b s ch mem (c.output ch) (r*c.stride ch+(b*B+target.1.val)) =
      .mk .real (scalar M ch
        ((s.mem c.X (r*c.sx+(b*B+target.1.val))).read .real)
        ((s.mem c.Y (r*c.sy+(b*B+target.1.val))).read .real)
        ((s.mem c.DOUT (r*c.sd+(b*B+target.1.val))).read .real)) := by
  have unique : ∀ i : TileIndex [B], b*B+i.1.val < c.cols →
      r*c.stride ch+(b*B+i.1.val) = r*c.stride ch+(b*B+target.1.val) → i = target := by
    intro i _ heq
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ := List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold scatter
  rw [hall, List.foldl_append, List.foldl_cons]
  simp only [active, if_true]
  suffices h : _ = Cell.mk .real (scalar M ch (raw M c B r b s c.X c.sx target)
      (raw M c B r b s c.Y c.sy target) (raw M c B r b s c.DOUT c.sd target)) by
    simpa only [raw, active, if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell, raw, active]

def Outside (c : Layout) (B : Nat) (R : Bool) (r b : Nat) (region : RegionName) (a : Nat) : Prop :=
  ∀ ch, enabled R ch → region ≠ c.output ch ∨ ∀ i : TileIndex [B],
    b*B+i.1.val < c.cols → a ≠ r*c.stride ch+(b*B+i.1.val)

theorem scatter_frame {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B r b : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α)
    (region : RegionName) (a : Nat)
    (outside : region ≠ c.output ch ∨ ∀ i : TileIndex [B],
      b*B+i.1.val < c.cols → a ≠ r*c.stride ch+(b*B+i.1.val)) :
    scatter M c B r b s ch mem region a = mem region a := by
  exact fold_frame _ _ _ _ _ _ _ _ outside

theorem tile_frame {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (R : Bool) (r b : Nat) (s : State α) (region : RegionName) (a : Nat)
    (outside : Outside c B R r b region a) :
    (sourceRun M c B R r b s).val.mem region a = s.mem region a := by
  rw [sourceRun_mem]
  cases R <;> simp only [tileMemory, Bool.false_eq_true, if_false, if_true]
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .dy (Or.inl (by decide))),
      scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .dx (Or.inl (by decide)))]
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .out (Or.inr rfl)),
      scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .dy (Or.inl (by decide))),
      scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .dx (Or.inl (by decide)))]

theorem tile_readback {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (separate : Function.Injective c.output) (B : Nat) (R : Bool) (r b : Nat) (s : State α)
    (ch : Channel) (on : enabled R ch) (target : TileIndex [B])
    (active : b*B+target.1.val < c.cols) :
    (sourceRun M c B R r b s).val.mem (c.output ch) (r*c.stride ch+(b*B+target.1.val)) =
      .mk .real (scalar M ch
        ((s.mem c.X (r*c.sx+(b*B+target.1.val))).read .real)
        ((s.mem c.Y (r*c.sy+(b*B+target.1.val))).read .real)
        ((s.mem c.DOUT (r*c.sd+(b*B+target.1.val))).read .real)) := by
  have distinct {a b : Channel} (h : a ≠ b) : c.output a ≠ c.output b := fun e => h (separate e)
  rw [sourceRun_mem]
  cases R <;> cases ch <;> simp only [tileMemory, Bool.false_eq_true, if_false, if_true]
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (distinct (by decide)))]
    exact scatter_readback _ _ _ _ _ _ _ _ _ active
  · exact scatter_readback _ _ _ _ _ _ _ _ _ active
  · simp [enabled] at on
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (distinct (by decide))),
      scatter_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (distinct (by decide)))]
    exact scatter_readback _ _ _ _ _ _ _ _ _ active
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (distinct (by decide)))]
    exact scatter_readback _ _ _ _ _ _ _ _ _ active
  · exact scatter_readback _ _ _ _ _ _ _ _ _ active

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluBackward.Whole

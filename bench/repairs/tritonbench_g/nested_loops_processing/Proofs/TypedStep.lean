import bench.repairs.tritonbench_g.nested_loops_processing.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

abbrev OffsetTile := TileIndex [2,2] → Nat
abbrev Payload := TileIndex [2,2] → Int
abbrev Unique := TileIndex [2,2] → Bool

def outer (count cols : Nat) : List ComputeStmt :=
  match ((source count cols).surfaceBody[7]? : Option ComputeStmt) with
  | some (.forRange "i" 0 2 1 body) => body
  | _ => []
def middle (count cols : Nat) : List ComputeStmt :=
  match ((outer count cols)[1]? : Option ComputeStmt) with
  | some (.forRange "j" 0 2 1 body) => body
  | _ => []
def inner (count cols : Nat) : List ComputeStmt :=
  match ((middle count cols)[2]? : Option ComputeStmt) with
  | some (.forRange "k" 0 2 1 body) => body
  | _ => []

/-- The canonical source retains all loop bodies and the three barriers
in each inner iteration. Region declarations select int32 loads and stores. -/
theorem full_loop_shape (count cols : Nat) :
    (source count cols).surfaceBody.length = 8 ∧
    (outer count cols).length = 3 ∧
    (middle count cols).length = 3 ∧
    (inner count cols).length = 11 := by exact ⟨rfl, rfl, rfl, rfl⟩

theorem inner_barriers (count cols : Nat) :
    (inner count cols)[3]? = some (.effectMarker "tl.debug_barrier") ∧
    (inner count cols)[6]? = some (.effectMarker "tl.debug_barrier") ∧
    (inner count cols)[9]? = some (.effectMarker "tl.debug_barrier") := by
  exact ⟨rfl, rfl, rfl⟩

def bump (ptr : OffsetTile) (delta : Nat) : OffsetTile := fun i ↦ ptr i + delta

def loaded {α : Type} [Inhabited α] (count : Nat) (s : State α) (ptr : OffsetTile) : Payload :=
  fun i ↦ if ptr i < count then (s.mem "input" (min (ptr i) (count-1))).read .int else 0

def scatter {α : Type} (count : Nat) (u : Unique) (s : State α)
    (ptr : OffsetTile) (value : Payload) : State α :=
  (TileShape.allIndices [2,2]).foldl (fun t i ↦
    if decide (ptr i < count) && u i then
      t.write "output" (min (ptr i) (count-1)) (.mk .int (value i)) else t) s

@[simp] theorem scatter_regs {α : Type} (count : Nat) (u : Unique) (s : State α)
    (ptr : OffsetTile) (value : Payload) :
    (scatter count u s ptr value).regs = s.regs := by
  have fold : ∀ (ls : List (TileIndex [2,2])) (s : State α),
      (ls.foldl (fun t i ↦ if decide (ptr i<count) && u i then
        t.write "output" (min (ptr i) (count-1)) (.mk .int (value i)) else t) s).regs = s.regs := by
    intro ls
    induction ls with
    | nil => intro s; rfl
    | cons i ls ih =>
      intro s
      rw [List.foldl_cons, ih]
      split <;> rfl
  exact fold _ s

@[simp] theorem masked_fold_regs {α ι : Type} (cond : ι → Prop)
    [DecidablePred cond] (off : ι → Nat) (value : ι → Int) (ls : List ι) (s : State α) :
    (ls.foldl (fun t i ↦ if cond i then t.write "output" (off i) (.mk .int (value i)) else t) s).regs = s.regs := by
  induction ls generalizing s with
  | nil => rfl
  | cons i ls ih => rw [List.foldl_cons, ih]; split <;> rfl

def innerState {α : Type} [Inhabited α] (count : Nat) (u : Unique)
    (s : State α) (a c : OffsetTile) (v1 v2 : Payload) (delta : Nat) : State α :=
  let a' := bump a delta
  let v3 := loaded count s a'
  let t := s.setReg "a_offsets" .nat [2,2] a'
  let t := t.setReg "a3" .int [2,2] v3
  let t := scatter count u t c v1
  let t := t.setReg "c_offsets" .nat [2,2] (bump c delta)
  let t := scatter count u t (bump c delta) v2
  let t := t.setReg "c_offsets" .nat [2,2] (bump (bump c delta) delta)
  let t := scatter count u t (bump (bump c delta) delta) v3
  t.setReg "c_offsets" .nat [2,2] (bump (bump (bump c delta) delta) delta)

theorem inner_step {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α)
    (a c : OffsetTile) (v1 v2 : Payload) (u : Unique)
    (ha : s.regs .nat [2,2] "a_offsets" = some a)
    (hc : s.regs .nat [2,2] "c_offsets" = some c)
    (h1 : s.regs .int [2,2] "a1" = some v1)
    (h2 : s.regs .int [2,2] "a2" = some v2)
    (hu : s.regs .bool [2,2] "unique" = some u) :
    run M (inner count cols) s = some (innerState count u s a c v1 v2 2) := by
  simp only [inner, middle, outer, source, kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Bool.false_eq_true, ↓reduceIte,
    run, step, evalExpr, evalComputeOp]
  simp [ComputeDType.eraseDType, Option.bind, evalOp_unfold, FP.Structural.store, numeric, natLt, natLe,
    ha, hc, h1, h2, hu, scatter, innerState, loaded, bump, ← min_def_lt]
  unfold FP.Structural.bop
  simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex,
    ha, hc, h1, h2, hu, scatter, innerState, loaded, bump, ← min_def_lt]
  rfl

end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms

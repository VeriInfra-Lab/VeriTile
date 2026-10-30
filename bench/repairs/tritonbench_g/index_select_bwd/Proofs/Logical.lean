import bench.tritonbench_g.index_select_bwd.Proofs.Math
import Mathlib.Algebra.BigOperators.Group.List.Basic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Logical
noncomputable section
open Classical
abbrev Coord (rows cols : Nat) := Fin rows × Fin cols

/-- Independent row-gather derivative, including every duplicate and the
original nonzero destination. No source, schedule or final memory occurs here. -/
def target {R N C : Nat} (index : Fin N → Fin R) (g : Coord R C → ℝ)
    (h : Coord N C → ℝ) (q : Coord R C) : ℝ :=
  g q + ((List.finRange N).map fun i => if index i=q.1 then h (i,q.2) else 0).sum

/-- Production arithmetic order: increasing input row, initial destination as
left-fold seed. Columns do not reorder any individual cell's additions. -/
def row {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N)
    (g : Coord R C → A) : Coord R C → A :=
  fun q => if index i=q.1 then add (g q) (h (i,q.2)) else g q

def rows {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A)
    (is : List (Fin N)) (g : Coord R C → A) : Coord R C → A :=
  is.foldl (fun state i => row add index h i state) g

theorem rows_cell {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (is : List (Fin N))
    (g : Coord R C → A) (q : Coord R C) :
    rows add index h is g q = is.foldl
      (fun acc i => if index i=q.1 then add acc (h (i,q.2)) else acc) (g q) := by
  induction is generalizing g with
  | nil => rfl
  | cons i rest ih => simpa only [rows,List.foldl_cons,row] using ih (row add index h i g)

theorem real_rows {R N C : Nat} (index : Fin N → Fin R) (h : Coord N C → ℝ)
    (is : List (Fin N)) (g : Coord R C → ℝ) (q : Coord R C) :
    rows (·+·) index h is g q = g q + (is.map fun i => if index i=q.1 then h (i,q.2) else 0).sum := by
  induction is generalizing g with
  | nil => simp [rows]
  | cons i rest ih =>
    change rows (·+·) index h rest (row (·+·) index h i g) q = _
    rw [ih]
    by_cases same : index i=q.1 <;> simp [row,same,List.map_cons,List.sum_cons,add_assoc]

theorem complete_real {R N C : Nat} (index : Fin N → Fin R) (g : Coord R C → ℝ)
    (h : Coord N C → ℝ) (q : Coord R C) :
    rows (·+·) index h (List.finRange N) g q = target index g h q := real_rows _ _ _ _ _

theorem unselected {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (is : List (Fin N))
    (g : Coord R C → A) (q : Coord R C) (miss : ∀ i∈is,index i≠q.1) :
    rows add index h is g q=g q := by
  induction is generalizing g with
  | nil => rfl
  | cons i rest ih =>
    change rows add index h rest (row add index h i g) q=g q
    rw [ih _ (fun j hj => miss j (by simp [hj]))]
    simp [row,miss i (by simp)]

/-- Reuse original row-major helper; only the erroneous overwrite operation is
replaced. Its enumeration itself was already correct. -/
abbrev events (N C : Nat) := VeriTile.Bench.TritonBenchG.IndexSelectBwd.scatterLanes N C

def eventStep {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (e : Coord N C)
    (g : Coord R C → A) : Coord R C → A :=
  fun q => if q=(index e.1,e.2) then add (g q) (h e) else g q

def eventRun {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A)
    (es : List (Coord N C)) (g : Coord R C → A) : Coord R C → A :=
  es.foldl (fun state e => eventStep add index h e state) g

theorem eventRun_cell {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (es : List (Coord N C))
    (g : Coord R C → A) (q : Coord R C) :
    eventRun add index h es g q=es.foldl
      (fun acc e => if q=(index e.1,e.2) then add acc (h e) else acc) (g q) := by
  induction es generalizing g with
  | nil => rfl
  | cons e rest ih => simpa only [eventRun,List.foldl_cons,eventStep] using ih (eventStep add index h e g)

theorem fold_once {A K : Type} [DecidableEq K] (f : A → A) (q : K)
    (ks : List K) (unique : ks.Nodup) (a : A) :
    ks.foldl (fun acc k => if k=q then f acc else acc) a = if q∈ks then f a else a := by
  induction ks generalizing a with
  | nil => simp
  | cons k rest ih =>
    have nd := List.nodup_cons.mp unique
    simp only [List.foldl_cons]
    rw [ih nd.2]
    by_cases same : k=q
    · subst k
      simp [nd.1]
    · simp [same,eq_comm (a:=q) (b:=k)]

theorem columns {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N)
    (g : Coord R C → A) :
    eventRun add index h ((List.finRange C).map fun c => (i,c)) g=row add index h i g := by
  funext q
  rw [eventRun_cell,List.foldl_map]
  by_cases same : index i=q.1
  · have step : (fun (acc : A) (c : Fin C) => if q=(index i,c) then add acc (h (i,c)) else acc) =
        (fun acc c => if c=q.2 then add acc (h (i,q.2)) else acc) := by
      funext acc c
      by_cases hc : c=q.2
      · subst c; simp [same]
      · have ne : q≠(index i,c) := by intro he; exact hc (congrArg Prod.snd he).symm
        simp [hc,ne]
    rw [step,fold_once _ _ _ (List.nodup_finRange C)]
    simp [row,same]
  · have ne : ∀ c : Fin C,q≠(index i,c) := by
      intro c he; exact same (congrArg Prod.fst he).symm
    simp only [ne,if_false]
    simp [row,same]

theorem events_rows {A : Type} {R N C : Nat} (add : A → A → A)
    (index : Fin N → Fin R) (h : Coord N C → A) (g : Coord R C → A) :
    eventRun add index h (events N C) g=rows add index h (List.finRange N) g := by
  unfold events VeriTile.Bench.TritonBenchG.IndexSelectBwd.scatterLanes
  generalize List.finRange N=is
  induction is generalizing g with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.flatMap_cons,eventRun,List.foldl_append]
    change eventRun add index h (rest.flatMap fun i => (List.finRange C).map fun c => (i,c))
      (eventRun add index h ((List.finRange C).map fun c => (i,c)) g)=_
    rw [columns,ih]
    rfl

#axiomsClean events_rows
#specNonCircular target avoiding [rows,row]
#axiomsClean complete_real
#axiomsClean rows_cell
#axiomsClean unselected
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Logical

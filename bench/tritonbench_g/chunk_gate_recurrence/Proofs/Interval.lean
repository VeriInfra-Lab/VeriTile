import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Machine

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine
instance (x : Int) : Decidable (I32 x) := inferInstanceAs (Decidable (_∧_))

structure Interval where
  lo : Int
  hi : Int
  deriving DecidableEq,Repr

def Interval.Contains (i : Interval) (x : Int) : Prop := i.lo ≤ x ∧ x ≤ i.hi

def Interval.singleton (x : Int) : Interval := ⟨x,x⟩
def Interval.add (a b : Interval) : Interval := ⟨a.lo+b.lo,a.hi+b.hi⟩
def Interval.sub (a b : Interval) : Interval := ⟨a.lo-b.hi,a.hi-b.lo⟩
def Interval.mul (a b : Interval) : Interval :=
  ⟨min (min (a.lo*b.lo) (a.lo*b.hi)) (min (a.hi*b.lo) (a.hi*b.hi)),
   max (max (a.lo*b.lo) (a.lo*b.hi)) (max (a.hi*b.lo) (a.hi*b.hi))⟩
def Interval.div (a b : Interval) : Interval := ⟨a.lo/b.hi,a.hi/b.lo⟩
def Interval.mod (_a b : Interval) : Interval := ⟨0,b.hi-1⟩
def Interval.safe (i : Interval) : Bool := decide (I32 i.lo ∧ I32 i.hi ∧ i.lo ≤ i.hi)

theorem Interval.safe_value (i : Interval) (x : Int) (safe : i.safe=Bool.true) (inside : i.Contains x) : I32 x := by
  simp only [Interval.safe,decide_eq_true_eq,I32,Interval.Contains] at *
  omega

theorem Interval.add_contains (a b : Interval) (x y : Int) (hx : a.Contains x) (hy : b.Contains y) :
    (a.add b).Contains (x+y) := by simp only [Contains,add] at *; omega

theorem Interval.sub_contains (a b : Interval) (x y : Int) (hx : a.Contains x) (hy : b.Contains y) :
    (a.sub b).Contains (x-y) := by simp only [Contains,sub] at *; omega

theorem scalar_bounds (a : Int) (b : Interval) (y : Int) (hy : b.Contains y) :
    min (a*b.lo) (a*b.hi) ≤ a*y ∧ a*y ≤ max (a*b.lo) (a*b.hi) := by
  by_cases positive : 0 ≤ a
  · have lo := mul_le_mul_of_nonneg_left hy.1 positive
    have hi := mul_le_mul_of_nonneg_left hy.2 positive
    exact ⟨le_trans (min_le_left _ _) lo,le_trans hi (le_max_right _ _)⟩
  · have lo := mul_le_mul_of_nonpos_left hy.2 (by omega : a ≤ 0)
    have hi := mul_le_mul_of_nonpos_left hy.1 (by omega : a ≤ 0)
    exact ⟨le_trans (min_le_right _ _) lo,le_trans hi (le_max_left _ _)⟩

theorem Interval.mul_contains (a b : Interval) (x y : Int) (hx : a.Contains x) (hy : b.Contains y) :
    (a.mul b).Contains (x*y) := by
  have low := scalar_bounds a.lo b y hy
  have high := scalar_bounds a.hi b y hy
  have lowLo := min_le_left (min (a.lo*b.lo) (a.lo*b.hi)) (min (a.hi*b.lo) (a.hi*b.hi))
  have highLo := min_le_right (min (a.lo*b.lo) (a.lo*b.hi)) (min (a.hi*b.lo) (a.hi*b.hi))
  have lowHi := le_max_left (max (a.lo*b.lo) (a.lo*b.hi)) (max (a.hi*b.lo) (a.hi*b.hi))
  have highHi := le_max_right (max (a.lo*b.lo) (a.lo*b.hi)) (max (a.hi*b.lo) (a.hi*b.hi))
  simp only [Contains,mul]
  by_cases positive : 0 ≤ y
  · have lo := mul_le_mul_of_nonneg_right hx.1 positive
    have hi := mul_le_mul_of_nonneg_right hx.2 positive
    omega
  · have lo := mul_le_mul_of_nonpos_right hx.2 (by omega : y ≤ 0)
    have hi := mul_le_mul_of_nonpos_right hx.1 (by omega : y ≤ 0)
    omega


theorem ediv_antitone (x b c : Int) (hx : 0 ≤ x) (hb : 0 < b) (hbc : b ≤ c) : x/c ≤ x/b := by
  apply Int.le_ediv_of_mul_le hb
  have nonnegative := Int.ediv_nonneg hx (by omega : 0 ≤ c)
  exact le_trans (mul_le_mul_of_nonneg_left hbc nonnegative) (Int.ediv_mul_le x (by omega : c≠0))

theorem Interval.div_contains (a b : Interval) (x y : Int) (hx : a.Contains x) (hy : b.Contains y)
    (nonnegative : 0 ≤ a.lo) (positive : 0 < b.lo) : (a.div b).Contains (x/y) := by
  have yl : 0 < y := lt_of_lt_of_le positive hy.1
  exact ⟨le_trans (ediv_antitone a.lo y b.hi nonnegative yl hy.2) (Int.ediv_le_ediv yl hx.1),
    le_trans (Int.ediv_le_ediv yl hx.2) (ediv_antitone a.hi b.lo y (le_trans nonnegative (le_trans hx.1 hx.2)) positive hy.1)⟩

theorem Interval.mod_contains (a b : Interval) (x y : Int) (_hx : a.Contains x) (hy : b.Contains y)
    (positive : 0 < b.lo) : (a.mod b).Contains (x%y) := by
  have yl : 0 < y := lt_of_lt_of_le positive hy.1
  have nonnegative := Int.emod_nonneg x (by omega : y≠0)
  have bound := Int.emod_lt_of_pos x yl
  refine ⟨nonnegative,?_⟩
  change x%y ≤ b.hi-1
  have h := hy.2
  omega

inductive BExpr where
  | lit (interval : Interval)
  | add (a b : BExpr)
  | sub (a b : BExpr)
  | mul (a b : BExpr)
  | div (a b : BExpr)
  | mod (a b : BExpr)

def BExpr.bound : BExpr → Interval
  | .lit i => i
  | .add a b => a.bound.add b.bound
  | .sub a b => a.bound.sub b.bound
  | .mul a b => a.bound.mul b.bound
  | .div a b => a.bound.div b.bound
  | .mod a b => a.bound.mod b.bound

def BExpr.valid (e : BExpr) : Bool :=
  match e with
  | .lit i => i.safe
  | .add a b | .sub a b | .mul a b => a.valid && b.valid && e.bound.safe
  | .div a b | .mod a b => a.valid && b.valid && decide (0 ≤ a.bound.lo ∧ 0 < b.bound.lo) && e.bound.safe

inductive BExpr.Binds : BExpr → Expr → Prop where
  | lit {i : Interval} {x : Int} (inside : i.Contains x) : Binds (.lit i) (.lit x)
  | add {a b : BExpr} {x y : Expr} (left : Binds a x) (right : Binds b y) : Binds (.add a b) (.add x y)
  | sub {a b : BExpr} {x y : Expr} (left : Binds a x) (right : Binds b y) : Binds (.sub a b) (.sub x y)
  | mul {a b : BExpr} {x y : Expr} (left : Binds a x) (right : Binds b y) : Binds (.mul a b) (.mul x y)
  | div {a b : BExpr} {x y : Expr} (left : Binds a x) (right : Binds b y) : Binds (.div a b) (.div x y)
  | mod {a b : BExpr} {x y : Expr} (left : Binds a x) (right : Binds b y) : Binds (.mod a b) (.mod x y)

theorem BExpr.certified (a : BExpr) (e : Expr) (binding : a.Binds e) (checked : a.valid=Bool.true) :
    e.Valid ∧ a.bound.Contains e.eval := by
  induction binding with
  | lit inside => exact ⟨Interval.safe_value _ _ checked inside,inside⟩
  | add l r il ir =>
    simp only [valid,Bool.and_eq_true] at checked
    obtain ⟨lv,li⟩ := il checked.1.1
    obtain ⟨rv,ri⟩ := ir checked.1.2
    have inside := Interval.add_contains _ _ _ _ li ri
    exact ⟨⟨lv,rv,Interval.safe_value _ _ checked.2 inside⟩,inside⟩
  | sub l r il ir =>
    simp only [valid,Bool.and_eq_true] at checked
    obtain ⟨lv,li⟩ := il checked.1.1
    obtain ⟨rv,ri⟩ := ir checked.1.2
    have inside := Interval.sub_contains _ _ _ _ li ri
    exact ⟨⟨lv,rv,Interval.safe_value _ _ checked.2 inside⟩,inside⟩
  | mul l r il ir =>
    simp only [valid,Bool.and_eq_true] at checked
    obtain ⟨lv,li⟩ := il checked.1.1
    obtain ⟨rv,ri⟩ := ir checked.1.2
    have inside := Interval.mul_contains _ _ _ _ li ri
    exact ⟨⟨lv,rv,Interval.safe_value _ _ checked.2 inside⟩,inside⟩
  | div l r il ir =>
    simp only [valid,Bool.and_eq_true,decide_eq_true_eq] at checked
    obtain ⟨lv,li⟩ := il checked.1.1.1
    obtain ⟨rv,ri⟩ := ir checked.1.1.2
    have inside := Interval.div_contains _ _ _ _ li ri checked.1.2.1 checked.1.2.2
    exact ⟨⟨lv,rv,le_trans checked.1.2.1 li.1,lt_of_lt_of_le checked.1.2.2 ri.1,
      Interval.safe_value _ _ checked.2 inside⟩,inside⟩
  | mod l r il ir =>
    simp only [valid,Bool.and_eq_true,decide_eq_true_eq] at checked
    obtain ⟨lv,li⟩ := il checked.1.1.1
    obtain ⟨rv,ri⟩ := ir checked.1.1.2
    have inside := Interval.mod_contains _ _ _ _ li ri checked.1.2.2
    exact ⟨⟨lv,rv,le_trans checked.1.2.1 li.1,lt_of_lt_of_le checked.1.2.2 ri.1,
      Interval.safe_value _ _ checked.2 inside⟩,inside⟩

#axiomsClean BExpr.certified
#axiomsClean Interval.mul_contains
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked

import bench.repairs.tritonbench_g.rope_transform.Proofs.Row
namespace VeriTile.Bench.Repairs.RopeTransform.Grid
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

def setup {α : Type} (r : Request) (a : Nat) (s : State α) : State α :=
  {s with regs:=fun _ _ _ => none
          pids:=fun ax => if ax=0 then a else 0
          numPids:=fun ax => if ax=0 then r.rows else 1}
def step {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (a : Nat) (s : State α) :=
  (T.execution M r.parameters (setup r a s)).val
def run {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (initial : State α) : Nat → State α
  | 0 => initial
  | n+1 => step M r n (run M r initial n)
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | n+1 => (sourceRun M r initial n).bind
    (fun s => FP.Structural.exec M (kernel r) (setup r n s))

theorem successful {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α) (n : Nat) :
    sourceRun M r s n=some (run M r s n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [sourceRun,ih,Option.bind_some]
    exact (T.execution M r.parameters (setup r n (run M r s n))).property

def Frame {α : Type} (r : Request) (n : Nat) (s t : State α) : Prop :=
  ∀ name o, (∀ key, name≠region key ∨ n*r.stride key≤o) → t.mem name o=s.mem name o

theorem prefix_frame {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α) (n : Nat) :
    Frame r n s (run M r s n) := by
  induction n with
  | zero => intro name o h; rfl
  | succ n ih =>
    intro name o ho
    change (T.execution M r.parameters (setup r n (run M r s n))).val.mem name o=_
    rw [frame M r (setup r n (run M r s n)) name o]
    · apply ih
      intro key
      rcases ho key with hn | hn
      · exact Or.inl hn
      · right
        exact le_trans (Nat.mul_le_mul_right (r.stride key) (by omega : n≤n+1)) hn
    · intro key
      rcases ho key with hn | hn
      · exact Or.inl hn
      · exact Or.inr (Or.inr hn)

theorem input_current {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (n : Nat) (key : Bool) (h c : Nat) (hh : h<r.heads key) (hc : c<r.dimension) :
    (run M r s n).mem (region key) (r.offset key n h c)=s.mem (region key) (r.offset key n h c) := by
  apply prefix_frame M r s n
  intro target
  by_cases hk : key=target
  · subst target; exact Or.inr (offset_window r key n h c hh hc).1
  · left; cases key <;> cases target <;> simp_all [region]

theorem coefficient_unchanged {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (n : Nat) (name : RegionName) (hn : name="C" ∨ name="S") (o : Nat) :
    (run M r s n).mem name o=s.mem name o := by
  apply prefix_frame M r s n
  intro key
  left
  rcases hn with hn|hn <;> subst name <;> cases key <;> decide

theorem value_current {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α)
    (n : Nat) (key second : Bool) (h j : Nat) (hh : h<r.heads key) (hj : j<r.half) :
    value M r (setup r n (run M r s n)) key n h second j=value M r s key n h second j := by
  have h0 : j<r.dimension := by unfold Request.dimension; omega
  have h1 : j+r.half<r.dimension := by unfold Request.dimension; omega
  unfold value
  simp only [setup]
  rw [input_current M r s n key h j hh h0,input_current M r s n key h (j+r.half) hh h1]
  rw [coefficient_unchanged M r s n "C" (Or.inl rfl),
    coefficient_unchanged M r s n "S" (Or.inr rfl)]

theorem outputs {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α) (n : Nat) :
    ∀ key a h second j, a<n → h<r.heads key → j<r.half →
      (run M r s n).mem (region key) (r.offset key a h (r.column second j))=
        Cell.mk .fp32 (value M r s key a h second j) := by
  induction n with
  | zero => intro key a h second j ha; omega
  | succ n ih =>
    intro key a h second j ha hh hj
    by_cases han : a=n
    · subst a
      change (T.execution M r.parameters (setup r n (run M r s n))).val.mem _ _=_
      have ht := readback M r (setup r n (run M r s n)) key second h j hh hj
      have hp : (setup r n (run M r s n)).pids 0=n := by simp [setup]
      rw [hp,value_current M r s n key second h j hh hj] at ht
      exact ht
    · have ha' : a<n := by omega
      have ho := (offset_window r key a h (r.column second j) hh (column_lt r second j hj)).2
      have hn := Nat.mul_le_mul_right (r.stride key) (show a+1≤n by omega)
      have frameHere : RowOutside r n (region key) (r.offset key a h (r.column second j)) := by
        intro target
        by_cases hk : key=target
        · subst target; exact Or.inr (Or.inl (lt_of_lt_of_le ho hn))
        · left; cases key <;> cases target <;> simp_all [region]
      change (T.execution M r.parameters (setup r n (run M r s n))).val.mem _ _=_
      rw [frame M r (setup r n (run M r s n)) _ _ frameHere]
      exact ih key a h second j ha' hh hj

/-- Complete sequential launch semantics. Race-free CTA scheduling is an
ordinary runtime binding; no arbitrary-interleaving equivalence is assumed. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (r : Request) (s : State α) :
    sourceRun M r s r.rows=some (run M r s r.rows) ∧
    (∀ key a h second j, a<r.rows → h<r.heads key → j<r.half →
      (run M r s r.rows).mem (region key) (r.offset key a h (r.column second j))=
        Cell.mk .fp32 (value M r s key a h second j)) ∧
    Frame r r.rows s (run M r s r.rows) :=
  ⟨successful M r s r.rows,outputs M r s r.rows,prefix_frame M r s r.rows⟩

theorem different_rows (r : Request) (key : Bool) (a b h g c d : Nat)
    (hh : h<r.heads key) (hg : g<r.heads key) (hc : c<r.dimension) (hd : d<r.dimension)
    (hab : a≠b) : r.offset key a h c≠r.offset key b g d := by
  intro he
  exact hab (offset_injective r key a b h g c d hh hg hc hd he).1

#axiomsClean complete
#axiomsClean different_rows
#axiomsClean input_current
end
end VeriTile.Bench.Repairs.RopeTransform.Grid

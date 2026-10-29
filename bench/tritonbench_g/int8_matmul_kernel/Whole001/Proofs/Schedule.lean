import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001.Schedule
noncomputable section
/-- A finite schedule executes against the evolving memory. -/
def run {P A V : Type} (execute : P → (A → V) → (A → V))
    (schedule : List P) (memory : A → V) : A → V :=
  schedule.foldl (fun current program => execute program current) memory

/-- Generic induction principle for source kernels. The premise is a local
read footprint, not a final-value or execution-success condition. -/
theorem freeze_reads {P A V : Type} [DecidableEq P]
    (execute fixed : P → (A → V) → (A → V))
    (reads writes : P → A → Prop) (entry : A → V)
    (readRule : ∀ p memory, (∀ a, reads p a → memory a=entry a) → execute p memory=fixed p memory)
    (frame : ∀ p memory a, ¬writes p a → execute p memory a=memory a)
    (schedule : List P) (nodup : schedule.Nodup)
    (separate : ∀ p∈schedule, ∀ q∈schedule, p≠q → ∀ a, reads q a → ¬writes p a)
    (memory : A → V) (same : ∀ p∈schedule, ∀ a, reads p a → memory a=entry a) :
    run execute schedule memory=run fixed schedule memory := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons p tail ih =>
    have head := readRule p memory (same p (by simp))
    have tailNodup := nodup.of_cons
    have sepTail : ∀ q∈tail, ∀ r∈tail, q≠r → ∀ a, reads r a → ¬writes q a := by
      intro q hq r hr hqr a ha
      exact separate q (by simp [hq]) r (by simp [hr]) hqr a ha
    have sameTail : ∀ q∈tail, ∀ a, reads q a → execute p memory a=entry a := by
      intro q hq a ha
      have different : p≠q := by
        intro he
        subst q
        exact (List.nodup_cons.mp nodup).1 hq
      rw [frame p memory a (separate p (by simp) q (by simp [hq]) different a ha)]
      exact same q (by simp [hq]) a ha
    change run execute tail (execute p memory)=run fixed tail (fixed p memory)
    rw [ih tailNodup sepTail (execute p memory) sameTail,head]

def put {P A V : Type} (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (p : P) (memory : A → V) : A → V :=
  fun a => if writes p a then values p a else memory a

theorem fixed_frame {P A V : Type} (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (schedule : List P) (memory : A → V) (a : A)
    (outside : ∀ p∈schedule, ¬writes p a) :
    run (put writes values) schedule memory a=memory a := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons p tail ih =>
    change run (put writes values) tail (put writes values p memory) a=memory a
    rw [ih _ (by intro q hq; exact outside q (by simp [hq]))]
    exact if_neg (outside p (by simp))

theorem fixed_output {P A V : Type} [DecidableEq P]
    (writes : P → A → Prop) [∀ p, DecidablePred (writes p)]
    (values : P → A → V) (schedule : List P) (memory : A → V) (p : P) (a : A)
    (active : writes p a) (unique : ∀ q∈schedule, writes q a → q=p) :
    run (put writes values) schedule memory a=
      if p∈schedule then values p a else memory a := by
  induction schedule generalizing memory with
  | nil => simp [run]
  | cons q tail ih =>
    change run (put writes values) tail (put writes values q memory) a=_
    rw [ih _ (by intro r hr hw; exact unique r (by simp [hr]) hw)]
    by_cases present : p∈tail
    · simp [present]
    · by_cases same : p=q
      · subst q
        simp [present,put,active]
      · have miss : ¬writes q a := by
          intro hw
          exact same (unique q (by simp) hw).symm
        simp [present,same,put,miss]

#axiomsClean freeze_reads
#axiomsClean fixed_frame
#axiomsClean fixed_output
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001.Schedule

import bench.tritonbench_g.lora_expand_gemv.IO

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOGrid
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 6000000

/-- Exactly the finite source inputs, with no restriction on unrelated cells. -/
structure Reads (g : Geometry) (raw : Nat → Int) (mem : Memory) : Prop where
  index : ∀ b,b<g.B → (mem "I" b).read .int=logicalIndex g raw b
  x : ∀ b k,b<g.B → k<g.K → mem "X" (xoff g b k)=entry g raw "X" (xoff g b k)
  w : ∀ l n k,l<g.L → n<g.N → k<g.K →
    mem "W" (woff g l n k)=entry g raw "W" (woff g l n k)

theorem reads_put (g : Geometry) (raw : Nat → Int) (mem : Memory)
    (h : Reads g raw mem) (o : Nat) (c : Cell Word) : Reads g raw (put mem o c) := by
  constructor
  · simpa [put] using h.index
  · simpa [put] using h.x
  · simpa [put] using h.w

def Snapshot (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (initial : Memory) (seen : List Nat) (mem : Memory) : Prop :=
  Reads g raw mem ∧
  (∀ q,q<g.count → mem "Y" (offset g q)=
    if q∈seen then cell plans g raw q else entry g raw "Y" (offset g q)) ∧
  (∀ r o,(r≠("Y" : RegionName) ∨ ∀ q,q<g.count → o≠offset g q) → mem r o=initial r o)

theorem snapshot_put (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (initial : Memory) (seen : List Nat) (mem : Memory)
    (h : Snapshot plans g raw initial seen mem) (q : Nat) (hq : q<g.count) :
    Snapshot plans g raw initial (q::seen) (put mem (offset g q) (cell plans g raw q)) := by
  refine ⟨reads_put g raw mem h.1 _ _,?_,?_⟩
  · intro p hp
    by_cases he : p=q
    · subst p; simp [put]
    · have ho : offset g p≠offset g q := fun eq => he (offset_injective g raw v p q hp hq eq)
      simp [put,ho,he,h.2.1 p hp]
  · intro r o ho
    have hn : ¬(r="Y" ∧ o=offset g q) := by
      intro he
      rcases ho with hr|ha
      · exact hr he.1
      · exact ha q hq he.2
    simpa [put,hn] using h.2.2 r o ho

theorem reference_initial (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (q : Nat) (s : State Word) (fuel n : Nat) (mem : Memory) :
    Branches.reference .ieeeValue plans .singleHalf .single g.parameters
      ((launch rest plans g q).initial s) fuel n mem =
    Branches.reference .ieeeValue plans .singleHalf .single g.parameters
      (program g (q/g.N) (q%g.N) s.mem) fuel n mem := by
  induction fuel generalizing n mem with
  | zero => rfl
  | succ fuel ih =>
    simp only [Branches.reference]
    split
    · rw [show Branches.writeBlock .ieeeValue plans .singleHalf .single g.parameters
        ((launch rest plans g q).initial s) mem n =
        Branches.writeBlock .ieeeValue plans .singleHalf .single g.parameters
        (program g (q/g.N) (q%g.N) s.mem) mem n from rfl]
      exact ih _ _
    · rfl

theorem one_program (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (initial : Memory) (seen : List Nat) (s : State Word)
    (h : Snapshot plans g raw initial seen s.mem) (q : Nat) (hq : q<g.count) (new : q∉seen) :
    ∃ next,(launch rest plans g q).run s=some next ∧
      next.mem=put s.mem (offset g q) (cell plans g raw q) := by
  have hc := coordinates g q hq
  have hy : s.mem "Y" (yoff g (q/g.N) (q%g.N))=entry g raw "Y" (yoff g (q/g.N) (q%g.N)) :=
    by simpa [offset,new] using h.2.1 q hq
  obtain ⟨next,he,hm,_⟩ := Branches.correctness rest .ieeeValue plans
    g.parameters ((launch rest plans g q).initial s) (by change 0<256; decide)
  refine ⟨next,he,?_⟩
  have hm0 : Branches.expectedMemory .ieeeValue plans .singleHalf .single g.parameters
      ((launch rest plans g q).initial s)=
      Branches.expectedMemory .ieeeValue plans .singleHalf .single g.parameters
      (program g (q/g.N) (q%g.N) s.mem) := by
    unfold Branches.expectedMemory
    rw [reference_initial]
    rfl
  rw [hm0] at hm
  rw [hm,write_single plans g _ _ s.mem hc.1,h.1.index _ hc.2]
  by_cases ha : logicalIndex g raw (q/g.N)=-1
  · simp only [ha,ite_true,cell,resultCell]
    funext r a
    by_cases ho : r="Y" ∧ a=offset g q
    · rcases ho with ⟨rfl,rfl⟩
      simpa [put,offset] using hy
    · simp [put,ho]
  · simp only [ha,ite_false]
    have hiBounds := (v.ids _ hc.2).resolve_left ha
    have hl : (logicalIndex g raw (q/g.N)).toNat<g.L :=
      Int.toNat_lt hiBounds.1 |>.2 hiBounds.2
    rw [source_value_of_reads plans g raw v _ _ hc.2 hc.1 s.mem
      (h.1.index _ hc.2) (fun k hk => h.1.x _ k hc.2 hk)
      (fun k hk => h.1.w _ _ k hl hc.1 hk) hy ha]
    simp [offset,cell,resultCell,ha]

theorem execute_prefix (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (initial : Memory) (todo seen : List Nat) (s : State Word)
    (h : Snapshot plans g raw initial seen s.mem)
    (bounded : ∀ q∈todo,q<g.count) (distinct : todo.Nodup)
    (disjoint : ∀ q∈todo,q∉seen) :
    ∃ final,runSchedule (todo.map (launch rest plans g)) s=some final ∧
      Snapshot plans g raw initial (todo.reverse++seen) final.mem := by
  induction todo generalizing seen s with
  | nil => exact ⟨s,rfl,by simpa using h⟩
  | cons q qs ih =>
    have hq := bounded q (by simp)
    obtain ⟨next,he,hm⟩ := one_program rest plans g raw v initial seen s h q hq (disjoint q (by simp))
    have hn : Snapshot plans g raw initial (q::seen) next.mem := by
      rw [hm]; exact snapshot_put plans g raw v initial seen s.mem h q hq
    have hdistinct := List.nodup_cons.mp distinct
    obtain ⟨final,hr,hf⟩ := ih (q::seen) next hn
      (fun p hp => bounded p (by simp [hp])) hdistinct.2 (by
        intro p hp
        have hpq : p≠q := by intro eq; exact hdistinct.1 (eq ▸ hp)
        simp [hpq,disjoint p (by simp [hp])])
    refine ⟨final,?_,?_⟩
    · simp [runSchedule,he,hr]
    · simpa [List.reverse_cons,List.append_assoc] using hf

#axiomsClean one_program
#axiomsClean execute_prefix
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOGrid

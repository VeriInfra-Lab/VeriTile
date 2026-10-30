import bench.repairs.tritonbench_g.apply_penalty.Proofs.Schedule
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Contract
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Scatter

namespace VeriTile.Bench.Repairs.ApplyPenalty
noncomputable section
set_option maxHeartbeats 1000000

def address (a : Arguments) (b lane : Nat) : Nat := b*a.vocab+a.token (a.start b+lane)
def value {Word : Type} (ops : ScalarOps Word) (a : Arguments) (input : Inputs Word)
    (memory : Nat → Word) (b lane : Nat) : Word :=
  (arithmetic ops (memory (address a b lane)) (input.presence b)
    (input.frequency b) (input.repetition b) (a.counts (a.start b+lane))).result

theorem event_iff (a : Arguments) (b lane : Nat) :
    lane ∈ rowEvents a b ↔ lane < a.block ∧ a.start b+lane < a.stop b := by
  simp [rowEvents]

theorem cover (a : Arguments) (h : InputContract a) (b k : Nat)
    (hb : b < a.batches) (hk : a.start b ≤ k) (he : k < a.stop b) :
    k-a.start b ∈ rowEvents a b ∧ a.start b+(k-a.start b)=k := by
  have tile := (h.windows b hb).2.2.2.1
  exact ⟨(event_iff _ _ _).mpr ⟨by omega,by omega⟩,by omega⟩

theorem address_unique (a : Arguments) (h : InputContract a) (b c i j : Nat)
    (hb : b < a.batches) (hc : c < a.batches)
    (hi : i ∈ rowEvents a b) (hj : j ∈ rowEvents a c)
    (same : address a b i = address a c j) : b=c ∧ i=j := by
  have ai := (event_iff a b i).mp hi
  have aj := (event_iff a c j).mp hj
  have ti := ((h.windows b hb).2.2.2.2.1 _ (by omega) ai.2).2.1
  have tj := ((h.windows c hc).2.2.2.2.1 _ (by omega) aj.2).2.1
  have eq := coordinate_index a.vocab b c _ _ ti tj same
  refine ⟨eq.1,?_⟩
  rcases eq with ⟨eq,keys⟩
  subst c
  have keys := (h.windows b hb).2.2.2.2.2 _ _ (by omega) ai.2 (by omega) aj.2 keys
  omega

theorem row_as_scatter {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (memory : Nat → Word) (b : Nat) :
    rowRun ops a input memory b =
      Scatter.run (rowEvents a b) (address a b)
        (fun lane => (rowLanes ops a input memory b lane).result) memory := rfl

theorem row_selected {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (h : InputContract a) (input : Inputs Word) (memory : Nat → Word) (b lane : Nat)
    (hb : b < a.batches) (hl : lane ∈ rowEvents a b) :
    rowRun ops a input memory b (address a b lane)=value ops a input memory b lane := by
  rw [row_as_scatter]
  rw [Scatter.readback_on _ _ _ _ lane hl]
  · have active := ((event_iff a b lane).mp hl).2
    simp [value,rowLanes,address,active]
  · intro j hj same
    rw [(address_unique a h b b j lane hb hb hj hl same).2]

theorem row_frame {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (memory : Nat → Word) (b q : Nat)
    (miss : ∀ lane ∈ rowEvents a b,q ≠ address a b lane) :
    rowRun ops a input memory b q=memory q := by
  rw [row_as_scatter]
  exact Scatter.frame _ _ _ _ _ miss

def Values {Word : Type} (ops : ScalarOps Word) (a : Arguments) (input : Inputs Word)
    (order : List Nat) (initial final : Nat → Word) : Prop :=
  ∀ b ∈ order, ∀ lane ∈ rowEvents a b,
    final (address a b lane)=value ops a input initial b lane

def Frame {Word : Type} (a : Arguments) (order : List Nat)
    (initial final : Nat → Word) : Prop :=
  ∀ q,(∀ b ∈ order,∀ lane ∈ rowEvents a b,q ≠ address a b lane) → final q=initial q

/-- Whole programs read memory produced by earlier programs. Disjoint rows
derive their entry operands; no snapshot-result equality is an assumption. -/
theorem runRows_correct {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (h : InputContract a) (input : Inputs Word) (order : List Nat)
    (unique : order.Nodup) (bounds : ∀ b ∈ order,b < a.batches) (memory : Nat → Word) :
    Values ops a input order memory (runRows ops a input order memory) ∧
    Frame a order memory (runRows ops a input order memory) := by
  induction order generalizing memory with
  | nil => exact ⟨by simp [Values],by intro q _; rfl⟩
  | cons head tail ih =>
    have hn := List.nodup_cons.mp unique
    have bh := bounds head (by simp)
    have bt : ∀ b ∈ tail,b < a.batches := fun b hb => bounds b (by simp [hb])
    let mid := rowRun ops a input memory head
    obtain ⟨hv,hf⟩ := ih hn.2 bt mid
    constructor
    · intro b hb lane hl
      rcases List.mem_cons.mp hb with same | member
      · subst b
        have keep := hf (address a head lane) (by
          intro c hc j hj eq
          exact hn.1 ((address_unique a h head c lane j bh (bt c hc) hl hj eq).1 ▸ hc))
        exact keep.trans (row_selected ops a h input memory head lane bh hl)
      · have keep : mid (address a b lane)=memory (address a b lane) := by
          apply row_frame
          intro j hj eq
          exact hn.1 ((address_unique a h b head lane j (bt b member) bh hl hj eq).1 ▸ member)
        have result := hv b member lane hl
        simpa only [value,keep,runRows] using result
    · intro q miss
      have tailMiss : ∀ b ∈ tail,∀ lane ∈ rowEvents a b,q ≠ address a b lane :=
        fun b hb lane hl => miss b (by simp [hb]) lane hl
      exact (hf q tailMiss).trans (row_frame ops a input memory head q
        (fun lane hl => miss head (by simp) lane hl))

/-- Arbitrary complete program order; duplicate tokens were rejected by the
same contract that drives Python admission. Every selected and unselected
logical word is covered, including empty rows and zero-batch calls. -/
theorem complete_grid {Word : Type} (ops : ScalarOps Word) (a : Arguments)
    (h : InputContract a) (input : Inputs Word) (order : List Nat)
    (schedule : order.Perm (List.range a.batches)) :
    (∀ b,b < a.batches → ∀ k,a.start b ≤ k → k < a.stop b →
      gridRun ops a input order (b*a.vocab+a.token k)=
        (arithmetic ops (input.logits (b*a.vocab+a.token k)) (input.presence b)
          (input.frequency b) (input.repetition b) (a.counts k)).result) ∧
    (∀ q,(∀ b,b < a.batches → ∀ k,a.start b ≤ k → k < a.stop b →
      q ≠ b*a.vocab+a.token k) → gridRun ops a input order q=input.logits q) := by
  have rows := runRows_correct ops a h input order (schedule.nodup_iff.mpr List.nodup_range)
    (fun b hb => List.mem_range.mp (schedule.mem_iff.mp hb)) input.logits
  constructor
  · intro b hb k hk he
    obtain ⟨hl,eq⟩ := cover a h b k hb hk he
    have result := rows.1 b (schedule.mem_iff.mpr (List.mem_range.mpr hb)) _ hl
    simpa [value,address,eq,gridRun] using result
  · intro q miss
    apply rows.2 q
    intro b hb lane hl
    exact miss b (List.mem_range.mp (schedule.mem_iff.mp hb)) _ (by omega)
      ((event_iff _ _ _).mp hl).2

#axiomsClean address_unique
#axiomsClean row_as_scatter
#axiomsClean runRows_correct
#axiomsClean complete_grid
end
end VeriTile.Bench.Repairs.ApplyPenalty

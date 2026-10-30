import bench.repairs.tritonbench_g.attention_llama.Proofs.HostMemory

namespace VeriTile.Bench.Repairs.AttentionLlama.HostGrid
open MaskCoverage HostMemory

@[ext] structure Index (B heads H D : Nat) where
  batch : Fin B
  head : Fin heads
  query : Fin H
  channel : Fin D
  deriving DecidableEq, Fintype

@[ext] structure Event (B heads H D : Nat) where
  pid : Fin (count H)
  headId : Fin (B * heads)
  lane : Fin 64
  channel : Fin D
  deriving DecidableEq, Fintype

def index? {B heads H D : Nat} (event : Event B heads H D) : Option (Index B heads H D) :=
  if hh : 0 < heads then
    if hq : event.pid.val * 64 + event.lane.val < H then
      some ⟨⟨event.headId.val / heads,
          Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using event.headId.isLt)⟩,
        ⟨event.headId.val % heads, Nat.mod_lt _ hh⟩,
        ⟨event.pid.val * 64 + event.lane.val, hq⟩, event.channel⟩
    else none
  else none

def eventFor {B heads H D : Nat} (i : Index B heads H D) : Event B heads H D :=
  ⟨⟨i.query.val / 64, (query_program_exists H i.query.val i.query.isLt).1⟩,
    ⟨i.batch.val * heads + i.head.val, by nlinarith [i.batch.isLt, i.head.isLt]⟩,
    ⟨i.query.val % 64, Nat.mod_lt _ (by decide)⟩, i.channel⟩

theorem index_eventFor {B heads H D : Nat} (i : Index B heads H D) :
    index? (eventFor i) = some i := by
  have hh : 0 < heads := Nat.zero_lt_of_lt i.head.isLt
  have hq := (query_program_exists H i.query.val i.query.isLt).2.2
  have hbh := batch_head_roundtrip heads i.batch.val i.head.val i.head.isLt
  simp only [index?, eventFor, hh, ↓reduceDIte, hq, i.query.isLt]
  congr 1
  apply Index.ext
  · exact Fin.ext hbh.1
  · exact Fin.ext hbh.2
  · rfl
  · rfl

theorem eventFor_index {B heads H D : Nat} (event : Event B heads H D)
    (i : Index B heads H D) (hi : index? event = some i) : eventFor i = event := by
  unfold index? at hi
  split at hi
  · split at hi
    · cases Option.some.inj hi
      apply Event.ext
      · apply Fin.ext
        simp only [eventFor]
        exact (query_program_unique (event.pid.val * 64 + event.lane.val)
          event.pid.val event.lane.val event.lane.isLt rfl).1.symm
      · apply Fin.ext
        simp only [eventFor]
        simpa [Nat.mul_comm] using Nat.div_add_mod event.headId.val heads
      · apply Fin.ext
        simp only [eventFor]
        exact (query_program_unique (event.pid.val * 64 + event.lane.val)
          event.pid.val event.lane.val event.lane.isLt rfl).2.symm
      · rfl
    · contradiction
  · contradiction

theorem unique_event {B heads H D : Nat} (a b : Event B heads H D)
    (i : Index B heads H D) (ha : index? a = some i) (hb : index? b = some i) : a = b :=
  (eventFor_index a i ha).symm.trans (eventFor_index b i hb)

/-- Any native program order is permitted by the fresh-output frame. This
enumeration includes every physical grid/lane event, then applies the source
query mask. It has no separate logical-coverage premise. -/
noncomputable def schedule (B heads H D : Nat) : List (Index B heads H D) :=
  (Finset.univ.toList : List (Event B heads H D)).filterMap index?

theorem schedule_covers {B heads H D : Nat} (i : Index B heads H D) :
    i ∈ schedule B heads H D := by
  apply List.mem_filterMap.mpr
  refine ⟨eventFor i, ?_, index_eventFor i⟩
  simp

theorem logical_result_and_frame {B heads H D : Nat} {α : Type}
    (address : Index B heads H D → Nat) (word : Index B heads H D → α)
    (hinj : Function.Injective address) (memory : Nat → α) :
    (∀ i, (returnTensor address
      (storeAll address word memory (schedule B heads H D))).read i = word i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address word memory (schedule B heads H D) location = memory location) :=
  returned_values_and_frame address word hinj _ schedule_covers memory

end VeriTile.Bench.Repairs.AttentionLlama.HostGrid

import bench.repairs.tritonbench_g.attention_llama.Proofs.SourceGraph

/-! Scalar-store composition for a fresh output allocation with arbitrary
injective logical strides. Readback and the complement frame follow from the
actual stores, rather than being supplied as a host result premise. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.HostMemory

def storeAll {ι α : Type} [DecidableEq ι] (address : ι → Nat) (word : ι → α)
    (memory : Nat → α) (schedule : List ι) : Nat → α :=
  schedule.foldl (fun mem i => Function.update mem (address i) (word i)) memory

theorem readback {ι α : Type} [DecidableEq ι] (address : ι → Nat) (word : ι → α)
    (hinj : Function.Injective address) (schedule : List ι) (memory : Nat → α) (i : ι) :
    storeAll address word memory schedule (address i) =
      if i ∈ schedule then word i else memory (address i) := by
  induction schedule generalizing memory with
  | nil => simp [storeAll]
  | cons j rest ih =>
    simp only [storeAll, List.foldl_cons]
    change storeAll address word (Function.update memory (address j) (word j)) rest
      (address i) = _
    rw [ih]
    by_cases hr : i ∈ rest
    · simp [hr]
    · by_cases he : i = j
      · subst j
        simp [hr]
      · have hn : address i ≠ address j := fun h => he (hinj h)
        simp [hr, he, hn]

theorem frame {ι α : Type} [DecidableEq ι] (address : ι → Nat) (word : ι → α)
    (schedule : List ι) (memory : Nat → α) (location : Nat)
    (houtside : ∀ i ∈ schedule, location ≠ address i) :
    storeAll address word memory schedule location = memory location := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons j rest ih =>
    simp only [storeAll, List.foldl_cons]
    change storeAll address word (Function.update memory (address j) (word j)) rest location = _
    rw [ih _ (fun i hi => houtside i (List.mem_cons_of_mem j hi))]
    exact Function.update_of_ne (houtside j List.mem_cons_self) ..

/-- The ordinary tensor return contract supplies a layout and reads the
resulting memory. It does not supply any attention value theorem. -/
structure ReturnedTensor (ι α : Type) where
  storageOffset : ι → Nat
  read : ι → α

def returnTensor {ι α : Type} (address : ι → Nat) (memory : Nat → α) : ReturnedTensor ι α :=
  ⟨address, fun i => memory (address i)⟩

theorem returned_values_and_frame {ι α : Type} [DecidableEq ι]
    (address : ι → Nat) (word : ι → α) (hinj : Function.Injective address)
    (schedule : List ι) (hcovers : ∀ i, i ∈ schedule) (memory : Nat → α) :
    (∀ i, (returnTensor address (storeAll address word memory schedule)).read i = word i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address word memory schedule location = memory location) := by
  constructor
  · intro i
    exact (readback address word hinj schedule memory i).trans (if_pos (hcovers i))
  · intro location hl
    exact frame address word schedule memory location (fun i _ => hl i)

end VeriTile.Bench.Repairs.AttentionLlama.HostMemory

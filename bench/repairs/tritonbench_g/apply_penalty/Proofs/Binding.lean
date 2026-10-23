import bench.repairs.tritonbench_g.apply_penalty.Proofs.Memory

/-! Direct input reads at the seven actual data pointers. Only bounded cells
are observed. The source's read-only arguments remain immutable by a derived
byte frame, even though read-only arguments may alias one another. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.Physical
noncomputable section
set_option maxHeartbeats 1000000

def observe {Value Byte : Type} (codec : Codec Value Byte) (a : Arguments)
    (heap : Nat → Byte) (role : Fin 7) (fallback : Nat → Value) (k : Nat) : Value :=
  if k<a.capacity role then read codec (a.tensor role).dataPtr.toNat heap k else fallback k

def observedArguments {Byte : Type} (codec : Codec (BitVec 32) Byte)
    (a : Arguments) (heap : Nat → Byte) : Arguments :=
  { a with ids := observe codec a heap 4 a.ids
           counts := observe codec a heap 5 a.counts
           cumsum := observe codec a heap 6 a.cumsum }

def observedInputs {Word Byte : Type} (codec : Codec Word Byte)
    (a : Arguments) (input : Inputs Word) (heap : Nat → Byte) : Inputs Word :=
  { input with presence := observe codec a heap 1 input.presence
               frequency := observe codec a heap 2 input.frequency
               repetition := observe codec a heap 3 input.repetition }

structure ReadOnlyInputs {Word Byte : Type} (fpCodec : Codec Word Byte)
    (integer : Codec (BitVec 32) Byte) (a : Arguments) (input : Inputs Word) (heap : Nat → Byte) : Prop where
  presence : ∀ k,k<a.capacity 1 → read fpCodec (a.tensor 1).dataPtr.toNat heap k=input.presence k
  frequency : ∀ k,k<a.capacity 2 → read fpCodec (a.tensor 2).dataPtr.toNat heap k=input.frequency k
  repetition : ∀ k,k<a.capacity 3 → read fpCodec (a.tensor 3).dataPtr.toNat heap k=input.repetition k
  ids : ∀ k,k<a.capacity 4 → read integer (a.tensor 4).dataPtr.toNat heap k=a.ids k
  counts : ∀ k,k<a.capacity 5 → read integer (a.tensor 5).dataPtr.toNat heap k=a.counts k
  cumsum : ∀ k,k<a.capacity 6 → read integer (a.tensor 6).dataPtr.toNat heap k=a.cumsum k

theorem observed_arguments_eq {Word Byte : Type} (fpCodec : Codec Word Byte)
    (integer : Codec (BitVec 32) Byte) (a : Arguments) (input : Inputs Word) (heap : Nat → Byte)
    (binding : ReadOnlyInputs fpCodec integer a input heap) : observedArguments integer a heap=a := by
  have ids : observe integer a heap 4 a.ids=a.ids := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.ids k h
    · rfl
  have counts : observe integer a heap 5 a.counts=a.counts := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.counts k h
    · rfl
  have cumsum : observe integer a heap 6 a.cumsum=a.cumsum := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.cumsum k h
    · rfl
  simp [observedArguments,ids,counts,cumsum]

theorem observed_inputs_eq {Word Byte : Type} (fpCodec : Codec Word Byte)
    (integer : Codec (BitVec 32) Byte) (a : Arguments) (input : Inputs Word) (heap : Nat → Byte)
    (binding : ReadOnlyInputs fpCodec integer a input heap) : observedInputs fpCodec a input heap=input := by
  have p : observe fpCodec a heap 1 input.presence=input.presence := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.presence k h
    · rfl
  have f : observe fpCodec a heap 2 input.frequency=input.frequency := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.frequency k h
    · rfl
  have r : observe fpCodec a heap 3 input.repetition=input.repetition := by
    funext k; simp only [observe]; split <;> rename_i h
    · exact binding.repetition k h
    · rfl
  simp [observedInputs,p,f,r]

theorem read_metadata_frame {Word Value Byte : Type} (fpCodec : Codec Word Byte)
    (codec : Codec Value Byte) (ops : ScalarOps Word) (a : Arguments) (h : InputContract a)
    (input : Inputs Word) (heap : Nat → Byte) (order : List Nat)
    (bounds : ∀ b ∈ order,b<a.batches) (r : Fin 7) (hr : 0<r.val) (k : Nat) (hk : k<a.capacity r) :
    read codec (a.tensor r).dataPtr.toNat (rows fpCodec ops a input order heap) k=
      read codec (a.tensor r).dataPtr.toNat heap k := by
  unfold read
  apply congrArg codec.decode
  funext byte
  exact metadata_frame_order fpCodec ops a h input heap order bounds r hr k hk byte

theorem binding_preserved {Word Byte : Type} (fpCodec : Codec Word Byte)
    (integer : Codec (BitVec 32) Byte) (ops : ScalarOps Word) (a : Arguments) (h : InputContract a)
    (input : Inputs Word) (heap : Nat → Byte) (order : List Nat)
    (bounds : ∀ b ∈ order,b<a.batches) (binding : ReadOnlyInputs fpCodec integer a input heap) :
    ReadOnlyInputs fpCodec integer a input (rows fpCodec ops a input order heap) := by
  constructor
  · intro k hk; rw [read_metadata_frame fpCodec fpCodec ops a h input heap order bounds 1 (by decide) k hk]; exact binding.presence k hk
  · intro k hk; rw [read_metadata_frame fpCodec fpCodec ops a h input heap order bounds 2 (by decide) k hk]; exact binding.frequency k hk
  · intro k hk; rw [read_metadata_frame fpCodec fpCodec ops a h input heap order bounds 3 (by decide) k hk]; exact binding.repetition k hk
  · intro k hk; rw [read_metadata_frame fpCodec integer ops a h input heap order bounds 4 (by decide) k hk]; exact binding.ids k hk
  · intro k hk; rw [read_metadata_frame fpCodec integer ops a h input heap order bounds 5 (by decide) k hk]; exact binding.counts k hk
  · intro k hk; rw [read_metadata_frame fpCodec integer ops a h input heap order bounds 6 (by decide) k hk]; exact binding.cumsum k hk

/-- Every block observes its metadata from its actual incoming byte heap. -/
def nativeRows {Word Byte : Type} (fpCodec : Codec Word Byte) (integer : Codec (BitVec 32) Byte)
    (ops : ScalarOps Word) (a : Arguments) (input : Inputs Word) : List Nat → (Nat → Byte) → (Nat → Byte)
  | [],heap => heap
  | b::rest,heap => nativeRows fpCodec integer ops a input rest
      (row fpCodec ops (observedArguments integer a heap) (observedInputs fpCodec a input heap) heap b)

theorem native_rows_eq {Word Byte : Type} (fpCodec : Codec Word Byte)
    (integer : Codec (BitVec 32) Byte) (ops : ScalarOps Word) (a : Arguments) (h : InputContract a)
    (input : Inputs Word) (order : List Nat) (bounds : ∀ b ∈ order,b<a.batches)
    (heap : Nat → Byte) (binding : ReadOnlyInputs fpCodec integer a input heap) :
    nativeRows fpCodec integer ops a input order heap=rows fpCodec ops a input order heap := by
  induction order generalizing heap with
  | nil => rfl
  | cons b rest ih =>
    rw [nativeRows,observed_arguments_eq fpCodec integer a input heap binding,
      observed_inputs_eq fpCodec integer a input heap binding]
    apply ih (fun c hc => bounds c (by simp [hc]))
    exact binding_preserved fpCodec integer ops a h input heap [b]
      (by intro c hc; have eq := List.mem_singleton.mp hc; subst c; exact bounds b (by simp)) binding

#axiomsClean observed_arguments_eq
#axiomsClean observed_inputs_eq
#axiomsClean binding_preserved
#axiomsClean native_rows_eq
end
end VeriTile.Bench.Repairs.ApplyPenalty.Physical

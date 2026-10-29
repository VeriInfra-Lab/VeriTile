import bench.repairs.tritonbench_g.apply_penalty.Proofs.Execution

/-! A direct physical byte-store interpretation. Every lane reads the current
incoming heap; there is no staging allocation or final copyback. Codec laws
are local FP32 packing laws, containing no kernel or whole-result equation. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.Physical
noncomputable section
set_option maxHeartbeats 1000000

structure Codec (Word Byte : Type) where
  encode : Word → Fin 4 → Byte
  decode : (Fin 4 → Byte) → Word
  roundtrip : ∀ w,decode (encode w)=w

def read {Word Byte : Type} (codec : Codec Word Byte) (base : Nat)
    (heap : Nat → Byte) (q : Nat) : Word := codec.decode (fun byte => heap (base+4*q+byte.val))

def write {Word Byte : Type} (codec : Codec Word Byte) (base : Nat)
    (heap : Nat → Byte) (q : Nat) (word : Word) : Nat → Byte :=
  Scatter.run (List.finRange 4) (fun byte => base+4*q+byte.val) (codec.encode word) heap

theorem word_collision (base q r : Nat) (b c : Fin 4)
    (same : base+4*q+b.val=base+4*r+c.val) : q=r ∧ b=c := by
  have hb := b.isLt
  have hc := c.isLt
  exact ⟨by omega,Fin.ext (by omega)⟩

theorem write_read {Word Byte : Type} (codec : Codec Word Byte) (base : Nat)
    (heap : Nat → Byte) (q : Nat) (word : Word) :
    read codec base (write codec base heap q word)=store (read codec base heap) q word := by
  funext r
  by_cases same : r=q
  · subst r
    simp only [store,if_true,read,write]
    rw [show (fun byte : Fin 4 => Scatter.run (List.finRange 4)
      (fun byte => base+4*q+byte.val) (codec.encode word) heap (base+4*q+byte.val))=
      codec.encode word by
        funext byte
        apply Scatter.readback_injective _ _ _ _ byte (by simp)
        intro b c eq; exact (word_collision base q q b c eq).2]
    exact codec.roundtrip word
  · simp only [store,same,if_false,read,write]
    apply congrArg codec.decode
    funext byte
    apply Scatter.frame
    intro b _ eq
    exact same (word_collision base r q byte b eq).1

theorem write_frame {Word Byte : Type} (codec : Codec Word Byte) (base : Nat)
    (heap : Nat → Byte) (q : Nat) (word : Word) (address : Nat)
    (miss : ∀ byte : Fin 4,address ≠ base+4*q+byte.val) :
    write codec base heap q word address=heap address :=
  Scatter.frame _ _ _ _ _ (fun b _ => miss b)

def stores {Word Byte ι : Type} (codec : Codec Word Byte) (base : Nat)
    (events : List ι) (key : ι → Nat) (value : ι → Word) (heap : Nat → Byte) : Nat → Byte :=
  events.foldl (fun memory event => write codec base memory (key event) (value event)) heap

theorem stores_read {Word Byte ι : Type} (codec : Codec Word Byte) (base : Nat)
    (events : List ι) (key : ι → Nat) (value : ι → Word) (heap : Nat → Byte) :
    read codec base (stores codec base events key value heap)=
      Scatter.run events key value (read codec base heap) := by
  induction events generalizing heap with
  | nil => rfl
  | cons e tail ih =>
    simp only [stores,List.foldl_cons] at *
    rw [ih,write_read]
    rfl

theorem stores_frame {Word Byte ι : Type} (codec : Codec Word Byte) (base : Nat)
    (events : List ι) (key : ι → Nat) (value : ι → Word) (heap : Nat → Byte) (q : Nat)
    (miss : ∀ e ∈ events,∀ byte : Fin 4,q ≠ base+4*key e+byte.val) :
    stores codec base events key value heap q=heap q := by
  induction events generalizing heap with
  | nil => rfl
  | cons e tail ih =>
    change stores codec base tail key value (write codec base heap (key e) (value e)) q=heap q
    rw [ih _ (fun e he => miss e (by simp [he]))]
    exact write_frame codec base heap _ _ q (miss e (by simp))

def row {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (heap : Nat → Byte) (b : Nat) : Nat → Byte :=
  stores codec (a.tensor 0).dataPtr.toNat (rowEvents a b) (address a b)
    (fun lane => (rowLanes ops a input (read codec (a.tensor 0).dataPtr.toNat heap) b lane).result) heap

theorem row_read {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (heap : Nat → Byte) (b : Nat) :
    read codec (a.tensor 0).dataPtr.toNat (row codec ops a input heap b)=
      rowRun ops a input (read codec (a.tensor 0).dataPtr.toNat heap) b := by
  rw [row,stores_read,row_as_scatter]

def rows {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) : List Nat → (Nat → Byte) → (Nat → Byte)
  | [],heap => heap
  | b::rest,heap => rows codec ops a input rest (row codec ops a input heap b)

theorem rows_read {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (order : List Nat) (heap : Nat → Byte) :
    read codec (a.tensor 0).dataPtr.toNat (rows codec ops a input order heap)=
      runRows ops a input order (read codec (a.tensor 0).dataPtr.toNat heap) := by
  induction order generalizing heap with
  | nil => rfl
  | cons b rest ih =>
    rw [rows,ih,row_read]
    rfl

theorem rows_frame {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word) (a : Arguments)
    (input : Inputs Word) (order : List Nat) (heap : Nat → Byte) (q : Nat)
    (miss : ∀ b ∈ order,∀ lane ∈ rowEvents a b,∀ byte : Fin 4,
      q ≠ (a.tensor 0).dataPtr.toNat+4*address a b lane+byte.val) :
    rows codec ops a input order heap q=heap q := by
  induction order generalizing heap with
  | nil => rfl
  | cons b rest ih =>
    rw [rows,ih _ (fun b hb => miss b (by simp [hb]))]
    exact stores_frame _ _ _ _ _ _ _ (miss b (by simp))

/-- Checked byte-range separation preserves every read-only input role,
including metadata sharing storage with another read-only role. -/
theorem metadata_frame_order {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word)
    (a : Arguments) (h : InputContract a) (input : Inputs Word) (heap : Nat → Byte)
    (order : List Nat) (bounds : ∀ b ∈ order,b<a.batches)
    (r : Fin 7) (hr : 0<r.val) (k : Nat) (hk : k<a.capacity r) (byte : Fin 4) :
    rows codec ops a input order heap ((a.tensor r).dataPtr.toNat+4*k+byte.val)=
      heap ((a.tensor r).dataPtr.toNat+4*k+byte.val) := by
  apply rows_frame
  intro b hb lane hl other eq
  have valid := (event_iff a b lane).mp hl
  have bound := offset_bound a h b (a.start b+lane) (bounds b hb) (by omega) valid.2
  have bb := byte.isLt
  have bo := other.isLt
  have hd := h.disjoint r hr
  change _ = (a.tensor 0).dataPtr.toNat+4*(b*a.vocab+a.token (a.start b+lane))+other.val at eq
  rcases hd with empty | empty | left | right <;> omega

theorem metadata_frame {Word Byte : Type} (codec : Codec Word Byte) (ops : ScalarOps Word)
    (a : Arguments) (h : InputContract a) (input : Inputs Word) (heap : Nat → Byte)
    (r : Fin 7) (hr : 0<r.val) (k : Nat) (hk : k<a.capacity r) (byte : Fin 4) :
    rows codec ops a input (List.range a.batches) heap ((a.tensor r).dataPtr.toNat+4*k+byte.val)=
      heap ((a.tensor r).dataPtr.toNat+4*k+byte.val) :=
  metadata_frame_order codec ops a h input heap _ (fun _ hb => List.mem_range.mp hb) r hr k hk byte

/-- Successful backed Tensor views supply this allocation range law; it is
not a new wrapper guard or a numerical/whole-output hypothesis. -/
def Backed (a : Arguments) : Prop :=
  ∀ r : Fin 7,0<a.capacity r → (a.tensor r).dataPtr.toNat+4*a.capacity r ≤ 2^64

theorem active_pointer_bounds (a : Arguments) (h : InputContract a) (backed : Backed a)
    (b lane : Nat) (hb : b<a.batches) (hl : lane ∈ rowEvents a b) (byte : Fin 4) :
    (a.tensor 0).dataPtr.toNat+4*address a b lane+byte.val < 2^64 := by
  have active := (event_iff a b lane).mp hl
  have bound := offset_bound a h b (a.start b+lane) hb (by omega) active.2
  have allocation := backed 0 (by omega)
  have small := byte.isLt
  unfold address
  omega

#axiomsClean write_read
#axiomsClean rows_read
#axiomsClean rows_frame
#axiomsClean metadata_frame
#axiomsClean active_pointer_bounds
end
end VeriTile.Bench.Repairs.ApplyPenalty.Physical

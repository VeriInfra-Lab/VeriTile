import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Tensor
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def count {phase} (r : Request phase) (j : Fin (outputs phase)) : Nat :=
  r.dimensions.B*r.dimensions.H*r.dimensions.N*resultWidth r j

structure Lane {phase} (r : Request phase) where
  output : Fin (outputs phase)
  element : Fin (count r output)
  byte : Fin 4

def address {phase} (r : Request phase) (i : Lane r) : Nat :=
  (r.result i.output).tensor.dataPtr.toNat+4*i.element.val+i.byte.val

theorem lane_span {phase} (r : Request phase) (h : Contract r) (i : Lane r) :
    (r.result i.output).tensor.storage.base≤address r i ∧
    address r i<(r.result i.output).tensor.storage.base+(r.result i.output).storageBytes := by
  have hi : i.element.val<(r.result i.output).tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ (h.result i.output)]
    exact i.element.isLt
  exact payload_span _ (h.result i.output).1 ((h.result i.output).2.2.2.1 rfl) _ hi i.byte

theorem address_unique {phase} (r : Request phase) (h : Contract r) (i j : Lane r)
    (he : address r i=address r j) : i=j := by
  have si := lane_span r h i
  have sj := lane_span r h j
  have same : i.output=j.output := by
    by_contra ne
    have sep := h.outputOutput i.output j.output ne
    unfold separate at sep
    rcases sep with sep|sep <;> omega
  cases i with
  | mk ri ei bi =>
    cases j with
    | mk rj ej bj =>
      dsimp only at same
      subst rj
      change (r.result ri).tensor.dataPtr.toNat+4*ei.val+bi.val =
        (r.result ri).tensor.dataPtr.toNat+4*ej.val+bj.val at he
      have eiEq : ei=ej := Fin.ext (by have := bi.isLt; have := bj.isLt; omega)
      subst ej
      have biEq : bi=bj := Fin.ext (by omega)
      subst bj
      rfl

def writeback {phase} {α : Type} (r : Request phase) (values : Lane r → α)
    (before : Nat → α) (a : Nat) : α :=
  if h : ∃ i : Lane r,address r i=a then values (Classical.choose h) else before a

theorem writeback_readback {phase} {α : Type} (r : Request phase) (h : Contract r)
    (values : Lane r → α) (before : Nat → α) (i : Lane r) :
    writeback r values before (address r i)=values i := by
  have ex : ∃ j : Lane r,address r j=address r i := ⟨i,rfl⟩
  rw [writeback,dif_pos ex]
  rw [address_unique r h (Classical.choose ex) i (Classical.choose_spec ex)]

theorem writeback_frame {phase} {α : Type} (r : Request phase) (values : Lane r → α)
    (before : Nat → α) (a : Nat) (outside : ¬∃ i : Lane r,address r i=a) :
    writeback r values before a=before a := by rw [writeback,dif_neg outside]

/-- Preservation covers every byte in every read allocation, including original
G, offsets before/after a view, unused padding and partial trailing storage bytes. -/
def ReadFrame {phase} {α : Type} (r : Request phase) (before after : Nat → α) : Prop :=
  ∀ x∈readBackings r,∀ off,off<x.storageBytes →
    after (x.tensor.storage.base+off)=before (x.tensor.storage.base+off)

theorem writeback_inputs {phase} {α : Type} (r : Request phase) (h : Contract r)
    (values : Lane r → α) (before : Nat → α) : ReadFrame r before (writeback r values before) := by
  intro x hx off hoff
  apply writeback_frame
  rintro ⟨i,he⟩
  have span := lane_span r h i
  have sep := h.outputRead i.output x hx
  unfold separate at sep
  rcases sep with sep|sep <;> omega

#axiomsClean address_unique
#axiomsClean writeback_readback
#axiomsClean writeback_inputs
#specNonCircular writeback avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly

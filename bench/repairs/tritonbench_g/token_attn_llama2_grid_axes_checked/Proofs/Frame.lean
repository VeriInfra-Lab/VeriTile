import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Contract
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Complete

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural Whole
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def outputAddress (t : TensorCall) (words : Nat → Word) (b h n : Nat) : Nat :=
  t.output.tensor.dataPtr.toNat+4*(h*t.call.outHead+
    ((metadata t words).begin b+n)*t.call.outSeq)

def Exterior (t : TensorCall) (words : Nat → Word) (address : Nat) : Prop :=
  ∀ b<t.call.batches, ∀ h<t.call.heads, ∀ n<(metadata t words).len b,
    address≠outputAddress t words b h n

def Selected (t : TensorCall) (words : Nat → Word) (address : Nat) : Prop :=
  ∃ b<t.call.batches, ∃ h<t.call.heads, ∃ n<(metadata t words).len b,
    address=outputAddress t words b h n

/-- Native observation of exactly the complete grid's source store addresses.
The public theorem also retains the complete raw typed execution and frame. -/
def project {A} (t : TensorCall) (words : Nat → Word) (before : Cells A)
    (raw : State A) : Cells A := fun address =>
  if Selected t words address then
    raw.mem "Out" ((address-t.output.tensor.dataPtr.toNat)/4)
  else before address

theorem project_read {A} (t : TensorCall) (words : Nat → Word) (before : Cells A)
    (raw : State A) (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads)
    (hn : n<(metadata t words).len b) :
    project t words before raw (outputAddress t words b h n)=
      raw.mem "Out" (h*t.call.outHead+((metadata t words).begin b+n)*t.call.outSeq) := by
  have selected : Selected t words (outputAddress t words b h n) := ⟨b,hb,h,hh,n,hn,rfl⟩
  simp only [project,if_pos selected]
  simp [outputAddress]

theorem project_frame {A} (t : TensorCall) (words : Nat → Word) (before : Cells A)
    (raw : State A) (address : Nat) (outside : Exterior t words address) :
    project t words before raw address=before address := by
  have miss : ¬Selected t words address := by
    rintro ⟨b,hb,h,hh,n,hn,same⟩
    exact outside b hb h hh n hn same
  simp [project,miss]

theorem length_le (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (b : Nat) (hb : b<t.call.batches) : (metadata t words).len b≤t.maximum := by
  have bound := (contract.2.2.1.lengths ⟨b,hb⟩).2
  unfold Whole.Metadata.len
  rw [show finiteRead (metadata t words).lengths b=(metadata t words).lengths ⟨b,hb⟩ by
    simp [finiteRead,hb]]
  exact Int.toNat_le.mpr bound

theorem coordinate_active (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (b n : Nat) (hb : b<t.call.batches) (hn : n<(metadata t words).len b) : active t := by
  have bound := length_le t words contract b hb
  exact ⟨by omega,by omega⟩

theorem output_physical (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads) (hn : n<(metadata t words).len b) :
    outputAddress t words b h n=t.output.tensor.storage.address (t.output.tensor.physicalIndex
      (h*t.call.outputTokens+(metadata t words).begin b+n)) := by
  have launch := coordinate_active t words contract b n hb hn
  have domain := valid t words contract launch
  have layout := dense t contract.1 (contract.2.2.2 launch).1
  have bounds := (domain.starts ⟨b,hb⟩).2
  dsimp at bounds
  have size := domain.outputBounds h hh ((metadata t words).begin b+n) (by omega)
  have contiguous : t.output.tensor.metadata.isContiguous=Bool.true :=
    contract.1.contiguous t.output.observation (by simp [TensorCall.observations])
  have equal : h*t.call.outHead+((metadata t words).begin b+n)*t.call.outSeq=
      h*t.call.outputTokens+(metadata t words).begin b+n := by
    rw [layout.out.1,layout.out.2]
    omega
  unfold outputAddress
  rw [equal]
  exact payload_address t.output contiguous _ (by simpa only [←equal] using size)

theorem read_exterior (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) (r : Observation)
    (member : r∈[t.q.observation,t.k.observation,t.locations.observation,t.starts.observation,t.lengths.observation])
    (i : Nat) (hi : i<r.metadata.numel) : Exterior t words (r.metadata.dataPtr.toNat+4*i) := by
  intro b hb h hh n hn same
  have domain := valid t words contract launch
  have bounds := (domain.starts ⟨b,hb⟩).2
  dsimp at bounds
  have out := domain.outputBounds h hh ((metadata t words).begin b+n) (by omega)
  have apart := (contract.2.2.2 launch).2.1.2.2 r member
  change Apart t.output.observation r at apart
  unfold Apart at apart
  change t.output.tensor.metadata.numel=0 ∨ r.metadata.numel=0 ∨
    t.output.tensor.dataPtr.toNat+4*t.output.tensor.metadata.numel≤r.metadata.dataPtr.toNat ∨
    r.metadata.dataPtr.toNat+4*r.metadata.numel≤t.output.tensor.dataPtr.toNat at apart
  change _<t.output.tensor.metadata.numel at out
  unfold outputAddress at same
  rcases apart with empty|empty|left|right <;> omega

#axiomsClean project_read
#axiomsClean project_frame
#axiomsClean coordinate_active
#axiomsClean output_physical
#axiomsClean read_exterior
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativePipeline
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealTensors
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

abbrev Tensor := Torch.FP32.Tensor
abbrev TensorLaws := Torch.FP32.TensorLaws

def tensorRegions : List RegionName := ["Q","K","V","DO","Out","NewDO","DQ","DK","DV"]
def rowRegions : List RegionName := ["L","M","Delta"]
def writeRegions : List RegionName := ["Out","NewDO","DQ","DK","DV","L","M","Delta"]
def count (c : Shape) (r : RegionName) : Nat:=if r∈rowRegions then c.groups*c.rows else c.elements

/-- Ordinary Tensor API observations for the actual input, contiguous incoming
cotangent copy, fresh scratch, and returned objects. These laws contain no
kernel execution or arithmetic-result equality. Input views may alias. -/
structure Objects (c : Shape) where
  get : RegionName → Tensor
  laws : ∀ r,TensorLaws (get r)
  contiguous : ∀ r,(get r).metadata.isContiguous=Bool.true
  resolved : ∀ r,(get r).metadata.isNeg=Bool.false
  tensorShape : ∀ r∈tensorRegions,(get r).metadata.shape=[c.batches,c.heads,c.rows,c.dim]
  rowShape : ∀ r∈rowRegions,(get r).metadata.shape=[c.groups,c.rows]
  device : ∀ r∈tensorRegions++rowRegions,(get r).metadata.device=(get "Q").metadata.device
  cuda : (get "Q").metadata.device.kind="cuda"
  /-- Fresh write allocations are disjoint from every other live Tensor view.
  This is an allocation/address contract, never an output-value assumption. -/
  separate : ∀ r∈writeRegions,∀ s∈tensorRegions++rowRegions,r≠s →
    (Torch.FP32.dataView (get r) (laws r) (contiguous r)).base+4*(get r).metadata.numel≤
      (Torch.FP32.dataView (get s) (laws s) (contiguous s)).base ∨
    (Torch.FP32.dataView (get s) (laws s) (contiguous s)).base+4*(get s).metadata.numel≤
      (Torch.FP32.dataView (get r) (laws r) (contiguous r)).base

def address (c : Shape) (t : Objects c) (r : RegionName) (o : Nat) : Nat:=
  (Torch.FP32.dataView (t.get r) (t.laws r) (t.contiguous r)).address o

theorem numel (c : Shape) (t : Objects c) (r : RegionName) (hr : r∈tensorRegions++rowRegions) :
    (t.get r).metadata.numel=count c r := by
  by_cases hrow : r∈rowRegions
  · simp [Torch.TensorMetadata.numel,t.rowShape r hrow,count,hrow]
  · have ht : r∈tensorRegions:=by simpa [List.mem_append,hrow] using hr
    simp [Torch.TensorMetadata.numel,t.tensorShape r ht,count,hrow,Shape.elements,Shape.groups,Nat.mul_assoc]

theorem written_region (c : Shape) (r : RegionName) (o : Nat) (h : Grid.Written c r o) : r∈writeRegions := by
  rcases h with ⟨hr,_⟩|⟨hr,_⟩ <;> simp_all [writeRegions] <;> tauto

theorem written_bound (c : Shape) (r : RegionName) (o : Nat) (h : Grid.Written c r o) : o<count c r := by
  rcases h with ⟨hr,ho⟩|⟨hr,ho⟩
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl <;> simpa [count,rowRegions] using ho
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> simpa [count,rowRegions] using ho

theorem write_live (r : RegionName) (h : r∈writeRegions) : r∈tensorRegions++rowRegions := by
  simp only [writeRegions,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> simp [tensorRegions,rowRegions]

theorem address_unique (c : Shape) (t : Objects c) (r s : RegionName) (o p : Nat)
    (hr : Grid.Written c r o) (hs : s∈tensorRegions++rowRegions) (hp : p<count c s)
    (he : address c t r o=address c t s p) : r=s ∧ o=p := by
  have hro:=written_bound c r o hr
  have hrw:=written_region c r o hr
  have hn:=numel c t r (write_live r hrw)
  have hsn:=numel c t s hs
  have eqr : r=s := by
    by_contra hne
    have sep:=t.separate r hrw s hs hne
    unfold address Torch.FP32.DataView.address at he
    rw [hn,hsn] at sep
    rcases sep with sep|sep <;> omega
  subst s
  refine ⟨rfl,?_⟩
  unfold address Torch.FP32.DataView.address at he
  omega

/-- A scalar storage-view interpretation of exactly the DSL's written cells.
It uses actual Tensor addresses, preserving all addresses outside those views.
No byte-heap or host implementation is modeled here. -/
def publish {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α) (a : Nat) : α:=by
  classical
  exact if h : ∃ r o,Grid.Written c r o ∧ address c t r o=a then
    (final (Classical.choose h) (Classical.choose (Classical.choose_spec h))).read .fp32
  else before a

theorem publish_readback {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α)
    (r : RegionName) (o : Nat) (hr : Grid.Written c r o) :
    publish c t before final (address c t r o)=(final r o).read .fp32 := by
  have h : ∃ s p,Grid.Written c s p ∧ address c t s p=address c t r o:=⟨r,o,hr,rfl⟩
  rw [publish,dif_pos h]
  have chosen:=Classical.choose_spec (Classical.choose_spec h)
  have same:=address_unique c t _ r _ o chosen.1 (write_live r (written_region c r o hr)) (written_bound c r o hr) chosen.2
  rw [same.2,same.1]

theorem publish_frame {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α)
    (a : Nat) (ha : ¬∃ r o,Grid.Written c r o ∧ address c t r o=a) : publish c t before final a=before a := by
  simp only [publish,dif_neg ha]

theorem publish_input {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α)
    (r : RegionName) (hr : r∈(["Q","K","V","DO"] : List RegionName)) (o : Nat) (ho : o<c.elements) :
    publish c t before final (address c t r o)=before (address c t r o) := by
  apply publish_frame
  rintro ⟨s,p,hw,he⟩
  have live : r∈tensorRegions++rowRegions := by simp only [List.mem_cons,List.not_mem_nil,or_false] at hr;rcases hr with rfl|rfl|rfl|rfl <;> simp [tensorRegions]
  have bound : o<count c r := by simp only [List.mem_cons,List.not_mem_nil,or_false] at hr;rcases hr with rfl|rfl|rfl|rfl <;> simpa [count,rowRegions] using ho
  have same:=(address_unique c t s r p o hw live bound he).1
  subst s
  have wr:=written_region c r p hw
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl <;> simp [writeRegions] at wr

/-- Real erasure keeps actual Tensor storage offsets and logical indices. -/
def logical {α : Type} (read : Nat → α) (t : Tensor) (i : Nat) : α:=
  read (t.storage.address (t.physicalIndex i))

theorem logical_direct {α : Type} (c : Shape) (t : Objects c) (read : Nat → α) (r : RegionName) (o : Nat)
    (ho : o<(t.get r).metadata.numel) : logical read (t.get r) o=read (address c t r o) := by
  unfold logical address Torch.FP32.DataView.address Torch.FP32.DataView.base Torch.FP32.dataView
  rw [if_neg (by omega : (t.get r).metadata.numel≠0),(t.laws r).contiguous (t.contiguous r) o ho]
  congr 1
  simp [Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem actual_pointer (c : Shape) (t : Objects c) (r : RegionName) (hn : 0<(t.get r).metadata.numel) :
    BitVec.ofNat 64 (address c t r 0)=(t.get r).dataPtr :=
  Torch.FP32.actual_pointer (t.get r) (t.laws r) (t.contiguous r) hn

theorem native_logical (c : Shape) (t : Objects c) (read : Nat → Native.Word) (r : RegionName) (o : Nat) :
    (Torch.FP32.logicalValue (fun a=>⟨read a⟩) (t.get r) (t.laws r) o).bits=logical read (t.get r) o := by
  simp only [Torch.FP32.logicalValue,t.resolved r,Bool.false_eq_true,ite_false,logical]

structure Result (α : Type) where
  output : Tensor
  dq : Tensor
  dk : Tensor
  dv : Tensor
  scaleGradient : Option Tensor
  read : Nat → α

def result {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α) : Result α:=
  ⟨t.get "Out",t.get "DQ",t.get "DK",t.get "DV",none,publish c t before final⟩

#axiomsClean publish_readback
#axiomsClean publish_frame
#axiomsClean publish_input
#axiomsClean native_logical
#specNonCircular Objects avoiding [FP.Structural.exec,Grid.wholeRun,Native.Selected.nativeRun]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage

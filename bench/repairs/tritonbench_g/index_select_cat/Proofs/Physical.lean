import bench.repairs.tritonbench_g.index_select_cat.Proofs.Host
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.FP
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.IndexSelectCat.Physical
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev encode := KvCacheFilling.wordByte
abbrev decode := KvCacheFilling.decodeWord

def read (m : Memory) (p : Nat) : Word := decode (fun b => m (p+b.val))
def words (m : Memory) (v : View) (i : Nat) : Word := read m (Native.address v 4 i)

/-- Physical realization of the typed source's complete FP32 word stores.
This is a memory observation of those same stores, not a host copyback.
Native bounds/alignment and the ordinary b32 load/store contract establish
the four-byte realization; all bytes outside the output window are retained. -/
def realize (r : Request) (output : Nat → Word) (m : Memory) (p : Nat) : Byte :=
  let base := r.output.metadata.dataPtr.toNat
  if base ≤ p ∧ p<base+4*(r.N*r.D) then encode (output ((p-base)/4)) ((p-base)%4) else m p

def run (rest : Algebra Word) (r : Request) (m : Memory) (object : Nat)
    (incoming : VeriTile.Torch.Device) : Memory :=
  realize r (Host.run rest r (words m r.source) (words m r.output) object incoming).output m

theorem realize_read (r : Request) (output : Nat → Word) (m : Memory)
    (i b : Nat) (hi : i<r.N*r.D) (hb : b<4) :
    realize r output m (Native.address r.output 4 i+b)=encode (output i) b := by
  have window : r.output.metadata.dataPtr.toNat ≤ Native.address r.output 4 i+b ∧
      Native.address r.output 4 i+b<r.output.metadata.dataPtr.toNat+4*(r.N*r.D) := by
    unfold Native.address
    omega
  simp only [realize]
  rw [if_pos window]
  simp only [Native.address]
  have sub : r.output.metadata.dataPtr.toNat+4*i+b-r.output.metadata.dataPtr.toNat=4*i+b := by omega
  rw [sub]
  rw [show (4*i+b)/4=i by omega,show (4*i+b)%4=b by omega]

theorem byte_readback (rest : Algebra Word) (r : Request) (checked : checkInputs r=Bool.true)
    (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device)
    (i j b : Nat) (hi : i<r.N) (hj : j<r.D) (hb : b<4) :
    run rest r m object incoming (Native.address r.output 4 (i*r.D+j)+b)=
      m (Native.address r.source 4 (r.row i*r.D+j)+b) := by
  rw [run,realize_read r _ m _ b (Native.flat_bound _ _ _ _ hi hj) hb]
  have source := (Host.whole_call rest r checked (words m r.source) (words m r.output) object incoming).1 i j hi hj
  rw [source]
  exact KvCacheFilling.fp32_byte_codec (fun b => m (Native.address r.source 4 (r.row i*r.D+j)+b.val)) ⟨b,hb⟩

theorem word_readback (rest : Algebra Word) (r : Request) (checked : checkInputs r=Bool.true)
    (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device)
    (i j : Nat) (hi : i<r.N) (hj : j<r.D) :
    words (run rest r m object incoming) r.output (i*r.D+j)=words m r.source (r.row i*r.D+j) := by
  unfold words read
  congr 1
  funext b
  exact byte_readback rest r checked m object incoming i j b hi hj b.isLt

theorem frame (rest : Algebra Word) (r : Request) (m : Memory) (object : Nat)
    (incoming : VeriTile.Torch.Device) (p : Nat)
    (outside : p<r.output.metadata.dataPtr.toNat ∨ r.output.metadata.dataPtr.toNat+4*(r.N*r.D) ≤ p) :
    run rest r m object incoming p=m p := by
  simp only [run,realize]
  rw [if_neg (by omega)]

/-- Actual contiguous Tensor windows inside the backing allocations queried
by the guard. These are ordinary storage/allocator facts, not output facts. -/
structure StorageBinding (r : Request) : Prop where
  output : 0<r.N*r.D → r.output.storageBase ≤ r.output.metadata.dataPtr.toNat ∧
    r.output.metadata.dataPtr.toNat+4*(r.N*r.D) ≤ r.output.storageBase+r.output.storageBytes
  source : 0<r.M*r.D → r.source.storageBase ≤ r.source.metadata.dataPtr.toNat ∧
    r.source.metadata.dataPtr.toNat+4*(r.M*r.D) ≤ r.source.storageBase+r.source.storageBytes
  index : 0<r.N → r.index.storageBase ≤ r.index.metadata.dataPtr.toNat ∧
    r.index.metadata.dataPtr.toNat+8*r.N ≤ r.index.storageBase+r.index.storageBytes
  native : Native.Storage r

theorem outside_separate (out input : View) (count : Nat) (sep : separate out input)
    (span : 0<count → out.storageBase ≤ out.metadata.dataPtr.toNat ∧
      out.metadata.dataPtr.toNat+4*count ≤ out.storageBase+out.storageBytes)
    (p : Nat) (inside : input.storageBase ≤ p ∧ p < input.storageBase+input.storageBytes) :
    p<out.metadata.dataPtr.toNat ∨ out.metadata.dataPtr.toNat+4*count ≤ p := by
  by_cases empty : count=0
  · omega
  have os := span (by omega)
  rcases sep with h|h|h|h <;> omega

theorem input_frame (rest : Algebra Word) (r : Request) (checked : checkInputs r=Bool.true)
    (storage : StorageBinding r) (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device)
    (p : Nat) (inside :
      (r.source.storageBase ≤ p ∧ p<r.source.storageBase+r.source.storageBytes) ∨
      (r.index.storageBase ≤ p ∧ p<r.index.storageBase+r.index.storageBytes)) :
    run rest r m object incoming p=m p := by
  have h := (Native.checkInputs_iff r).mp checked
  have sep := h.1.2.2.2.2.2.2.2.2.2.2.2
  apply frame
  rcases inside with hs|hi
  · exact outside_separate r.output r.source _ sep.1 storage.output p hs
  · exact outside_separate r.output r.index _ sep.2 storage.output p hi

def indexAt (r : Request) (m : Memory) (i : Nat) : Int :=
  (BitVec.ofNat 64 ((List.range 8).map fun b =>
    (m (Native.address r.index 8 i+b)).toNat*2^(8*b)).sum).toInt

/-- The ordinary logical int64 CPU observation equals the same contiguous,
nonnegative-view-bit native int64 load. This names only original input bytes. -/
def IndexBinding (r : Request) (m : Memory) : Prop :=
  ∀ i,i<r.N → r.indices i=indexAt r m i

theorem index_preserved (rest : Algebra Word) (r : Request) (checked : checkInputs r=Bool.true)
    (storage : StorageBinding r) (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device)
    (i : Nat) (hi : i<r.N) : indexAt r (run rest r m object incoming) i=indexAt r m i := by
  unfold indexAt
  congr 3
  apply List.map_congr_left
  intro b hb
  have bbound := List.mem_range.mp hb
  have span := storage.index (by omega)
  have inside : r.index.storageBase ≤ Native.address r.index 8 i+b ∧
      Native.address r.index 8 i+b<r.index.storageBase+r.index.storageBytes := by
    unfold Native.address
    omega
  rw [input_frame rest r checked storage m object incoming _ (Or.inr inside)]

open VeriTile.Torch.FP32

structure Objects (r : Request) where
  (source output : Tensor)
  sourceMetadata : source.metadata=r.source.metadata
  outputMetadata : output.metadata=r.output.metadata
  sourceLaws : TensorLaws source
  outputLaws : TensorLaws output

theorem direct (t : Tensor) (laws : TensorLaws t) (m : Memory)
    (contiguous : t.metadata.isContiguous=Bool.true) (negative : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    read m (t.metadata.dataPtr.toNat+4*i)=logicalValue (read m) t laws i := by
  have eq := direct_read_eq_logical (read m) t laws contiguous negative i hi
  have ptr := congrArg BitVec.toNat (actual_pointer t laws contiguous (by omega))
  have span := (dataView t laws contiguous).spanFits
  have bound : (dataView t laws contiguous).base<2^64 := by
    change (dataView t laws contiguous).base+4*t.metadata.numel ≤ 2^64 at span
    omega
  simp only [DataView.address,Nat.mul_zero,Nat.add_zero,Tensor.dataPtr,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt bound] at ptr
  simpa only [DataView.address,ptr] using eq

theorem logical_readback (rest : Algebra Word) (r : Request) (checked : checkInputs r=Bool.true)
    (objects : Objects r) (m : Memory) (incoming : VeriTile.Torch.Device)
    (i j : Nat) (hi : i<r.N) (hj : j<r.D) :
    logicalValue (read (run rest r m objects.output.object incoming)) objects.output objects.outputLaws (i*r.D+j)=
      Host.target r.indices
        (fun row col=>logicalValue (read m) objects.source objects.sourceLaws (row*r.D+col)) i j := by
  have h := (Native.checkInputs_iff r).mp checked
  have om := h.1.2.2.2.2.2.2.2.1 r.output (by simp)
  have sm := h.1.2.2.2.2.2.2.2.1 r.source (by simp)
  have ob : i*r.D+j<objects.output.metadata.numel := by
    rw [objects.outputMetadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,h.1.2.1,Nat.mul_comm] using Native.flat_bound r.N r.D i j hi hj
  have ri := (Native.index_valid r h i hi (by omega)).2.1
  have sb : r.row i*r.D+j<objects.source.metadata.numel := by
    rw [objects.sourceMetadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,h.1.1,Nat.mul_comm] using Native.flat_bound r.M r.D (r.row i) j ri hj
  rw [←direct objects.output objects.outputLaws _ (by simpa [objects.outputMetadata] using om.2.2.1)
    (by simpa [objects.outputMetadata] using om.2.2.2.1) _ ob]
  rw [objects.outputMetadata]
  change words (run rest r m objects.output.object incoming) r.output (i*r.D+j)=_
  rw [word_readback rest r checked m objects.output.object incoming i j hi hj]
  simpa only [words,Native.address,objects.sourceMetadata,Host.target,Request.row] using
    direct objects.source objects.sourceLaws m (by simpa [objects.sourceMetadata] using sm.2.2.1)
      (by simpa [objects.sourceMetadata] using sm.2.2.2.1) _ sb

#axiomsClean byte_readback
#axiomsClean word_readback
#axiomsClean frame
#axiomsClean input_frame
#axiomsClean index_preserved
#axiomsClean logical_readback
end
end VeriTile.Bench.Repairs.IndexSelectCat.Physical

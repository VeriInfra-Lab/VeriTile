import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Host
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.FP
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Physical
open VeriTile.Triton FP.Structural Logical
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev encode := KvCacheFilling.wordByte
abbrev decode := KvCacheFilling.decodeWord

def read (m : Memory) (p : Nat) : Word := decode (fun b => m (p+b.val))
def words (m : Memory) (v : View) (i : Nat) : Word := read m (Native.address v 4 i)

/-- The typed source's two memory regions name the same original caller
bytes at every actual active slice lane. The destination observation is of
the current state, so earlier repeated-index additions feed the next load. -/
theorem source_loads (r : Request) (guard : InputContract r) (m : Memory)
    (i : Fin r.N) (k lane : Nat) (active : k*512+lane<r.C) :
    let state := Program.initial (Native.selected r guard)
      (fun q : Coord r.N r.C => words m r.incoming (Host.flat q)) i k
      (fun q : Coord r.R r.C => words m r.destination (Host.flat q))
    state.mem Program.destinationRegion lane=Cell.mk .real
      (read m (Native.address r.destination 4 ((Native.selected r guard i).val*r.C+k*512)+4*lane)) ∧
    state.mem Program.incomingRegion lane=Cell.mk .real
      (read m (Native.address r.incoming 4 (i.val*r.C+k*512)+4*lane)) := by
  simp only [Program.initial,dif_pos active,
    if_neg (by decide : Program.destinationRegion≠Program.incomingRegion)]
  simp only [Host.flat,words,Native.slice_address]
  trivial

theorem decode_encode (w : Word) : decode (fun b => encode w b)=w := by
  rcases w with ⟨bits⟩
  change FP.Value.ofBits .fp32 _ = {bits:=bits}
  unfold FP.Value.ofBits
  congr 1
  apply BitVec.eq_of_toNat_eq
  have bound := bits.isLt
  simp only [encode,KvCacheFilling.wordByte,
    FP.Format.width,FP.Format.fractionBits,BitVec.toNat_ofNat]
  norm_num at *
  omega

/-- The physical observation of the complete typed source's native b32
stores. This realizes the same words in the original caller window; it is
not a Python payload transfer or a host copyback. -/
def realize (r : Request) (output : Nat → Word) (m : Memory) (p : Nat) : Byte :=
  let base := r.destination.metadata.dataPtr.toNat
  if base ≤ p ∧ p<base+4*(r.R*r.C) then encode (output ((p-base)/4)) ((p-base)%4) else m p

def run (rest : Algebra Word) (r : Request) (guard : InputContract r) (m : Memory)
    (object : Nat) (device : VeriTile.Torch.Device) : Memory :=
  realize r (Host.run rest r guard (words m r.destination) (words m r.incoming) object device).destination m

theorem realize_byte (r : Request) (output : Nat → Word) (m : Memory)
    (i b : Nat) (hi : i<r.R*r.C) (hb : b<4) :
    realize r output m (Native.address r.destination 4 i+b)=encode (output i) b := by
  have window : r.destination.metadata.dataPtr.toNat ≤ Native.address r.destination 4 i+b ∧
      Native.address r.destination 4 i+b<r.destination.metadata.dataPtr.toNat+4*(r.R*r.C) := by
    unfold Native.address
    omega
  simp only [realize]
  rw [if_pos window]
  simp only [Native.address]
  have sub : r.destination.metadata.dataPtr.toNat+4*i+b-r.destination.metadata.dataPtr.toNat=4*i+b := by omega
  rw [sub,show (4*i+b)/4=i by omega,show (4*i+b)%4=b by omega]

theorem word_source (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (m : Memory) (object : Nat) (device : VeriTile.Torch.Device) (q : Coord r.R r.C) :
    words (run rest r guard m object device) r.destination (Host.flat q)=
      (Host.run rest r guard (words m r.destination) (words m r.incoming) object device).destination (Host.flat q) := by
  unfold words read
  have bytes : (fun b : Fin 4 => run rest r guard m object device
      (Native.address r.destination 4 (Host.flat q)+b.val))=
      (fun b : Fin 4 => encode ((Host.run rest r guard (words m r.destination)
        (words m r.incoming) object device).destination (Host.flat q)) b.val) := by
    funext b
    exact realize_byte r _ m _ b.val (Host.flat_bound q) b.isLt
  rw [bytes,decode_encode]
  rfl

theorem word_readback (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (m : Memory) (object : Nat) (device : VeriTile.Torch.Device) (q : Coord r.R r.C) :
    words (run rest r guard m object device) r.destination (Host.flat q)=
      Host.orderedTarget (Source.scalar rest) r (words m r.destination) (words m r.incoming) q := by
  rw [word_source,Host.ordered_readback]

theorem exterior_frame (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (m : Memory) (object : Nat) (device : VeriTile.Torch.Device) (p : Nat)
    (outside : p<r.destination.metadata.dataPtr.toNat ∨ r.destination.metadata.dataPtr.toNat+4*(r.R*r.C) ≤ p) :
    run rest r guard m object device p=m p := by
  simp only [run,realize]
  rw [if_neg (by omega)]

theorem unselected_bytes (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (m : Memory) (object : Nat) (device : VeriTile.Torch.Device) (q : Coord r.R r.C)
    (miss : ∀ i : Fin r.N,r.indices i≠(q.1.val : Int)) (b : Fin 4) :
    run rest r guard m object device (Native.address r.destination 4 (Host.flat q)+b.val)=
      m (Native.address r.destination 4 (Host.flat q)+b.val) := by
  rw [run,realize_byte r _ m _ b.val (Host.flat_bound q) b.isLt,
    Host.unselected_frame rest r guard _ _ object device q miss]
  exact KvCacheFilling.fp32_byte_codec
    (fun b : Fin 4 => m (Native.address r.destination 4 (Host.flat q)+b.val)) b

/-- Ordinary storage observation: actual windows fit the queried backing
intervals and native pointer space. No arithmetic result occurs here. -/
structure StorageBinding (r : Request) : Prop where
  destination : 0<r.R*r.C → r.destination.storageBase ≤ r.destination.metadata.dataPtr.toNat ∧
    r.destination.metadata.dataPtr.toNat+4*(r.R*r.C) ≤ r.destination.storageBase+r.destination.storageBytes
  incoming : 0<r.N*r.C → r.incoming.storageBase ≤ r.incoming.metadata.dataPtr.toNat ∧
    r.incoming.metadata.dataPtr.toNat+4*(r.N*r.C) ≤ r.incoming.storageBase+r.incoming.storageBytes
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

theorem input_frame (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (storage : StorageBinding r) (m : Memory) (object : Nat) (device : VeriTile.Torch.Device)
    (p : Nat) (inside :
      (r.incoming.storageBase ≤ p ∧ p<r.incoming.storageBase+r.incoming.storageBytes) ∨
      (r.index.storageBase ≤ p ∧ p<r.index.storageBase+r.index.storageBytes)) :
    run rest r guard m object device p=m p := by
  rcases guard.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,sepH,sepI⟩
  apply exterior_frame
  rcases inside with hh|hi
  · exact outside_separate r.destination r.incoming _ sepH storage.destination p hh
  · exact outside_separate r.destination r.index _ sepI storage.destination p hi

def indexAt (r : Request) (m : Memory) (i : Nat) : Int :=
  (BitVec.ofNat 64 ((List.range 8).map fun b =>
    (m (Native.address r.index 8 i+b)).toNat*2^(8*b)).sum).toInt

/-- Logical CPU metadata observation of the original native int64 bytes.
The kernel has no index pointer: the host submits these selected row views. -/
def IndexBinding (r : Request) (m : Memory) : Prop :=
  ∀ i,i<r.N → r.indices i=indexAt r m i

theorem index_preserved (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (storage : StorageBinding r) (m : Memory) (object : Nat) (device : VeriTile.Torch.Device)
    (i : Nat) (hi : i<r.N) : indexAt r (run rest r guard m object device) i=indexAt r m i := by
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
  rw [input_frame rest r guard storage m object device _ (Or.inr inside)]

open VeriTile.Torch.FP32

structure Objects (r : Request) where
  (destination incoming : Tensor)
  destinationMetadata : destination.metadata=r.destination.metadata
  incomingMetadata : incoming.metadata=r.incoming.metadata
  destinationLaws : TensorLaws destination
  incomingLaws : TensorLaws incoming

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

theorem destination_direct (r : Request) (guard : InputContract r) (objects : Objects r)
    (m : Memory) (q : Coord r.R r.C) :
    words m r.destination (Host.flat q)=
      logicalValue (read m) objects.destination objects.destinationLaws (Host.flat q) := by
  rcases guard.1 with ⟨shape,_,_,_,_,_,_,_,_,views,_⟩
  have vm := views r.destination (by simp)
  have bound : Host.flat q<objects.destination.metadata.numel := by
    rw [objects.destinationMetadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,shape] using Host.flat_bound q
  simpa only [words,Native.address,objects.destinationMetadata] using direct objects.destination
    objects.destinationLaws m (by simpa [objects.destinationMetadata] using vm.2.2.1)
    (by simpa [objects.destinationMetadata] using vm.2.2.2.1) _ bound

theorem incoming_direct (r : Request) (guard : InputContract r) (objects : Objects r)
    (m : Memory) (q : Coord r.N r.C) :
    words m r.incoming (Host.flat q)=
      logicalValue (read m) objects.incoming objects.incomingLaws (Host.flat q) := by
  rcases guard.1 with ⟨_,shape,_,_,_,_,_,_,_,views,_⟩
  have vm := views r.incoming (by simp)
  have bound : Host.flat q<objects.incoming.metadata.numel := by
    rw [objects.incomingMetadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,shape] using Host.flat_bound q
  simpa only [words,Native.address,objects.incomingMetadata] using direct objects.incoming
    objects.incomingLaws m (by simpa [objects.incomingMetadata] using vm.2.2.1)
    (by simpa [objects.incomingMetadata] using vm.2.2.2.1) _ bound

theorem target_logical (add : Word → Word → Word) (r : Request) (guard : InputContract r)
    (objects : Objects r) (m : Memory) (q : Coord r.R r.C) :
    Host.orderedTarget add r (words m r.destination) (words m r.incoming) q=
      Host.orderedTarget add r (logicalValue (read m) objects.destination objects.destinationLaws)
        (logicalValue (read m) objects.incoming objects.incomingLaws) q := by
  unfold Host.orderedTarget
  rw [destination_direct r guard objects m q]
  congr 1
  funext acc i
  rw [incoming_direct r guard objects m (i,q.2)]

theorem logical_readback (rest : Algebra Word) (r : Request) (guard : InputContract r)
    (objects : Objects r) (m : Memory) (device : VeriTile.Torch.Device) (q : Coord r.R r.C) :
    logicalValue (read (run rest r guard m objects.destination.object device))
        objects.destination objects.destinationLaws (Host.flat q)=
      Host.orderedTarget (Source.scalar rest) r
        (logicalValue (read m) objects.destination objects.destinationLaws)
        (logicalValue (read m) objects.incoming objects.incomingLaws) q := by
  rw [←destination_direct r guard objects _ q,word_readback]
  exact target_logical _ r guard objects m q

#axiomsClean decode_encode
#axiomsClean source_loads
#axiomsClean word_readback
#axiomsClean exterior_frame
#axiomsClean unselected_bytes
#axiomsClean input_frame
#axiomsClean index_preserved
#axiomsClean logical_readback
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Physical

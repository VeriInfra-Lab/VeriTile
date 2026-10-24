import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelObserved
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
open VeriTile Triton FP.Structural
open Scalar
noncomputable section
set_option maxHeartbeats 4000000

def address (t : Torch.FP32.Tensor) (off : Nat) := t.dataPtr.toNat+4*off
def logical (read : Nat→β) (t : Torch.FP32.Tensor) (i : Nat) : β :=
  read (t.storage.address (t.physicalIndex i))

/-- Ordinary strided Tensor API bindings for the actual arguments. Coordinates
describe metadata only. No buffer values or execution results occur here. -/
structure Inputs (c : Geometry) (hasBias : Bool) where
  dt : Torch.FP32.Tensor
  rate : Torch.FP32.Tensor
  bias : Option Torch.FP32.Tensor
  dtLaws : Torch.FP32.TensorLaws dt
  rateLaws : Torch.FP32.TensorLaws rate
  biasLaws : ∀ t∈bias.toList, Torch.FP32.TensorLaws t
  dtShape : dt.metadata.shape=[c.batches,c.sequence,c.heads]
  rateShape : rate.metadata.shape=[c.heads]
  biasShape : ∀ t∈bias.toList, t.metadata.shape=[c.heads]
  biasEnabled : bias.isSome=hasBias
  dtCoordinates : ∀ b t h, b<c.batches → t<c.sequence → h<c.heads →
    dt.physicalIndex ((b*c.sequence+t)*c.heads+h)=
      dt.storageOffset+(b*c.dtBatch+t*c.dtTime+h*c.dtHead)
  rateCoordinates : ∀ h, h<c.heads → rate.physicalIndex h=rate.storageOffset+h*c.rateHead
  biasCoordinates : ∀ t∈bias.toList, ∀ h,h<c.heads →
    t.physicalIndex h=t.storageOffset+h*c.biasHead

def Inputs.all (a : Inputs c hasBias) : List Torch.FP32.Tensor := [a.dt,a.rate]++a.bias.toList

inductive Output where | prepared | cumulative deriving DecidableEq
def Output.region : Output→RegionName | .prepared=>"out" | .cumulative=>"sum"

structure OutputView (c : Geometry) where
  tensor : Torch.FP32.Tensor
  laws : Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape=[c.batches,c.heads,c.chunks,c.chunk]
  contiguous : tensor.metadata.isContiguous=Bool.true
  resolved : tensor.metadata.isNeg=Bool.false
  aligned : 0<tensor.metadata.numel → tensor.dataPtr.toNat%4=0

def Apart (a b : Torch.FP32.Tensor) : Prop :=
  a.storage.base+4*a.storage.elements ≤ b.storage.base ∨
    b.storage.base+4*b.storage.elements ≤ a.storage.base

/-- Two ordinary fresh output allocations. All input/input aliases, offsets
and zero strides remain admitted. Both outputs share the same physical heap. -/
structure Buffers (c : Geometry) (hasBias : Bool) where
  inputs : Inputs c hasBias
  output : Output→OutputView c
  device : ∀ r,(output r).tensor.metadata.device=inputs.dt.metadata.device
  outputSeparate : ∀ r s,r≠s → Apart (output r).tensor (output s).tensor
  inputSeparate : ∀ t∈inputs.all, ∀ r,Apart t (output r).tensor

theorem output_numel (v : OutputView c) : v.tensor.metadata.numel=c.outputSize := by
  simp [Torch.TensorMetadata.numel,v.shape,Geometry.outputSize,Nat.mul_assoc]

theorem tensor_address (t : Torch.FP32.Tensor) (laws : Torch.FP32.TensorLaws t)
    (index off : Nat) (hi : index<t.metadata.numel)
    (coord : t.physicalIndex index=t.storageOffset+off) :
    t.storage.address (t.physicalIndex index)=address t off := by
  have bound := laws.inBounds index hi
  rw [coord] at bound
  have span := t.storage.spanFits
  have base : t.storage.address t.storageOffset<2^64 := by
    unfold Torch.FP32.Storage.address
    omega
  have ptr := congrArg BitVec.toNat (laws.pointer (by omega))
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt base] at ptr
  unfold address
  rw [ptr,coord]
  simp only [Torch.FP32.Storage.address]
  ring

theorem output_coordinates (v : OutputView c) (i : Nat) (hi : i<c.outputSize) :
    v.tensor.storage.address (v.tensor.physicalIndex i)=address v.tensor i := by
  have h : i<v.tensor.metadata.numel := by simpa only [output_numel] using hi
  exact tensor_address v.tensor v.laws i i h (v.laws.contiguous v.contiguous i h)

theorem input_dt_numel (a : Inputs c flag) : a.dt.metadata.numel=c.batches*c.sequence*c.heads := by
  simp [Torch.TensorMetadata.numel,a.dtShape,Nat.mul_assoc]
theorem input_rate_numel (a : Inputs c flag) : a.rate.metadata.numel=c.heads := by
  simp [Torch.TensorMetadata.numel,a.rateShape]

theorem dt_index_bound (c : Geometry) (b t h : Nat)
    (hb : b<c.batches) (ht : t<c.sequence) (hh : h<c.heads) :
    (b*c.sequence+t)*c.heads+h<c.batches*c.sequence*c.heads :=
  pair_bound (pair_bound hb ht) hh

theorem input_dt_coordinates (a : Inputs c flag) (b t h : Nat)
    (hb : b<c.batches) (ht : t<c.sequence) (hh : h<c.heads) :
    a.dt.storage.address (a.dt.physicalIndex ((b*c.sequence+t)*c.heads+h))=
      address a.dt (b*c.dtBatch+t*c.dtTime+h*c.dtHead) := by
  apply tensor_address a.dt a.dtLaws _ _
    (by rw [input_dt_numel]; exact dt_index_bound c b t h hb ht hh)
  exact a.dtCoordinates b t h hb ht hh

theorem input_rate_coordinates (a : Inputs c flag) (h : Nat) (hh : h<c.heads) :
    a.rate.storage.address (a.rate.physicalIndex h)=address a.rate (h*c.rateHead) := by
  exact tensor_address a.rate a.rateLaws h _ (by simpa only [input_rate_numel] using hh)
    (a.rateCoordinates h hh)

theorem input_bias_coordinates (a : Inputs c flag) (t : Torch.FP32.Tensor)
    (present : t∈a.bias.toList) (h : Nat) (hh : h<c.heads) :
    t.storage.address (t.physicalIndex h)=address t (h*c.biasHead) := by
  apply tensor_address t (a.biasLaws t present) h _
    (by simpa [Torch.TensorMetadata.numel,a.biasShape t present] using hh)
  exact a.biasCoordinates t present h hh

/-- Every source input word is read at its actual pointer from one initial heap. -/
def snapshot (zero : α) (read : Nat→α) (a : Inputs c hasBias) : InputWords α where
  dt := fun off => read (address a.dt off)
  rate := fun off => read (address a.rate off)
  bias := fun off => match a.bias with | none=>zero | some t=>read (address t off)

/-- Every mapped source cell begins at its corresponding entry-memory word,
including the fresh output allocations before their first stores. -/
def entryState (zero : α) (read : Nat→α) (f : Buffers c hasBias) : State (Term α) :=
  { initial (snapshot zero read f.inputs) with
    mem := fun region off =>
      if region="out" then .mk .real (.input (read (address (f.output .prepared).tensor off)))
      else if region="sum" then .mk .real (.input (read (address (f.output .cumulative).tensor off)))
      else (initial (snapshot zero read f.inputs)).mem region off }

theorem entry_graph (zero : α) (read : Nat→α) (f : Buffers c hasBias) :
    InputGraph (entryState zero read f) (snapshot zero read f.inputs) := by
  refine ⟨?_,?_,?_⟩ <;> intro off <;> simp [entryState,initial,Cell.read]

theorem snapshot_dt (zero : α) (read : Nat→α) (a : Inputs c hasBias) (b t h : Nat)
    (hb : b<c.batches) (ht : t<c.sequence) (hh : h<c.heads) :
    (logicalInputs c (snapshot zero read a)).dt b t h=
      logical read a.dt ((b*c.sequence+t)*c.heads+h) := by
  simp only [logicalInputs,snapshot,logical,input_dt_coordinates a b t h hb ht hh]

theorem snapshot_rate (zero : α) (read : Nat→α) (a : Inputs c hasBias) (h : Nat) (hh : h<c.heads) :
    (logicalInputs c (snapshot zero read a)).rate h=logical read a.rate h := by
  simp only [logicalInputs,snapshot,logical,input_rate_coordinates a h hh]

theorem snapshot_bias (zero : α) (read : Nat→α) (a : Inputs c hasBias)
    (t : Torch.FP32.Tensor) (present : a.bias=some t) (h : Nat) (hh : h<c.heads) :
    (logicalInputs c (snapshot zero read a)).bias h=logical read t h := by
  have member : t∈a.bias.toList := by simp [present]
  simp only [logicalInputs,snapshot,present,logical,input_bias_coordinates a t member h hh]

theorem sizes_of_contract (a : Inputs c hasBias)
    (checked : InputContract a.dt.metadata a.rate.metadata (a.bias.map (·.metadata))
      (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits) : Sizes c := by
  have hb := checked.shapeBounds c.batches (by simp [a.dtShape])
  have hs := checked.shapeBounds c.sequence (by simp [a.dtShape])
  have hh := checked.shapeBounds c.heads (by simp [a.dtShape])
  refine ⟨hb,hs,hh,?_⟩
  have bound := checked.chunkBound
  simp only [Int.ofNat_eq_natCast] at bound
  exact_mod_cast bound

theorem cuda_launch (a : Inputs c hasBias)
    (checked : InputContract a.dt.metadata a.rate.metadata (a.bias.map (·.metadata))
      (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits)
    (l : Math.Launch) (member : l∈c.grid) :
    l.batch<2^31 ∧ l.chunk<65535 ∧ l.head<65535 := by
  have valid := (c.mem_grid l).1 member
  have batch := valid.1
  have time := chunk_start_lt c l valid
  have head := start_lt c.headPositive valid.2.2
  have positive : 0<a.dt.metadata.numel := by
    rw [input_dt_numel]
    exact Nat.mul_pos (Nat.mul_pos (by omega) (by omega)) (by omega)
  have bounds : c.batches<2^31 ∧ c.chunks≤65535 ∧ c.heads≤65535 := by
    simpa [a.dtShape,Geometry.chunks,ceilDiv] using checked.grid positive
  have hm : l.head≤l.head*c.BH := Nat.le_mul_of_pos_right _ c.headPositive
  exact ⟨lt_trans valid.1 bounds.1,lt_of_lt_of_le valid.2.1 bounds.2.1,by omega⟩

def Aligned (f : Buffers c hasBias) : Prop :=
  (∀ t∈f.inputs.all,∀ off,0<t.metadata.numel → address t off%4=0) ∧
  (∀ r : Output,∀ off,off<c.outputSize → address (f.output r).tensor off%4=0)

theorem aligned (f : Buffers c hasBias)
    (checked : InputContract f.inputs.dt.metadata f.inputs.rate.metadata
      (f.inputs.bias.map (·.metadata)) (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits) :
    Aligned f := by
  constructor
  · intro t member off positive
    have mem : t=f.inputs.dt ∨ t=f.inputs.rate ∨ t∈f.inputs.bias.toList := by
      simpa [Inputs.all] using member
    have metadata : t.metadata∈tensors f.inputs.dt.metadata f.inputs.rate.metadata
        (f.inputs.bias.map (·.metadata)) := by
      rcases mem with he|he|hm
      · subst t; simp [tensors]
      · subst t; simp [tensors]
      · cases hb : f.inputs.bias with
        | none => simp [hb] at hm
        | some v =>
          have he : t=v := by simpa [hb] using hm
          subst t
          simp [tensors,hb]
    have h := (checked.tensors t.metadata metadata).alignment positive
    simpa [address,Torch.FP32.Tensor.dataPtr,Nat.add_mod] using h
  · intro r off bound
    have positive : 0<(f.output r).tensor.metadata.numel := by rw [output_numel]; omega
    have h := (f.output r).aligned positive
    simpa [address,Nat.add_mod] using h

theorem native_logical (read : Nat→BitVec 32) (t : Torch.FP32.Tensor)
    (laws : Torch.FP32.TensorLaws t) (resolved : t.metadata.isNeg=Bool.false) (i : Nat) :
    (Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) t laws i).bits=logical read t i := by
  simp [Torch.FP32.logicalValue,resolved,logical]

/-- The unchanged checked-input contract resolves the Tensor negative-view
bit, so source raw words are exactly the caller's logical FP32 values. -/
theorem native_input_values (read : Nat→BitVec 32) (a : Inputs c hasBias)
    (checked : InputContract a.dt.metadata a.rate.metadata (a.bias.map (·.metadata))
      (Int.ofNat c.chunk) integerChunk booleanSoftplus floatLimits) :
    (∀ b t h,b<c.batches → t<c.sequence → h<c.heads →
      (logicalInputs c (snapshot 0 read a)).dt b t h=
        (Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) a.dt a.dtLaws
          ((b*c.sequence+t)*c.heads+h)).bits) ∧
    (∀ h,h<c.heads → (logicalInputs c (snapshot 0 read a)).rate h=
      (Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) a.rate a.rateLaws h).bits) ∧
    (∀ t,∀ present : a.bias=some t,∀ h,h<c.heads →
      (logicalInputs c (snapshot 0 read a)).bias h=
        (Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) t
          (a.biasLaws t (by simp [present])) h).bits) := by
  have dtResolved := (checked.tensors a.dt.metadata (by simp [tensors])).noNeg
  have rateResolved := (checked.tensors a.rate.metadata (by simp [tensors])).noNeg
  refine ⟨?_,?_,?_⟩
  · intro b t h hb ht hh
    rw [native_logical _ _ _ dtResolved,snapshot_dt 0 read a b t h hb ht hh]
  · intro h hh
    rw [native_logical _ _ _ rateResolved,snapshot_rate 0 read a h hh]
  · intro t present h hh
    have resolved := (checked.tensors t.metadata (by simp [tensors,present])).noNeg
    rw [native_logical _ _ _ resolved,snapshot_bias 0 read a t present h hh]

#guard_msgs (drop info) in
#specNonCircular Inputs avoiding [Kernel.launch,Kernel.sourceRun,Scalar.referenceLogicalPrepared]
#guard_msgs (drop info) in
#specNonCircular Buffers avoiding [Kernel.launch,Kernel.sourceRun,Scalar.referenceLogicalPrepared]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor

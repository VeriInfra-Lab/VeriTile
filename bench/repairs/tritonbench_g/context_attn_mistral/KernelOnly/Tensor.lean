import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Physical
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Footprint
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API)
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
open Checked NativeIndex Execution KernelPhysical
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def read (api : API Word Store Byte) (memory : Store) (address : Nat) : VeriTile.Torch.FP32.Word :=
  ⟨api.read memory address⟩

def role : DataRole → Checked.Role
  | .query => .query | .key => .key | .value => .value | .output => .output

def region (c : Host.Call) : DataRole → RegionName
  | .query => c.regions.query | .key => c.regions.key
  | .value => c.regions.value | .output => c.regions.out

/-- Ordinary API bindings of the original caller Tensor objects. These fields
refer only to metadata, addresses and storage layout, never a kernel result. -/
structure Views (c : Host.Call) (tokens : Nat) (request : Request) (buffers : Buffers) where
  tensor : DataRole → VeriTile.Torch.FP32.Tensor
  laws : ∀ r,TensorLaws (tensor r)
  object : ∀ r,(tensor r).object=(request.tensor (role r)).object
  shape : ∀ r,(tensor r).metadata.shape=[tokens,headsFor c r,c.layout.D]
  pointer : ∀ r,(tensor r).metadata.dataPtr.toNat=(buffers.data r).base
  contiguous : ∀ r,(tensor r).metadata.isContiguous=Bool.true
  notNeg : ∀ r,(tensor r).metadata.isNeg=Bool.false

theorem direct (api : API Word Store Byte) (memory : Store) (t : VeriTile.Torch.FP32.Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (notNeg : t.metadata.isNeg=Bool.false)
    (i : Nat) (bound : i<t.metadata.numel) :
    (logicalValue (read api memory) t laws i).bits=api.read memory (t.metadata.dataPtr.toNat+4*i) := by
  have size := laws.inBounds i bound
  rw [laws.contiguous contiguous i bound] at size
  have space := t.storage.spanFits
  have fits : t.storage.address t.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have pointer := laws.pointer (by omega : 0<t.metadata.numel)
  change t.metadata.dataPtr=BitVec.ofNat 64 (t.storage.address t.storageOffset) at pointer
  simp only [logicalValue,notNeg,Bool.false_eq_true,↓reduceIte,laws.contiguous contiguous i bound,read]
  rw [pointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]
  congr 1
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem numel (views : Views c tokens request buffers) (r : DataRole) :
    (views.tensor r).metadata.numel=size c tokens r := by
  simp [VeriTile.Torch.TensorMetadata.numel,views.shape,size,Nat.mul_assoc]

theorem base_role (binding : Binding c buffers base) (r : DataRole) :
    base (region c r)=(buffers.data r).base := by
  cases r
  · exact binding.query
  · exact binding.key
  · exact binding.value
  · exact binding.output

/-- Every original logical FP32 input word is exactly the physical source
leaf; the same statement also identifies the original output on entry. -/
theorem input_tensor (c : Host.Call) (api : API Word Store Byte) (base : Base) (buffers : Buffers)
    (binding : Binding c buffers base) (views : Views c tokens request buffers)
    (before : Store) (r : DataRole) (i : Nat) (bound : i<(views.tensor r).metadata.numel) :
    ((initial c api base before).mem (region c r) i).read .real=
      NativePhysical.Expr.leaf .fp32 ((logicalValue (read api before) (views.tensor r) (views.laws r) i).bits) := by
  have payload : region c r∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out] := by cases r <;> simp [region]
  simp only [initial,if_neg (binding.floating _ payload),Cell.read_same]
  rw [base_role binding r,direct api before (views.tensor r) (views.laws r) (views.contiguous r) (views.notNeg r) i bound,
    views.pointer]

theorem output_tensor (c : Host.Call) (api : API Word Store Byte) (buffers : Buffers)
    (admitted : Admission c tokens request buffers) (views : Views c tokens request buffers)
    (memory : Store) (q : Coord c) :
    (logicalValue (read api memory) (views.tensor .output) (views.laws .output) (offset q)).bits=
      api.read memory (address buffers q) := by
  rw [direct api memory (views.tensor .output) (views.laws .output) (views.contiguous .output)
    (views.notNeg .output) (offset q) (by rw [numel views .output];exact offset_bound admitted.geometry q)]
  rw [views.pointer]
  rfl

theorem metadata (c : Host.Call) (api : API Word Store Byte) (base : Base) (before : Store)
    (observed : Observed c api base before) : KernelFootprint.Metadata c (initial c api base before) := by
  constructor
  · intro b hb
    simpa [initial] using observed.1 b hb
  · intro b hb
    simpa [initial] using observed.2 b hb

#axiomsClean input_tensor
#axiomsClean output_tensor
#axiomsClean metadata
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelTensor

import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.ProgramGeometry
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Guards
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelTensor
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Tensor := VeriTile.Torch.TensorMetadata
abbrev Heap := Nat → Physical.Byte

def roles (s : Shape) : List RegionName := ["x","cos","sin"]++
  (if s.varlen then ["cu"] else [])++(if s.offsetTensor then ["offsets"] else [])

def descriptor (c : Checked.Call) (r : RegionName) : Checked.Tensor :=
  if r="x" then c.input else if r="cos" then c.cosine else if r="sin" then c.sine
  else if r="cu" then c.cumulative else c.offsets

/-- Observations of the actual Tensor handles passed to this kernel. The
checked descriptor contains the additional storage/stride observations.
These are representation laws about inputs, independent of kernel results. -/
structure Binding (c : Checked.Call) (tensors : RegionName→Tensor) : Prop where
  pointer : ∀ r,r∈roles c.host.shape → (tensors r).dataPtr.toNat=(descriptor c r).base
  shape : ∀ r,r∈roles c.host.shape → (tensors r).shape=(descriptor c r).shape
  cuda : ∀ r,r∈roles c.host.shape → (tensors r).device.kind="cuda"
  device : ∀ r,r∈roles c.host.shape → (tensors r).device.index=(descriptor c r).device
  strided : ∀ r,r∈roles c.host.shape → (tensors r).layout=.strided
  contiguous : ∀ r,r∈roles c.host.shape → (tensors r).isContiguous=(descriptor c r).contiguous
  noNeg : ∀ r,r∈roles c.host.shape → (tensors r).isNeg=Bool.false
  payload : ∀ r,r∈(["x","cos","sin"] : List RegionName) → (tensors r).dtype=.floating .fp32
  cumulative : c.host.shape.varlen=Bool.true → (tensors "cu").dtype=
    .other (if c.host.abi.cumulative64 then "int64" else "int32")
  offsets : c.host.shape.offsetTensor=Bool.true → (tensors "offsets").dtype=
    .other (if c.host.abi.offsets64 then "int64" else "int32")

def name : RotaryTransformOps.Region → RegionName
  | .input => "x" | .output => "out" | .cosine => "cos" | .sine => "sin"
  | .cumulative | .other 0 => "cu" | .offsets => "offsets" | .other _ => "other"

/-- Initial bytes are read at the actual Tensor data pointers. This is a
read-only view; it allocates nothing and contains no output formula. -/
def bytes (tensors : RegionName→Tensor) (heap : Heap) : Physical.Bytes :=
  fun r b => heap ((tensors (name r)).dataPtr.toNat+b)

def word (tensor : Tensor) (heap : Heap) (i : Nat) : Native.Word :=
  Physical.decode (fun byte => heap (tensor.dataPtr.toNat+4*i+byte.val))

theorem load_pointer (tensors : RegionName→Tensor) (heap : Heap)
    (r : RotaryTransformOps.Region) (i : Nat) :
    Physical.load (bytes tensors heap) r i=word (tensors (name r)) heap i := by
  simp [Physical.load,bytes,word,Nat.add_assoc]

def signed (m : Physical.Bytes) (r : RotaryTransformOps.Region) (wide : Bool) (i : Nat) : Int :=
  if wide then ((Physical.load m r (2*i+1) ++ Physical.load m r (2*i)) : BitVec 64).toInt
  else (Physical.load m r i).toInt

theorem signed_eq (m : Physical.Bytes) (r : RotaryTransformOps.Region) (wide : Bool) (i : Nat)
    (bound : Physical.integer m r wide i<2^31) :
    signed m r wide i=(Physical.integer m r wide i : Int) := by
  cases wide
  · exact Physical.signed32 _ bound
  · exact Physical.signed64 _ bound

/-- FP32 input cells preserve their raw word interpretation. Signed metadata
uses the original int32/int64 width at the Tensor pointer. Output initial cells
come from the already existing output Tensor, including any suffix copy. -/
def lift {α : Type} (observe : Native.Word→α) (c : Checked.Call)
    (tensors : RegionName→Tensor) (heap : Heap) (other : State α) : State α :=
  { other with mem := fun r i =>
      if r="cu" then Cell.mk .int (signed (bytes tensors heap) (.other 0) c.host.abi.cumulative64 i)
      else if r="offsets" then Cell.mk .int (signed (bytes tensors heap) .offsets c.host.abi.offsets64 i)
      else if r="x" ∨ r="out" ∨ r="cos" ∨ r="sin" then Cell.mk .real (observe (word (tensors r) heap i))
      else other.mem r i }

theorem lift_payload {α : Type} [Inhabited α] (observe : Native.Word→α) (c : Checked.Call)
    (tensors : RegionName→Tensor) (heap : Heap) (other : State α) (r : RegionName) (i : Nat)
    (role : r="x" ∨ r="out" ∨ r="cos" ∨ r="sin") :
    (lift observe c tensors heap other).mem r i=Cell.mk .real (observe (word (tensors r) heap i)) := by
  rcases role with rfl|rfl|rfl|rfl <;> simp [lift]

/-- A logical Tensor read is tied to the same physical input word that the
typed source loads. This law may use real interpretation or exact raw words. -/
def ReadLaw {α : Type} (observe : Native.Word→α) (tensors : RegionName→Tensor)
    (heap : Heap) (logical : RegionName→Nat→α) : Prop :=
  ∀ r,r∈(["x","cos","sin"] : List RegionName) → ∀ i,i<(tensors r).numel →
    observe (word (tensors r) heap i)=logical r i

theorem logical_read {α : Type} [Inhabited α] (observe : Native.Word→α) (c : Checked.Call)
    (tensors : RegionName→Tensor) (heap : Heap) (other : State α) (logical : RegionName→Nat→α)
    (reads : ReadLaw observe tensors heap logical) (r : RegionName)
    (role : r∈(["x","cos","sin"] : List RegionName)) (i : Nat) (inside : i<(tensors r).numel) :
    ((lift observe c tensors heap other).mem r i).read .real=logical r i := by
  rw [lift_payload observe c tensors heap other r i (by simp at role;tauto),Cell.read_same]
  exact reads r role i inside

theorem cumulative_bound (c : Config) (valid : c.Valid) (abi : IndexABI c)
    (packed : c.shape.varlen=Bool.true) (i : Nat) (inside : i≤c.shape.batches) :
    c.metadata.cumulative i<2^31 := by
  have total : c.shape.total<2^31 := by
    have a : c.shape.total≤c.shape.total*c.shape.heads := Nat.le_mul_of_pos_right _ valid.heads
    have b : c.shape.total*c.shape.heads≤c.shape.total*c.shape.heads*c.shape.width :=
      Nat.le_mul_of_pos_right _ valid.width
    have limit := abi.data
    unfold Host.size at limit
    omega
  have ending := valid.total
  simp only [packed,if_true] at ending
  have monotone := valid.cumulativeMonotone i c.shape.batches inside (Nat.le_refl _)
  omega

theorem offset_bound (c : Config) (abi : IndexABI c)
    (tensor : c.shape.offsetTensor=Bool.true) (i : Nat) (inside : i<c.shape.batches) :
    c.metadata.offsets i<2^31 := by
  have bound := abi.offsets i inside
  simp only [Config.shift,tensor,if_true] at bound
  omega

theorem lift_metadata {α : Type} [Inhabited α] (observe : Native.Word→α) (c : Checked.Call)
    (tensors : RegionName→Tensor) (heap : Heap) (other : State α)
    (accepted : Checked.check c (bytes tensors heap)=Bool.true) :
    ProgramModel.MetadataBinding (Checked.configuration c (bytes tensors heap))
      (lift observe c tensors heap other) := by
  let cfg := Checked.configuration c (bytes tensors heap)
  have valid : cfg.Valid := Checked.config_valid c _ accepted
  have abi : IndexABI cfg := Checked.indices c _ accepted
  have tensorsAccepted := (Checked.checks c _ accepted).1.1.2
  simp only [Checked.tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at tensorsAccepted
  have stride : c.host.abi.cumulativeStride=1 := by tauto
  constructor
  · intro packed i inside
    have bound := cumulative_bound cfg valid abi packed i inside
    have packed' : c.host.shape.varlen=Bool.true := packed
    have value : cfg.metadata.cumulative i=
        Physical.integer (bytes tensors heap) (.other 0) c.host.abi.cumulative64 i := by
      simp [cfg,Checked.configuration,Physical.originalConfiguration,Physical.canonicalMetadata,
        Physical.originalMetadata,packed',stride]
    rw [value] at bound
    change (Cell.mk .int (signed (bytes tensors heap) (.other 0) c.host.abi.cumulative64 i)).read .int=_
    rw [Cell.read_same,signed_eq _ _ _ _ bound]
    exact congrArg Int.ofNat value.symm
  · intro tensor i inside
    have bound := offset_bound cfg abi tensor i inside
    have tensor' : c.host.shape.offsetTensor=Bool.true := tensor
    have value : cfg.metadata.offsets i=
        Physical.integer (bytes tensors heap) .offsets c.host.abi.offsets64 i := by
      simp [cfg,Checked.configuration,Physical.originalConfiguration,Physical.canonicalMetadata,
        Physical.originalMetadata,tensor']
    rw [value] at bound
    change (Cell.mk .int (signed (bytes tensors heap) .offsets c.host.abi.offsets64 i)).read .int=_
    rw [Cell.read_same,signed_eq _ _ _ _ bound]
    exact congrArg Int.ofNat value.symm

#specNonCircular bytes avoiding [ProgramModel.result,source]
#axiomsClean load_pointer
#axiomsClean signed_eq
#axiomsClean lift_payload
#axiomsClean logical_read
#axiomsClean lift_metadata
end
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelTensor

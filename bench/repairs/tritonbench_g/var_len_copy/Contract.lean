import bench.repairs.tritonbench_g.var_len_copy.Kernels
import bench.repairs.tritonbench_g.var_len_copy.Proofs.Model
import VeriTile.Torch.Tensor
import VeriTile.Triton.Semantics.State
import VeriTile.Triton.Memory.Bounds

namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section

/-- Logical metadata and independent simultaneous initial-input copy target. -/
structure Request where
  count : Nat
  sourceSize : Nat
  outputSize : Nat
  segment : Nat → Segment

structure Valid (r : Request) : Prop where
  sourceBound : ∀ p, p < r.count → 0 < (r.segment p).len →
    0 ≤ (r.segment p).oldStart.toInt ∧ (r.segment p).old + (r.segment p).len ≤ r.sourceSize
  outputBound : ∀ p, p < r.count → 0 < (r.segment p).len →
    0 ≤ (r.segment p).newStart.toInt ∧ (r.segment p).dest + (r.segment p).len ≤ r.outputSize
  unique : ∀ p q i j, p < r.count → q < r.count → i < (r.segment p).len →
    j < (r.segment q).len → (r.segment p).dest+i=(r.segment q).dest+j → p=q ∧ i=j

def Touched (r : Request) (x : Nat) : Prop :=
  ∃ p i, p < r.count ∧ i < (r.segment p).len ∧ x=(r.segment p).dest+i

/-- No block size, loop counter, pointer or execution state occurs in this target. -/
def SegmentCopy {α : Type} (r : Request) (source before after : Nat → α) : Prop :=
  (∀ p i, p < r.count → i < (r.segment p).len →
    after ((r.segment p).dest+i)=source ((r.segment p).old+i)) ∧
  (∀ x, x<r.outputSize → ¬Touched r x → after x=before x)

/-- Ordinary backing bounds, disjoint output and actual signed-int32 metadata.
These are the caller buffers themselves; there is no snapshot allocation. -/
structure Prepared (l : Layout) (r : Request) (s : Memory) : Prop where
  counts : l.count=r.count
  sourceSize : l.sourceSize=r.sourceSize
  outputSize : l.outputSize=r.outputSize
  grid : l.count < 2^31
  oldCapacity : l.oldStarts+4*l.count ≤ 2^64
  lengthCapacity : l.lengths+4*l.count ≤ 2^64
  newCapacity : l.newStarts+4*l.count ≤ 2^64
  sourceCapacity : l.source+4*l.sourceSize ≤ 2^64
  outputCapacity : l.output+4*l.outputSize ≤ 2^64
  oldCells : ∀ p, p<l.count → s (addr l.oldStarts p)=(r.segment p).oldStart
  lengthCells : ∀ p, p<l.count → s (addr l.lengths p)=(r.segment p).length
  newCells : ∀ p, p<l.count → s (addr l.newStarts p)=(r.segment p).newStart
  disjoint : ∀ i, i<l.outputSize →
    (∀ j, j<l.sourceSize → addr l.output i ≠ addr l.source j) ∧
    (∀ p, p<l.count → addr l.output i ≠ addr l.oldStarts p ∧
      addr l.output i ≠ addr l.lengths p ∧ addr l.output i ≠ addr l.newStarts p)


inductive Role where | oldStarts | lengths | source | newStarts | output
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata

def request (m : Inputs) (segments : Nat → Segment) : Request :=
  ⟨(m .oldStarts).numel, (m .source).numel, (m .output).numel, segments⟩
def base (m : Inputs) (role : Role) : Nat := (m role).dataPtr.toNat

def directLayout (m : Inputs) : Layout :=
  ⟨base m .oldStarts,base m .lengths,base m .newStarts,base m .source,base m .output,
    (m .oldStarts).numel,(m .source).numel,(m .output).numel⟩

def DisjointOutput (m : Inputs) : Prop :=
  ∀ role, role ≠ .output →
    base m .output + 4*(m .output).numel ≤ base m role ∨
    base m role + 4*(m role).numel ≤ base m .output

def MetadataTypes (m : Inputs) : Prop :=
  (m .oldStarts).dtype = .other "int32" ∧ (m .lengths).dtype = .other "int32" ∧
  (m .newStarts).dtype = .other "int32" ∧
  (m .source).dtype = .floating .fp32 ∧ (m .output).dtype = .floating .fp32

/-- Separate nonempty output intervals exactly express unique segment writes. -/
def SegmentsDisjoint (r : Request) : Prop :=
  ∀ p q, p < r.count → q < r.count → p ≠ q →
    0 < (r.segment p).len → 0 < (r.segment q).len →
    (r.segment p).dest+(r.segment p).len ≤ (r.segment q).dest ∨
    (r.segment q).dest+(r.segment q).len ≤ (r.segment p).dest

structure InputContract (m : Inputs) (segments : Nat → Segment) : Prop where
  layout : ∀ role, (m role).layout = .strided
  rank : ∀ role, (m role).shape.length = 1
  dtype : MetadataTypes m
  cuda : ∀ role, (m role).device.kind = "cuda"
  device : ∀ role, (m role).device = (m .source).device
  contiguous : ∀ role, (m role).isContiguous = Bool.true
  noNeg : ∀ role, (m role).isNeg = Bool.false
  count : (m .oldStarts).shape = (m .lengths).shape ∧
    (m .oldStarts).shape = (m .newStarts).shape
  extent : (m .oldStarts).numel < 2^31
  alignment : ∀ role, 0 < (m role).numel → base m role % 4 = 0
  separation : DisjointOutput m
  sourceBound : ∀ p, p < (m .oldStarts).numel → 0 < (segments p).len →
    0 ≤ (segments p).oldStart.toInt ∧ (segments p).old+(segments p).len ≤ (m .source).numel
  outputBound : ∀ p, p < (m .oldStarts).numel → 0 < (segments p).len →
    0 ≤ (segments p).newStart.toInt ∧ (segments p).dest+(segments p).len ≤ (m .output).numel
  distinct : SegmentsDisjoint (request m segments)

inductive InputError where
  | tensors | layout | rank | dtype | cuda | device | contiguous | negative
  | count | extent | alignment | overlap | sourceBounds | outputBounds | destinations
  deriving DecidableEq, Repr

open Classical in
/-- Ordered interpretation of the actual Python checks; payloads are never copied. -/
def checkInputs (m : Inputs) (s : Nat → Segment) : Except InputError Unit :=
  if ¬(∀ r, (m r).layout = .strided) then .error .layout
  else if ¬(∀ r, (m r).shape.length = 1) then .error .rank
  else if ¬MetadataTypes m then .error .dtype
  else if ¬(∀ r, (m r).device.kind = "cuda") then .error .cuda
  else if ¬(∀ r, (m r).device = (m .source).device) then .error .device
  else if ¬(∀ r, (m r).isContiguous = Bool.true) then .error .contiguous
  else if ¬(∀ r, (m r).isNeg = Bool.false) then .error .negative
  else if ¬((m .oldStarts).shape = (m .lengths).shape ∧ (m .oldStarts).shape = (m .newStarts).shape) then .error .count
  else if ¬(m .oldStarts).numel < 2^31 then .error .extent
  else if ¬(∀ r, 0 < (m r).numel → base m r % 4 = 0) then .error .alignment
  else if ¬DisjointOutput m then .error .overlap
  else if ¬(∀ p, p < (m .oldStarts).numel → 0 < (s p).len →
    0 ≤ (s p).oldStart.toInt ∧ (s p).old+(s p).len ≤ (m .source).numel) then .error .sourceBounds
  else if ¬(∀ p, p < (m .oldStarts).numel → 0 < (s p).len →
    0 ≤ (s p).newStart.toInt ∧ (s p).dest+(s p).len ≤ (m .output).numel) then .error .outputBounds
  else if ¬SegmentsDisjoint (request m s) then .error .destinations
  else .ok ()

/-- None is the result of Python argument classification if any argument is
not a Tensor. It has no fabricated Tensor attributes. -/
def checkArguments (m : Option Inputs) (s : Nat → Segment) : Except InputError Unit :=
  match m with | none => .error .tensors | some m => checkInputs m s

def AdmittedArguments (m : Option Inputs) (s : Nat → Segment) : Prop :=
  match m with | none => False | some m => InputContract m s

/-- Initial signed metadata readback in the source kernel's logical regions.
The direct physical-pointer binding and capacities are proved in Proofs/Contract;
these pins select the same signed-int32 values for the Real interpretation. -/
structure KernelMetadataBinding (m : Inputs) (segments : Nat → Segment)
    (s : VeriTile.Triton.BlockState) : Prop where
  oldCells : ∀ p, p < (m .oldStarts).numel →
    s.readMemValue .int "old" p = (segments p).oldStart.toInt
  lengthCells : ∀ p, p < (m .oldStarts).numel →
    s.readMemValue .int "length" p = (segments p).length.toInt
  newCells : ∀ p, p < (m .oldStarts).numel →
    s.readMemValue .int "new" p = (segments p).newStart.toInt

/-- Logical extents of the same contiguous buffers. Equal metadata shapes
justify using the common count for all three signed metadata regions. -/
def kernelBounds (m : Inputs) : VeriTile.Triton.RegionBounds := fun r =>
  if r = "old" ∨ r = "length" ∨ r = "new" then (m .oldStarts).numel
  else if r = "source" then (m .source).numel
  else if r = "output" then (m .output).numel
  else 0

/-- One program's independent simultaneous copy formula. -/
def KernelCopy {α : Type} (a : Segment) (source after : Nat → α) : Prop :=
  ∀ i, i < a.len → after (a.dest+i) = source (a.old+i)

end
end VeriTile.Bench.Repairs.VarLenCopy

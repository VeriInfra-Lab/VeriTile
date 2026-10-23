import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Snapshot
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.LogicalReturn

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.CallerViews
open ReturnedStorage

structure InputView where
  base : Nat
  offset : Nat
  rowStride : Nat
  columnStride : Nat
  negative : Bool

structure WeightView where
  base : Nat
  offset : Nat
  slotStride : Nat
  rowStride : Nat
  columnStride : Nat
  negative : Bool
  /-- A 4D caller has singleton axis 1; squeeze preserves these three strides. -/
  singleton4D : Bool

structure IndexView where
  base : Nat
  offset : Nat
  stride : Nat
  negative : Bool

/-- Local typed storage reads at actual byte addresses. Different roles may
alias; no disjointness is required among caller tensors. -/
structure Memory (α : Type) where
  fp32 : Nat → α
  fp16 : Nat → α
  signed32 : Nat → BitVec 32

structure Views where
  input : InputView
  weight : WeightView
  indices : IndexView
  output : View

def inputAddress (v : InputView) (b k : Nat) : Nat :=
  v.base+4*(v.offset+b*v.rowStride+k*v.columnStride)
def weightAddress (v : WeightView) (a n k : Nat) : Nat :=
  v.base+2*(v.offset+a*v.slotStride+n*v.rowStride+k*v.columnStride)
def indexAddress (v : IndexView) (b : Nat) : Nat := v.base+4*(v.offset+b*v.stride)
def outputAddress (v : View) (b c : Nat) : Nat :=
  v.allocationBase+4*(v.storageOffset+b*v.rowStride+c*v.columnStride)

def logical {α : Type} (codec : LogicalReturn.Codec α) (views : Views) (memory : Memory α) : Snapshot.Logical α where
  input := fun b k => codec.decode views.input.negative (memory.fp32 (inputAddress views.input b k))
  weight := fun a n k => codec.decode views.weight.negative (memory.fp16 (weightAddress views.weight a n k))
  output := fun b c => codec.decode views.output.negative (memory.fp32 (outputAddress views.output b c))
  indices := fun b => (if views.indices.negative then -(memory.signed32 (indexAddress views.indices b))
    else memory.signed32 (indexAddress views.indices b)).toInt

theorem compact_metadata {α : Type} (codec : LogicalReturn.Codec α) (views : Views)
    (memory : Memory α) (h : Call) (zero : α) (packed : Snapshot.Packed α h (logical codec views memory) zero)
    (b : Nat) (inside : b<h.batches) :
    packed.indices b=(if views.indices.negative then
      -(memory.signed32 (views.indices.base+4*(views.indices.offset+b*views.indices.stride)))
      else memory.signed32 (views.indices.base+4*(views.indices.offset+b*views.indices.stride))).toInt :=
  packed.metadataCopy b inside

theorem supplied_output {α : Type} (codec : LogicalReturn.Codec α) (views : Views)
    (memory : Memory α) (i : Coord views.output) :
    (logical codec views memory).output i.1.val i.2.val=
      codec.decode views.output.negative (memory.fp32 (views.output.address i ⟨0,by decide⟩)) := by
  simp [logical,outputAddress,View.address,View.offset]

/-- Actual backing extents for the three caller read roles. Aliased roles may
name the same physical interval. The output extent is its original descriptor. -/
structure Allocations (views : Views) where
  inputBytes : Nat
  weightBytes : Nat
  indexBytes : Nat
  inputWindow : views.input.base+inputBytes≤2^64
  weightWindow : views.weight.base+weightBytes≤2^64
  indexWindow : views.indices.base+indexBytes≤2^64

def Owned (views : Views) (a : Allocations views) (address : Nat) : Prop :=
  (views.input.base≤address ∧ address<views.input.base+a.inputBytes) ∨
  (views.weight.base≤address ∧ address<views.weight.base+a.weightBytes) ∨
  (views.indices.base≤address ∧ address<views.indices.base+a.indexBytes) ∨
  (views.output.allocationBase≤address ∧ address<views.output.allocationBase+views.output.allocationBytes)

/-- Ordinary valid-view and wrapper shape facts for this same call. These
restrict no tensor dimensions beyond the actual logical shapes and backing. -/
structure Valid (h : Call) (views : Views) (a : Allocations views) : Prop where
  batches : h.batches≤views.output.rows
  columns : h.columns=views.output.columns
  input : ∀ b,b<h.batches → ∀ k,k<h.K →
    inputAddress views.input b k+4≤views.input.base+a.inputBytes
  weights : ∀ slot,slot<h.adapters → ∀ n,n<h.N → ∀ k,k<h.K →
    weightAddress views.weight slot n k+2≤views.weight.base+a.weightBytes
  indices : ∀ b,b<h.batches → indexAddress views.indices b+4≤views.indices.base+a.indexBytes
  output : views.output.Backed

/-- Local association of the pre-call output bytes with the same typed logical
values consumed by ADD_INPUTS. Encoding includes the original negative flag. -/
def InitialOutput {α : Type} (codec : LogicalReturn.Codec α) (encode : Encode α)
    (views : Views) (memory : Memory α) (before : Bytes) : Prop :=
  ∀ i : Coord views.output,∀ byte : Fin 4,
    before views.output.owner (views.output.address i byte)=
      encode views.output.negative ((logical codec views memory).output i.1.val i.2.val) byte


#axiomsClean compact_metadata
#axiomsClean supplied_output
end VeriTile.Bench.Repairs.BgmvExpandSlice.CallerViews

import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Memory

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Native
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000

/-- Actual dense clone/empty storage. Capacity is bytes, and neither the base
nor the storage extent is required to be divisible by the element width. -/
structure Allocation (n width : Nat) where
  base : Nat
  storageBase : Nat
  capacity : Nat
  positive : 0<width
  lower : storageBase≤base
  upper : base+n*width≤storageBase+capacity
  native : storageBase+capacity≤2^64

def Allocation.Contains (a : Allocation n width) (q : Nat) : Prop :=
  a.storageBase≤q ∧ q<a.storageBase+a.capacity

def Allocation.Active (a : Allocation n width) (q : Nat) : Prop :=
  a.base≤q ∧ q<a.base+n*width

theorem byte_bounds (a : Allocation n width) (i k : Nat) (active : i<n) (byte : k<width) :
    a.Contains (Memory.address a.base width i+k) ∧ Memory.address a.base width i+k<2^64 := by
  have product := Nat.mul_le_mul_right width (show i+1≤n by omega)
  have lower := a.lower
  have upper := a.upper
  have native := a.native
  unfold Allocation.Contains Memory.address
  constructor
  · constructor <;> nlinarith
  · nlinarith

def Separate (x : Allocation n iw) (y : Allocation n ow) : Prop :=
  y.storageBase+y.capacity≤x.storageBase ∨ x.storageBase+x.capacity≤y.storageBase

theorem separate_accesses (x : Allocation n iw) (y : Allocation n ow) (apart : Separate x y)
    (i j : Nat) (hi : i<n) (hj : j<n) :
    SpanDisjoint (Memory.address y.base ow i) (Memory.address y.base ow i+ow)
      (Memory.address x.base iw j) (Memory.address x.base iw j+iw) := by
  have ix := Nat.mul_le_mul_right iw (show j+1≤n by omega)
  have iy := Nat.mul_le_mul_right ow (show i+1≤n by omega)
  have xl := x.lower
  have xu := x.upper
  have yl := y.lower
  have yu := y.upper
  unfold Separate at apart
  unfold SpanDisjoint Memory.address
  rcases apart with h|h
  · left
    nlinarith
  · right
    nlinarith

/-- Actual finite signed-i32 offsets and masks of the unchanged JIT. Chunking
provides the bound, with no global-length guard and no padded allocation. -/
theorem indices (n c pid lane : Nat)
    (launched : pid<NativeIndex.grid (Chunking.chunkLength n c)) (inTile : lane<1024) :
    (NativeIndex.offset pid lane).toInt=(pid*1024+lane : Nat) ∧
    NativeIndex.mask (Chunking.chunkLength n c) pid lane=decide (pid*1024+lane<Chunking.chunkLength n c) :=
  Chunking.finite_indices n c pid lane launched inTile

theorem address_shift (base width start i : Nat) :
    base+start*width+i*width=Memory.address base width (start+i) := by
  unfold Memory.address
  ring

theorem pointer_at (base width start count pid lane : Nat)
    (small : count<2^31) (launched : pid<NativeIndex.grid count) (inTile : lane<1024)
    (bounded : Memory.address base width (start+(pid*1024+lane))<2^64) :
    NativeIndex.pointer (base+start*width) width pid lane=
      Memory.address base width (start+(pid*1024+lane)) := by
  have sum := address_shift base width start (pid*1024+lane)
  have fits : base+start*width+(pid*1024+lane)*width<2^64 := by
    rw [sum]
    exact bounded
  exact (Chunking.pointer_bounded count pid lane (base+start*width) width
    small launched inTile fits).trans sum

/-- A source lane's actual slice start/count are symbolic here. The host
instantiates start/count with chunkStart/chunkLength and obtains globalActive
from Chunking.chunk_active, rather than imposing any caller length bound. -/
theorem pointer (a : Allocation n width) (start count pid lane : Nat)
    (small : count<2^31) (launched : pid<NativeIndex.grid count) (inTile : lane<1024)
    (globalActive : start+(pid*1024+lane)<n) :
    NativeIndex.pointer (a.base+start*width) width pid lane=
      Memory.address a.base width (start+(pid*1024+lane)) := by
  apply pointer_at a.base width start count pid lane small launched inTile
  simpa only [Nat.add_zero] using (byte_bounds a _ 0 globalActive a.positive).2

theorem all_bytes (a : Allocation n width) (start count pid lane byte : Nat)
    (small : count<2^31) (launched : pid<NativeIndex.grid count) (inTile : lane<1024)
    (globalActive : start+(pid*1024+lane)<n) (inByte : byte<width) :
    a.Contains (NativeIndex.pointer (a.base+start*width) width pid lane+byte) ∧
      NativeIndex.pointer (a.base+start*width) width pid lane+byte<2^64 := by
  rw [pointer a start count pid lane small launched inTile globalActive]
  exact byte_bounds a _ byte globalActive inByte

theorem full_execution {M A B : Type} (ops : Memory.Primitives M A B iw ow)
    (twice : Bool) (compute : A → B) (x : Allocation n iw) (y : Allocation n ow)
    (apart : Separate x y) (before : M) :
    (∀ i,i<n → ops.readOutput (Memory.run ops twice compute x.base y.base (Chunking.batches n) before)
      (Memory.address y.base ow i)=compute (ops.readInput before (Memory.address x.base iw i))) ∧
    (∀ q,¬y.Active q →
      ops.byte (Memory.run ops twice compute x.base y.base (Chunking.batches n) before) q=ops.byte before q) := by
  have complete := Memory.complete ops twice compute x.base y.base n (Chunking.batches n) before
    (Chunking.full_coverage n) (fun i hi j hj => separate_accesses x y apart i j hi hj)
  refine ⟨complete.1,?_⟩
  intro q outside
  apply complete.2
  intro i hi
  have bound := Nat.mul_le_mul_right ow (show i+1≤n by omega)
  unfold Allocation.Active at outside
  unfold Memory.address
  by_cases less : q<y.base
  · left
    omega
  · right
    have : y.base+n*ow≤q := by omega
    nlinarith

#axiomsClean indices
#axiomsClean pointer
#axiomsClean all_bytes
#axiomsClean full_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Native

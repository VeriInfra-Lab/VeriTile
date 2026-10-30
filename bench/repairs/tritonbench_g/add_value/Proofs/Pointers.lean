import bench.repairs.tritonbench_g.add_value.Proofs.BindingIndex
import bench.tritonbench_g.add_example.Proofs.FinitePhysical
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.AddValue.Contiguous
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite (bytes SpanDisjoint)
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Source
namespace Native
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

inductive Slot | x | out deriving DecidableEq, Repr

def Slot.region : Slot → RegionName
  | .x => "x" | .out => "out"

export VeriTile.Torch.FP32 (Storage DataView)
export VeriTile.Torch.FP32.DataView (address)

structure Layout (n : Nat) where
  window : Slot → DataView
  capacity : ∀ r, n ≤ (window r).elements
  aligned : ∀ r, (window r).base % 4 = 0
  freshX : (window .x).base + 4*(window .x).elements ≤ (window .out).base ∨
    (window .out).base + 4*(window .out).elements ≤ (window .x).base

def view (a : DataView) (start : Nat) : PhysicalView where
  pointer := BitVec.ofNat 64 (address a start)
  base := a.base
  deviceBase := a.base
  toBacking := id
  capacity := 4*a.elements
  dtype := .fp32

def views (L : Layout n) (start : Nat) (r : RegionName) : PhysicalView :=
  if r="x" then view (L.window .x) start
  else view (L.window .out) start

@[simp] theorem views_slot (L : Layout n) (start : Nat) (r : Slot) :
    views L start r.region = view (L.window r) start := by
  cases r <;> simp [views,Slot.region]

theorem offset_word (n : Nat) (job : Nat) (valid : Index.ValidJob n job)
    (lane : Nat) (hl : lane < 1024) :
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane).signExtend 64 = BitVec.ofNat 64 (job*1024+lane) := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),(Index.source_abi n job valid lane hl).2.1]
  have bound := valid.1
  have hp := valid.2
  have hg : gridCount n ≤ 2^21 := by unfold gridCount; omega
  have hi : job*1024+lane < 2^63 := by norm_num at bound ⊢; omega
  have hn : (BitVec.ofNat 64 (job*1024+lane)).toNat=job*1024+lane := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem pointer_exact (L : Layout n) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat)
    (hl : lane ∈ Source.programLanes n job) :
    (view (L.window r) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane) =
      address (L.window r) (job*1024+lane) := by
  have ha := (Index.active_iff n job valid lane).mp hl
  have hi := Index.active_bound n job valid lane hl
  have cap := L.capacity r
  have span := (L.window r).spanFits
  have bound : address (L.window r) (job*1024+lane) < 2^64 := by
    unfold address
    omega
  simp only [PhysicalView.nativeAt,view,bytes]
  rw [offset_word n job valid lane ha.1,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  have eq : address (L.window r) 0 + (job*1024+lane)*4 =
      address (L.window r) (job*1024+lane) := by
    simp [address]
    omega
  rw [eq,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

theorem view_backing (a : DataView) (start : Nat) : (view a start).toBacking=id := rfl

theorem lane_exact (L : Layout n) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat)
    (hl : lane ∈ Source.programLanes n job) :
    Source.laneAddress (view (L.window r) 0) job lane =
      address (L.window r) (job*1024+lane) := by
  change (view (L.window r) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane) = _
  exact pointer_exact L job valid r lane hl

theorem inside (L : Layout n) (r : Slot) (i : Nat) (hi : i < n) (start : Nat) :
    (view (L.window r) start).Inside (address (L.window r) i) := by
  have cap := L.capacity r
  have span := (L.window r).spanFits
  simp only [PhysicalView.Inside,view,address,bytes]
  omega

/-- Only ordinary allocation access permissions and same-cell typed FP32
load/store are supplied. No address, execution, grid or result law is a field. -/
structure LocalMemory {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n) : Prop where
  readable : ∀ r k, k < 4*(L.window r).elements →
    ops.readableByte ops.launchDevice ((L.window r).base+k)
  writable : ∀ k, k < 4*(L.window .out).elements →
    ops.writableByte ops.launchDevice ((L.window .out).base+k)
  read_write : ∀ s a v, ops.read (ops.write s .fp32 a v) .fp32 a = v

theorem read_at_ok {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n)
    (memAPI : LocalMemory ops L) (r : Slot) (i : Nat) (hi : i<n) (start : Nat) :
    ReadOK ops (view (L.window r) start) (address (L.window r) i) := by
  have cap := L.capacity r
  have span := (L.window r).spanFits
  have align := L.aligned r
  simp only [ReadOK,view,id_eq,bytes]
  refine ⟨inside L r i hi start,?_,?_,?_,?_,?_⟩
  · simpa [address,Nat.add_mod] using align
  · simp [address]
  · unfold address; omega
  · unfold address; omega
  · intro k hk
    constructor
    · have bound : 4*i+k < 4*(L.window r).elements := by omega
      simpa [address,Nat.add_assoc] using memAPI.readable r _ bound
    · trivial

theorem read_ok {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n)
    (memAPI : LocalMemory ops L) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat) (hl : lane ∈ Source.programLanes n job) :
    ReadOK ops (view (L.window r) 0)
      ((view (L.window r) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane)) := by
  rw [pointer_exact L job valid r lane hl]
  exact read_at_ok ops L memAPI r _ (Index.active_bound n job valid lane hl) _

theorem write_ok {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n)
    (memAPI : LocalMemory ops L) (job : Nat) (valid : Index.ValidJob n job)
    (lane : Nat) (hl : lane ∈ Source.programLanes n job) :
    WriteOK ops (view (L.window .out) 0)
      ((view (L.window .out) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane)) := by
  have h := read_ok ops L memAPI job valid .out lane hl
  refine ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,?_⟩
  intro k hk
  refine ⟨?_,(h.2.2.2.2.2 k hk).2⟩
  rw [pointer_exact L job valid .out lane hl]
  have hi := Index.active_bound n job valid lane hl
  have cap := L.capacity .out
  change k<4 at hk
  have bound : 4 * (job*1024+lane)+k < 4*(L.window .out).elements := by omega
  simpa [address,Nat.add_assoc] using memAPI.writable _ bound

theorem output_disjoint_x (L : Layout n) (start : Nat) :
    (view (L.window .x) start).Disjoint (view (L.window .out) 0) := L.freshX

#axiomsClean pointer_exact
#axiomsClean read_ok
#axiomsClean write_ok
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Native
end VeriTile.Bench.Repairs.AddValue.Contiguous

import bench.repairs.tritonbench_g.vector_addition.Proofs.BindingIndex
import bench.tritonbench_g.add_example.Proofs.FinitePhysical
import bench.repairs.tritonbench_g.add_example.Proofs.Pointers

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite (bytes SpanDisjoint)
open VeriTile.Bench.TritonBenchG.AddExample.Finite
namespace Native
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

export VeriTile.Bench.Repairs.AddExample.Contiguous.Native
  (Slot Storage DataView address Layout view views views_slot LocalMemory
   inside read_at_ok output_disjoint_x output_disjoint_y)

def nativeLane (v : PhysicalView) (pid lane : Nat) : Nat :=
  v.nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane)

def lane (v : PhysicalView) (pid lane : Nat) : Nat :=
  v.at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane)

theorem offset_word (n job : Nat) (valid : Index.ValidJob n job)
    (lane : Nat) (hl : lane < 1024) :
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane).signExtend 64 = BitVec.ofNat 64 (job*1024+lane) :=
  VeriTile.Bench.TritonBenchG.VectorAddition.Finite.source_index_sign_extend n job lane valid.1 valid.2 hl

theorem pointer_exact (L : Layout n) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat)
    (hl : lane ∈ FiniteSource.programLanes n job) :
    nativeLane (view (L.window r) 0) job lane =
      address (L.window r) (job*1024+lane) := by
  have ha := (Index.active_iff n job valid lane).mp hl
  have hi := Index.active_bound n job valid lane hl
  have cap := L.capacity r
  have span := (L.window r).spanFits
  have bound : address (L.window r) (job*1024+lane) < 2^64 := by
    unfold address
    omega
  simp only [nativeLane,PhysicalView.nativeAt,view,bytes]
  rw [offset_word n job valid lane ha.1,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  have eq : address (L.window r) 0 + (job*1024+lane)*4 =
      address (L.window r) (job*1024+lane) := by
    simp [address]
    omega
  rw [eq,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

theorem lane_exact (L : Layout n) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat) (hl : lane ∈ FiniteSource.programLanes n job) :
    (view (L.window r) 0).at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane) =
      address (L.window r) (job*1024+lane) :=
  pointer_exact L job valid r lane hl

theorem read_ok {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n)
    (memAPI : LocalMemory ops L) (job : Nat) (valid : Index.ValidJob n job)
    (r : Slot) (lane : Nat) (hl : lane ∈ FiniteSource.programLanes n job) :
    ReadOK ops (view (L.window r) 0)
      (nativeLane (view (L.window r) 0) job lane) := by
  rw [pointer_exact L job valid r lane hl]
  exact read_at_ok ops L memAPI r _ (Index.active_bound n job valid lane hl) _

theorem write_ok {M A : Type} (ops : MemoryPrimitives M A) (L : Layout n)
    (memAPI : LocalMemory ops L) (job : Nat) (valid : Index.ValidJob n job)
    (lane : Nat) (hl : lane ∈ FiniteSource.programLanes n job) :
    WriteOK ops (view (L.window .out) 0)
      (nativeLane (view (L.window .out) 0) job lane) := by
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

#axiomsClean pointer_exact
#axiomsClean read_ok
#axiomsClean write_ok
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Native
end VeriTile.Bench.Repairs.VectorAddition.Contiguous

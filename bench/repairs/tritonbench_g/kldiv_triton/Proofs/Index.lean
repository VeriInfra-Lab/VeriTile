import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Math
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Integer

namespace VeriTile.Bench.Repairs.KldivTriton.Index
open L2NormBwd.Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Index (signed_ofNat)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds lt_rounds_iff)
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Torch contiguous() may retain any stride on empty tensors or singleton
columns. Only active coordinates need the kernel's unit-column address. -/
def UnitColumns (m : Matrix) : Prop :=
  m.rows = 0 ∨ m.cols ≤ 1 ∨ m.colStride = 1

theorem UnitColumns.of_stride {m : Matrix} (h : m.colStride = 1) : UnitColumns m :=
  Or.inr (Or.inr h)

theorem UnitColumns.active {m : Matrix} (h : UnitColumns m) (r c : Nat)
    (hr : r < m.rows) (hc : c < m.cols) : c * m.colStride = c := by
  rcases h with empty | singleton | unit
  · omega
  · have : c = 0 := by omega
    simp [this]
  · simp [unit]

/-- A nonempty prepared row's signed column bound follows from its actual
four-byte backing. This does not impose a new native-safe shape cap. -/
theorem columns_bound (m : Matrix) (backed : m.backed) (unit : UnitColumns m)
    (hasRow : 0 < m.rows) : m.cols ≤ 2^62 := by
  by_cases empty : m.cols = 0
  · simp [empty]
  have last := backed 0 (m.cols - 1) hasRow (by omega)
  have column := unit.active 0 (m.cols - 1) hasRow (by omega)
  have capacity := m.view.storage.representable
  simp only [Matrix.key, Matrix.index, Allocation.key, Allocation.contains, Nat.zero_mul,
    Nat.add_zero, column] at last
  omega

/-- The lane sum and the terminating loop increment both fit signed64.
Inactive lanes need no storage bounds and are never dereferenced. -/
theorem loop_bounds (N B k lane : Nat) (positive : 0 < B) (block : B ≤ 16384)
    (columns : N ≤ 2^62) (chunk : k < rounds N B) (hlane : lane < B) :
    k*B < 2^63 ∧ k*B+lane < 2^63 ∧ (k+1)*B < 2^63 ∧ N < 2^63 := by
  have active := (lt_rounds_iff N B k positive).mp chunk
  simp only [Nat.add_mul, Nat.one_mul]
  omega

theorem loop_mask_exact (N B k lane : Nat) (positive : 0 < B) (block : B ≤ 16384)
    (columns : N ≤ 2^62) (chunk : k < rounds N B) (hlane : lane < B) :
    (BitVec.ofNat 64 (k*B) + BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 N) =
      decide (k*B+lane < N) := by
  obtain ⟨_,sum,_,width⟩ := loop_bounds N B k lane positive block columns chunk hlane
  rw [←BitVec.ofNat_add]
  change decide ((BitVec.ofNat 64 (k*B+lane)).toInt < (BitVec.ofNat 64 N).toInt) = _
  rw [signed_ofNat 64 _ (by decide) sum, signed_ofNat 64 _ (by decide) width]
  simp only [Int.ofNat_lt]

theorem loop_increment_exact (N B k : Nat) (positive : 0 < B) (block : B ≤ 16384)
    (columns : N ≤ 2^62) (chunk : k < rounds N B) :
    (BitVec.ofNat 64 (k*B) + BitVec.ofNat 64 B).toInt = (((k+1)*B : Nat) : Int) := by
  obtain ⟨_,_,next,_⟩ := loop_bounds N B k 0 positive block columns chunk positive
  rw [←BitVec.ofNat_add, show k*B+B=(k+1)*B by ring]
  exact signed_ofNat 64 _ (by decide) next

/-- Reuse the already checked physical pointer, PID, stride-product and all
four active-byte bounds. The global launcher boundary remains explicit. -/
theorem active_pointer (m : Matrix) (metadata : L2NormBwd.Integer.Metadata m)
    (backed : m.backed) (grid : L2NormBwd.Integer.GridCompatible m.rows)
    (r c : Nat) (hr : r < m.rows) (hc : c < m.cols) :
    (L2NormBwd.Integer.pointer m r c).toNat = (m.key r c).address ∧
    ∀ byte, byte < 4 →
      (L2NormBwd.Integer.pointer m r c).toNat + byte < m.view.storage.baseByte + m.view.storage.bytes ∧
      (L2NormBwd.Integer.pointer m r c).toNat + byte < 2^64 := by
  have h := L2NormBwd.Integer.active_no_wrap m metadata backed grid r c hr hc
  exact ⟨h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩

#axiomsClean UnitColumns.active
#axiomsClean columns_bound
#axiomsClean loop_mask_exact
#axiomsClean loop_increment_exact
#axiomsClean active_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton.Index

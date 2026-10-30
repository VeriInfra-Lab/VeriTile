import bench.tritonbench_g.rope_embedding.Proofs.Kernel
namespace VeriTile.Bench.Repairs.RopeEmbedding.KernelAccess
open VeriTile.Triton
set_option maxHeartbeats 2000000

theorem formed_bounds (r : Request) (row head lane : Nat)
    (hr : row < r.rows) (hh : head < r.heads) (hl : lane < r.block) :
    r.size≠0 ∧ r.half≠0 ∧ r.offset row head (lane+r.half) ≤ outputLast r ∧
      row%r.sequence*r.half+lane ≤ tableLast r := by
  have half : r.half≠0 := by intro h; simp [Request.block,h] at hl
  have dimension : 0 < r.dimension := by unfold Request.half at half; omega
  have size : r.size≠0 := by
    unfold Request.size
    exact Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos (by omega) (by omega)) dimension)
  have sequence : 0 < r.sequence := by
    by_contra hn
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at hr
  have hm := Nat.mod_lt row sequence
  have ho1 := Nat.mul_le_mul_right r.heads (show row ≤ r.rows-1 by omega)
  have ho2 := Nat.mul_le_mul_right r.dimension
    (show row*r.heads+head ≤ (r.rows-1)*r.heads+(r.heads-1) by omega)
  have ht := Nat.mul_le_mul_right r.half (show row%r.sequence ≤ r.sequence-1 by omega)
  refine ⟨size,half,?_,?_⟩
  · unfold Request.offset outputLast; omega
  · unfold tableLast; omega

/-- All formed offsets, including masked padding lanes, fit signed i32;
all formed addresses fit pointer width. Active offsets additionally fit the
logical region. The source arithmetic never requires a wrapped address. -/
theorem checked_offsets (m : KernelMetadata) (accepted : m.check=Bool.true)
    (row head lane : Nat) (hr : row < m.request.rows) (hh : head < m.request.heads)
    (hl : lane < m.request.block) :
    m.request.offset row head (lane+m.request.half) < 2^31 ∧
    row%m.request.sequence*m.request.half+lane < 2^31 ∧
    m.q.dataPtr.toNat+4*m.request.offset row head (lane+m.request.half)+4 ≤ 2^64 ∧
    m.cosine.dataPtr.toNat+4*(row%m.request.sequence*m.request.half+lane)+4 ≤ 2^64 ∧
    m.sine.dataPtr.toNat+4*(row%m.request.sequence*m.request.half+lane)+4 ≤ 2^64 := by
  have hb := formed_bounds m.request row head lane hr hh hl
  have guard : m.request.size≠0 ∧ m.request.half≠0 →
      outputLast m.request < 2^31 ∧ tableLast m.request < 2^31 ∧
      m.q.dataPtr.toNat+4*outputLast m.request+4 ≤ 2^64 ∧
      m.cosine.dataPtr.toNat+4*tableLast m.request+4 ≤ 2^64 ∧
      m.sine.dataPtr.toNat+4*tableLast m.request+4 ≤ 2^64 := by
    simp only [KernelMetadata.check,Bool.and_eq_true,decide_eq_true_eq] at accepted
    tauto
  have hg := guard ⟨hb.1,hb.2.1⟩
  omega

/-- The natural index equations are the exact signed-i32 interpretation on
the checked interval, not an assumption of unbounded machine integers. -/
theorem signed32 (n : Nat) (h : n < 2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  have hn : n < 2^32 := by omega
  have hh : 2*n < 2^32 := by omega
  rw [BitVec.toInt_eq_toNat_cond]
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt hn,if_pos hh]

theorem group_bound (m : KernelMetadata) (accepted : m.check=Bool.true)
    (group : Nat) (hg : group < m.request.groups) :
    group*4 < m.request.heads ∧ group*4+4 < 2^31 := by
  have gh := group_head m.request group hg
  have metadata : m.request.heads+3 < 2^31 := by
    simp only [KernelMetadata.check,Bool.and_eq_true,decide_eq_true_eq] at accepted
    tauto
  constructor
  · exact gh
  · omega

#axiomsClean checked_offsets
#axiomsClean signed32
#axiomsClean group_bound
end VeriTile.Bench.Repairs.RopeEmbedding.KernelAccess

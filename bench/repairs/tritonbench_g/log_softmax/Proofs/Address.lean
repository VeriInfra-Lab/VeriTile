import bench.repairs.tritonbench_g.log_softmax.Proofs.Graph
import bench.tritonbench_g.softmax_flaggems.Proofs.RealAddress
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

/-! The actual dense FP32 reply bounds all widened native intermediates.
Clamping precedes multiplication even in masked lanes. Runtime grid product
and launch segmentation are separate ordinary runtime obligations. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Address
open VeriTile.Bench.Repairs.LayerNormFwd.Address
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

abbrev Coord (M N K : Nat) := Fin M × Fin N × Fin K

def offset (N K m n k : Nat) : Nat := m*N*K+n*K+k

def linear (q : Coord M N K) : Nat := offset N K q.1.val q.2.1.val q.2.2.val

structure Dense (count : Nat) where
  base : Nat
  finite : base+count*4≤2^64
  aligned : base%4=0

def Dense.address (a : Dense count) (i : Nat) : Nat := a.base+i*4

def Dense.contains (a : Dense count) (i : Nat) : Prop := a.base ≤ i ∧ i < a.base+count*4

theorem count_bound (a : Dense count) : count≤2^62 := by
  have h:=a.finite
  omega

theorem linear_bound (q : Coord M N K) : linear q<M*N*K :=
  VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
    q.1.isLt q.2.1.isLt q.2.2.isLt

theorem linear_injective : Function.Injective (@linear M N K) := by
  intro a b h
  have h' : (a.1.val*N+a.2.1.val)*K+a.2.2.val=
      (b.1.val*N+b.2.1.val)*K+b.2.2.val := by
    simpa [linear,offset,Nat.add_mul] using h
  obtain ⟨hmn,hk⟩:=VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.row_column_injective
    a.2.2.isLt b.2.2.isLt h'
  obtain ⟨hm,hn⟩:=VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.row_column_injective
    a.2.1.isLt b.2.1.isLt hmn
  exact Prod.ext (Fin.ext hm) (Prod.ext (Fin.ext hn) (Fin.ext hk))

theorem address_bytes (a : Dense count) (i : Nat) (hi : i<count) (b : Nat) (hb : b<4) :
    a.contains (a.address i+b) ∧ a.address i+b<2^64 := by
  have h:=a.finite
  simp only [Dense.contains,Dense.address]
  omega

theorem address_apart (a : Dense count) {i j : Nat} (h : i≠j) :
    Apart (a.address i) (a.address j) := by
  unfold Apart Dense.address
  omega

theorem coordinate_apart (a : Dense (M*N*K)) {p q : Coord M N K} (h : p≠q) :
    Apart (a.address (linear p)) (a.address (linear q)) :=
  address_apart a (fun he=>h (linear_injective he))

def safe (raw size : Nat) : Nat := min raw (size-1)

theorem safe_bound (positive : 0<size) : safe raw size<size := by
  have h:=Nat.min_le_right raw (size-1)
  unfold safe
  omega

theorem safe_active (active : raw<size) : safe raw size=raw := by
  unfold safe
  exact Nat.min_eq_left (by omega)

theorem masked_offset_bound (hm : 0<M) (hn : 0<N) (hk : 0<K) (m n k : Nat) :
    offset N K (safe m M) (safe n N) (safe k K)<M*N*K :=
  VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
    (safe_bound hm) (safe_bound hn) (safe_bound hk)

def machineOffset (N K m n k : Nat) : Nat :=
  add64 (add64 (mul64 (mul64 (wrap64 m) (wrap64 N)) (wrap64 K))
    (mul64 (wrap64 n) (wrap64 K))) (wrap64 k)

theorem machine_mod (N K m n k : Nat) :
    machineOffset N K m n k=offset N K m n k%2^64 := by
  simp [machineOffset,offset,add64,mul64,wrap64,Nat.add_mod,Nat.mul_mod]

/-- Each sequential product and addition in the source is exact, including
masked lanes. The count bound is derived from the actual allocated FP32 bytes. -/
theorem all_intermediates (a : Dense (M*N*K)) (hk : 0<K)
    (bound : offset N K m n k<M*N*K) :
    signed64 (m*N)=(m*N:Nat) ∧ signed64 (m*N*K)=(m*N*K:Nat) ∧
    signed64 (n*K)=(n*K:Nat) ∧ signed64 (m*N*K+n*K)=(m*N*K+n*K:Nat) ∧
    signed64 (offset N K m n k)=(offset N K m n k:Nat) ∧
    machineOffset N K m n k=offset N K m n k := by
  have count:=count_bound a
  have first : m*N≤m*N*K:=Nat.le_mul_of_pos_right _ hk
  have small : offset N K m n k<2^63 := by omega
  have expanded:=bound
  unfold offset at expanded
  refine ⟨signed64_exact (by omega),signed64_exact (by omega),
    signed64_exact (by omega),signed64_exact (by omega),signed64_exact small,?_⟩
  rw [machine_mod,Nat.mod_eq_of_lt (lt_trans small (by norm_num))]

theorem inactive_intermediates (a : Dense (M*N*K)) (hm : 0<M) (hn : 0<N) (hk : 0<K)
    (m n k : Nat) :
    machineOffset N K (safe m M) (safe n N) (safe k K)=
      offset N K (safe m M) (safe n N) (safe k K) ∧
    a.address (offset N K (safe m M) (safe n N) (safe k K))+3<2^64 := by
  have bound:=masked_offset_bound hm hn hk m n k
  exact ⟨(all_intermediates a hk bound).2.2.2.2.2,
    (address_bytes a _ bound 3 (by omega)).2⟩

/-- Hardware CTA identifiers are nonnegative signed-i32 values; all source
BLOCK_M choices are retained. The widened raw row and arange fit signed-i64. -/
theorem raw_row_exact (pid block lane : Nat) (hp : pid<2^31) (hb : block≤8)
    (hl : lane<block) :
    pid*block+lane<2^34 ∧ signed64 (pid*block+lane)=(pid*block+lane:Nat) := by
  have hmul:=Nat.mul_le_mul_left pid hb
  have bound : pid*block+lane<2^34 := by omega
  exact ⟨bound,signed64_exact (lt_trans bound (by norm_num))⟩

#axiomsClean linear_injective
#axiomsClean all_intermediates
#axiomsClean inactive_intermediates
#axiomsClean raw_row_exact
end
end VeriTile.Bench.Repairs.LogSoftmax.Address

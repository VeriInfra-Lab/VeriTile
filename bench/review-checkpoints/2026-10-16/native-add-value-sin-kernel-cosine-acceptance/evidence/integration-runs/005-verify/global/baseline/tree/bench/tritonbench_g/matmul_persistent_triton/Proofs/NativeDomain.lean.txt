import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopState

/-! Sufficient arithmetic domain for the signed-i32 source specialization.
These are explicit input/configuration inequalities, not assumptions of a
successful run or of whole-source correctness. Bounds cover masked-off lane
addresses too. Native compiler admission and pointer allocation are separate. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeDomain
open VeriTile.Triton
set_option maxHeartbeats 4000000

def limit : Nat := 2147483648
def rows (p : Parameters) := RealMath.blocks p.M p.BM
def columns (p : Parameters) := RealMath.blocks p.N p.BN
def count (p : Parameters) := RealMath.blocks p.K p.BK
def total (p : Parameters) := rows p * columns p
def rowPad (p : Parameters) := rows p * p.BM
def columnPad (p : Parameters) := columns p * p.BN
def kPad (p : Parameters) := count p * p.BK
def work (p : Parameters) (pid : Nat) := Schedule.tilesFor (total p) p.SMS pid
def trip (p : Parameters) (pid : Nat) := count p * work p pid
def grid (p : Parameters) := min p.SMS (total p)

/-- A readable sufficient no-overflow domain for all evaluated source integer
expressions. Nonnegative strides match the current mathematical Parameters.
The output bound is deliberately sufficient: native C uses two addptrs, not
an i32 addition of the two products. K=0 is included. -/
structure Admissible (p : Parameters) : Prop where
  m : 0 < p.M
  n : 0 < p.N
  bm : 0 < p.BM
  bn : 0 < p.BN
  bk : 0 < p.BK
  gm : 0 < p.GM
  sms : 0 < p.SMS
  scalar : ∀ x ∈ [p.M,p.N,p.K,p.BM,p.BN,p.BK,p.GM,p.SMS,
      p.sam,p.sak,p.sbk,p.sbn,p.scm,p.scn], x < limit
  ceil_m : p.M + p.BM - 1 < limit
  ceil_n : p.N + p.BN - 1 < limit
  ceil_k : p.K + p.BK - 1 < limit
  total_bound : total p < limit
  group_span : p.GM * columns p < limit
  trip_bound : count p * (total p / p.SMS + 1) < limit
  row_pad : rowPad p ≤ limit
  column_pad : columnPad p ≤ limit
  k_pad : kPad p ≤ limit
  a_offset : (p.M-1)*p.sam + (kPad p-1)*p.sak < limit
  b_offset : (kPad p-1)*p.sbk + (p.N-1)*p.sbn < limit
  c_offset : (rowPad p-1)*p.scm + (columnPad p-1)*p.scn < limit

theorem rows_pos (p : Parameters) (h : Admissible p) : 0 < rows p :=
  RealMath.blocks_positive _ _ h.m h.bm
theorem columns_pos (p : Parameters) (h : Admissible p) : 0 < columns p :=
  RealMath.blocks_positive _ _ h.n h.bn
theorem rows_lt (p : Parameters) (h : Admissible p) : rows p < limit :=
  lt_of_le_of_lt (Nat.div_le_self _ _) h.ceil_m
theorem columns_lt (p : Parameters) (h : Admissible p) : columns p < limit :=
  lt_of_le_of_lt (Nat.div_le_self _ _) h.ceil_n
theorem count_lt (p : Parameters) (h : Admissible p) : count p < limit :=
  lt_of_le_of_lt (Nat.div_le_self _ _) h.ceil_k

theorem work_le (p : Parameters) (pid : Nat) : work p pid ≤ total p / p.SMS + 1 := by
  unfold work Schedule.tilesFor
  split_ifs <;> omega

theorem work_lt (p : Parameters) (h : Admissible p) (pid : Nat) (hp : pid < p.SMS) :
    work p pid < limit := by
  have hw : work p pid ≤ total p := by
    by_cases hz : work p pid = 0
    · omega
    · have ht := (Schedule.ordinal_bound_iff (total p) p.SMS pid
        (work p pid-1) h.sms hp).mp (by change work p pid-1 < work p pid; omega)
      have hm : work p pid-1 ≤ (work p pid-1)*p.SMS := by
        exact Nat.le_mul_of_pos_right _ h.sms
      unfold Schedule.tileId at ht
      omega
  exact lt_of_le_of_lt hw h.total_bound

theorem trip_lt (p : Parameters) (h : Admissible p) (pid : Nat) : trip p pid < limit :=
  lt_of_le_of_lt (Nat.mul_le_mul_left _ (work_le p pid)) h.trip_bound

theorem tile_lt (p : Parameters) (h : Admissible p) (pid q : Nat)
    (hp : pid < p.SMS) (hq : q < work p pid) :
    Schedule.tileId p.SMS pid q < total p :=
  (Schedule.ordinal_bound_iff _ _ _ _ h.sms hp).mp hq

/-- Every source row/column lane is representable before clamping/masking. -/
theorem lane_lt {tiles block index lane : Nat} (hi : index < tiles) (hl : lane < block) :
    index*block+lane < tiles*block := by
  have hm := Nat.mul_le_mul_right block (Nat.succ_le_of_lt hi)
  nlinarith

theorem row_lane_lt (p : Parameters) (h : Admissible p) (row lane : Nat)
    (hr : row < rows p) (hl : lane < p.BM) : row*p.BM+lane < limit :=
  lt_of_lt_of_le (lane_lt hr hl) h.row_pad

theorem column_lane_lt (p : Parameters) (h : Admissible p) (column lane : Nat)
    (hc : column < columns p) (hl : lane < p.BN) : column*p.BN+lane < limit :=
  lt_of_lt_of_le (lane_lt hc hl) h.column_pad

theorem k_lane_lt (p : Parameters) (h : Admissible p) (ki lane : Nat)
    (hk : ki < count p) (hl : lane < p.BK) : ki*p.BK+lane < limit :=
  lt_of_lt_of_le (lane_lt hk hl) h.k_pad

/-- A clamp is evaluated as a signed comparison and select in the source;
its mathematical value stays inside the original input row/column domain. -/
def clamp (bound value : Nat) := if value < bound then value else 0

theorem clamp_lt (bound value : Nat) (hb : 0 < bound) : clamp bound value < bound := by
  unfold clamp
  split_ifs <;> omega

theorem a_offset_lt (p : Parameters) (h : Admissible p) (r k : Nat)
    (hr : r < p.M) (hk : k < kPad p) : r*p.sam+k*p.sak < limit := by
  have h₁ := Nat.mul_le_mul_right p.sam (show r ≤ p.M-1 by omega)
  have h₂ := Nat.mul_le_mul_right p.sak (show k ≤ kPad p-1 by omega)
  exact lt_of_le_of_lt (Nat.add_le_add h₁ h₂) h.a_offset

theorem b_offset_lt (p : Parameters) (h : Admissible p) (k c : Nat)
    (hk : k < kPad p) (hc : c < p.N) : k*p.sbk+c*p.sbn < limit := by
  have h₁ := Nat.mul_le_mul_right p.sbk (show k ≤ kPad p-1 by omega)
  have h₂ := Nat.mul_le_mul_right p.sbn (show c ≤ p.N-1 by omega)
  exact lt_of_le_of_lt (Nat.add_le_add h₁ h₂) h.b_offset

theorem c_offset_lt (p : Parameters) (h : Admissible p) (r c : Nat)
    (hr : r < rowPad p) (hc : c < columnPad p) : r*p.scm+c*p.scn < limit := by
  have h₁ := Nat.mul_le_mul_right p.scm (show r ≤ rowPad p-1 by omega)
  have h₂ := Nat.mul_le_mul_right p.scn (show c ≤ columnPad p-1 by omega)
  exact lt_of_le_of_lt (Nat.add_le_add h₁ h₂) h.c_offset

/-- All group quotients, products, positive denominators, remainders and
coordinates remain in the signed-positive i32 half, including the short group. -/
theorem group_bounds (p : Parameters) (h : Admissible p) (tile : Nat)
    (ht : tile < total p) :
    0 < p.GM * columns p ∧
    tile / (p.GM * columns p) < limit ∧
    Schedule.groupBase (rows p) (columns p) p.GM tile < rows p ∧
    0 < Schedule.groupSize (rows p) (columns p) p.GM tile ∧
    Schedule.groupSize (rows p) (columns p) p.GM tile ≤ p.GM ∧
    Schedule.row (rows p) (columns p) p.GM tile < rows p ∧
    Schedule.column (rows p) (columns p) p.GM tile < columns p := by
  have hg := Schedule.grouped_bounds (rows p) (columns p) p.GM tile h.gm ht
  have hb : Schedule.groupBase (rows p) (columns p) p.GM tile < rows p := by
    have hr := hg.2.1
    unfold Schedule.row at hr
    omega
  exact ⟨Nat.mul_pos h.gm (columns_pos p h),
    lt_of_le_of_lt (Nat.div_le_self _ _) (lt_trans ht h.total_bound),hb,hg.1,
    min_le_right _ _,hg.2.1,hg.2.2⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeDomain

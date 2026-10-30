import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Exact host launch-batch coverage, flattened coordinate decoding and tile
ownership. These claims apply to every allocated shape, without an i32-domain
restriction on logical dimensions or numbers of heads. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Grid

def limit : Nat := 2147483647

def batchBase (cap n : Nat) := n / cap * cap
def batchLane (cap n : Nat) := n % cap

theorem batch_covers {cap total n : Nat} (hc : 0 < cap) (hn : n < total) :
    batchBase cap n + batchLane cap n = n ∧
    batchLane cap n < min cap (total - batchBase cap n) := by
  have hs := Nat.mod_add_div n cap
  rw [Nat.mul_comm cap] at hs
  have hb := Nat.mod_lt n hc
  simp only [batchBase,batchLane]
  constructor
  · omega
  · apply Nat.lt_min.mpr
    exact ⟨hb,by omega⟩

theorem batch_base_is_loop_member {cap total n : Nat} (hc : 0 < cap) (hn : n < total) :
    batchBase cap n < total ∧ cap ∣ batchBase cap n := by
  have h := batch_covers hc hn
  constructor
  · omega
  · exact ⟨n/cap,by simp [batchBase,Nat.mul_comm]⟩

theorem batch_no_duplicate {cap a b x y : Nat} (hc : 0 < cap)
    (hx : x < cap) (hy : y < cap) (he : a*cap+x = b*cap+y) : a=b ∧ x=y := by
  have hdiv := congrArg (fun n => n/cap) he
  have hmod := congrArg (fun n => n%cap) he
  simp [Nat.add_comm, Nat.add_mul_div_right _ _ hc,Nat.div_eq_of_lt hx,Nat.div_eq_of_lt hy,
    Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy,hc.ne'] at hdiv hmod
  exact ⟨hdiv,hmod⟩

/-- Finite Cartesian launch flattening, with axis zero varying fastest. -/
def encode (nx ny x y z : Nat) := (z*ny+y)*nx+x
def decodeX (nx n : Nat) := n%nx
def decodeY (nx ny n : Nat) := (n/nx)%ny
def decodeZ (nx ny n : Nat) := n/(nx*ny)

theorem decode_encode {nx ny x y z : Nat} (hx : x < nx) (hy : y < ny) :
    decodeX nx (encode nx ny x y z) = x ∧
    decodeY nx ny (encode nx ny x y z) = y ∧
    decodeZ nx ny (encode nx ny x y z) = z := by
  have hnx : 0 < nx := Nat.zero_lt_of_lt hx
  have hny : 0 < ny := Nat.zero_lt_of_lt hy
  have hd : encode nx ny x y z / nx = z*ny+y := by
    simp [encode, Nat.add_comm _ x, Nat.add_mul_div_right _ _ hnx,Nat.div_eq_of_lt hx]
  constructor
  · simp [decodeX,encode,Nat.mod_eq_of_lt hx]
  constructor
  · simp [decodeY,hd,Nat.mod_eq_of_lt hy]
  · rw [decodeZ,← Nat.div_div_eq_div_mul,hd]
    simp [Nat.add_comm _ y, Nat.add_mul_div_right _ _ hny,Nat.div_eq_of_lt hy]

theorem encode_bounded {nx ny nz x y z : Nat} (hx : x < nx) (hy : y < ny) (hz : z < nz) :
    encode nx ny x y z < nz*ny*nx := by
  have hyz : z*ny+y<nz*ny := by nlinarith
  unfold encode
  nlinarith

theorem decode_bounded {nx ny nz n : Nat} (hx : 0 < nx) (hy : 0 < ny)
    (hn : n < nz*ny*nx) :
    decodeX nx n < nx ∧ decodeY nx ny n < ny ∧ decodeZ nx ny n < nz := by
  refine ⟨Nat.mod_lt _ hx,Nat.mod_lt _ hy,?_⟩
  unfold decodeZ
  apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos hx hy)).2
  nlinarith

theorem decode_encode_identity {nx ny n : Nat} (hx : 0 < nx) (hy : 0 < ny) :
    encode nx ny (decodeX nx n) (decodeY nx ny n) (decodeZ nx ny n) = n := by
  have h1 := Nat.mod_add_div n nx
  have h2 := Nat.mod_add_div (n/nx) ny
  unfold encode decodeX decodeY decodeZ
  rw [← Nat.div_div_eq_div_mul]
  nlinarith

/-- No native y/z grid limit: every actual dimension is on the x grid, bounded
by 2^31-1; the mathematical program ID may span multiple such launches. -/
theorem actual_grid_legal {total base : Nat} (hb : base < total) :
    0 < min limit (total-base) ∧ min limit (total-base) < 2^31 := by
  constructor
  · exact Nat.lt_min.mpr ⟨by norm_num [limit],by omega⟩
  · have h := Nat.min_le_left limit (total-base)
    norm_num [limit] at h ⊢

/-- Exhausted autotune configurations launch one guarded dummy program whose
linear index is outside their total program count. -/
theorem exhausted_autotune_no_active {base total lane : Nat} (h : total ≤ base) :
    ¬ base+lane < total := by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.Grid

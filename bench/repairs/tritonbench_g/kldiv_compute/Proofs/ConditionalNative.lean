import bench.tritonbench_g.vector_addition.Proofs.HostGrid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorModel

/-! Exact integer/ABI refinement of the repaired source. No newly imposed
n<2^31 bound: all ordinary signed64 numel values and CUDA-supported 1-D grids.
Inactive lanes perform no memory access. Only active pointer addresses require
backing; no rounded-tail allocation or all-lanes pointer-validity premise. -/
namespace VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 4000000

private def parameters (x y output : RegionName) (n : Nat) :
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters := ⟨x,y,output,n,1024⟩

def grid (n : Nat) : Nat := (n + 1023) / 1024

/-- Ordinary host numel representation and CUDA x-grid launch limit. Unsupported
grids are setup errors of the wrapper, not correctness preconditions on inputs. -/
structure Supported (n : Nat) : Prop where
  extent : n < 2^63
  gridLimit : grid n ≤ 2^31-1

def pid64 (pid : Nat) : BitVec 64 := (BitVec.ofNat 32 pid).signExtend 64
def lane64 (lane : Nat) : BitVec 64 := (BitVec.ofNat 32 lane).signExtend 64
def start (pid : Nat) : BitVec 64 := pid64 pid * BitVec.ofNat 64 1024
def offset (pid lane : Nat) : BitVec 64 := start pid + lane64 lane
def bound (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
def mask (n pid lane : Nat) : Bool := (offset pid lane).slt (bound n)
def pointer (base pid lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + offset pid lane * BitVec.ofNat 64 4

theorem nat64_signed (n : Nat) (h : n < 2^63) :
    (BitVec.ofNat 64 n).toInt = (n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem widen32_exact (n : Nat) (h : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have h32 : (BitVec.ofNat 32 n).toInt = (n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
      (by decide) (by omega) (by norm_num; omega)
  rw [h32, nat64_signed n (by omega)]

theorem launched_bounds (n pid lane : Nat) (api : Supported n)
    (hp : pid < grid n) (hl : lane < 1024) :
    pid < 2^31 ∧ pid*1024 < 2^41 ∧ pid*1024+lane < 2^41 := by
  have hg := api.gridLimit
  norm_num at hg ⊢
  omega

theorem offset_word (n pid lane : Nat) (api : Supported n)
    (hp : pid < grid n) (hl : lane < 1024) :
    start pid = BitVec.ofNat 64 (pid*1024) ∧
    offset pid lane = BitVec.ofNat 64 (pid*1024+lane) := by
  have bounds := launched_bounds n pid lane api hp hl
  have pidWord := widen32_exact pid bounds.1
  have laneWord := widen32_exact lane (by omega)
  simp [start, offset, pid64, lane64, pidWord, laneWord,
    BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

/-- Every source intermediate is represented exactly; the source's signed
comparison equals the independent logical range test, including masked tails. -/
theorem source_control_exact (n pid lane : Nat) (api : Supported n)
    (hp : pid < grid n) (hl : lane < 1024) :
    (pid64 pid).toInt = (pid : Int) ∧
    (start pid).toInt = (pid*1024 : Nat) ∧
    (offset pid lane).toInt = (pid*1024+lane : Nat) ∧
    mask n pid lane = decide (pid*1024+lane < n) := by
  have b := launched_bounds n pid lane api hp hl
  have w := offset_word n pid lane api hp hl
  have p : (pid64 pid).toInt = (pid : Int) := by
    rw [pid64, widen32_exact pid b.1, nat64_signed pid (by omega)]
  have s : (start pid).toInt = (pid*1024 : Nat) := by rw [w.1, nat64_signed _ (by omega)]
  have o : (offset pid lane).toInt = (pid*1024+lane : Nat) := by rw [w.2, nat64_signed _ (by omega)]
  refine ⟨p,s,o,?_⟩
  rw [mask, BitVec.slt_eq_decide, o, bound, nat64_signed n api.extent]
  simp only [Int.ofNat_lt]

structure Buffer where
  base : Nat
  capacity : Nat
  device : Nat

def Buffer.Valid (b : Buffer) (n : Nat) : Prop :=
  4*n ≤ b.capacity ∧ b.base+b.capacity ≤ 2^64 ∧ b.base % 4 = 0

def Buffer.Disjoint (a b : Buffer) : Prop :=
  a.base+a.capacity ≤ b.base ∨ b.base+b.capacity ≤ a.base

/-- The actual 4-byte FP32 load/store address, for every and only active lane,
lies inside the ordinary backing allocation and cannot wrap the pointer ABI. -/
theorem active_address_exact (b : Buffer) (n pid lane : Nat) (api : Supported n)
    (valid : b.Valid n) (hp : pid < grid n) (hl : lane < 1024)
    (active : mask n pid lane = Bool.true) :
    (pointer b.base pid lane).toNat = b.base + 4*(pid*1024+lane) ∧
    b.base ≤ (pointer b.base pid lane).toNat ∧
    (pointer b.base pid lane).toNat + 4 ≤ b.base+b.capacity ∧
    (pointer b.base pid lane).toNat + 4 ≤ 2^64 := by
  have ctl := (source_control_exact n pid lane api hp hl).2.2.2
  have hi : pid*1024+lane < n := by simpa [ctl] using active
  have fin : b.base + 4*(pid*1024+lane) < 2^64 := by
    rcases valid with ⟨cap,fin,_⟩; omega
  have ptr : (pointer b.base pid lane).toNat = b.base + 4*(pid*1024+lane) := by
    rw [pointer, (offset_word n pid lane api hp hl).2,
      BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
    rw [Nat.mul_comm (pid*1024+lane) 4, Nat.mod_eq_of_lt fin]
  refine ⟨ptr,?_,?_,?_⟩ <;> rw [ptr]
  · omega
  · rcases valid with ⟨cap,_,_⟩; omega
  · rcases valid with ⟨cap,fin,_⟩; omega

theorem active_byte_bound (b : Buffer) (n pid lane byte : Nat) (api : Supported n)
    (valid : b.Valid n) (hp : pid < grid n) (hl : lane < 1024)
    (active : mask n pid lane = Bool.true) (hb : byte < 4) :
    b.base ≤ (pointer b.base pid lane).toNat + byte ∧
    (pointer b.base pid lane).toNat + byte < b.base+b.capacity ∧
    (pointer b.base pid lane).toNat + byte < 2^64 := by
  have p := active_address_exact b n pid lane api valid hp hl active
  omega

/-- Every visible element has exactly one launched owning lane. -/
theorem owner (n i : Nat) (hi : i < n) :
    i / 1024 < grid n ∧ i % 1024 < 1024 ∧ (i/1024)*1024+i%1024 = i := by
  have covers := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.grid_covers
    (parameters ⟨"x"⟩ ⟨"y"⟩ ⟨"output"⟩ n) (by change 0 < (1024 : Nat); decide)
  change n ≤ grid n * 1024 at covers
  omega

theorem unique_lane (p q i j : Nat) (hi : i < 1024) (hj : j < 1024)
    (same : p*1024+i = q*1024+j) : p=q ∧ i=j := by omega

theorem distinct_store_bytes (base p q i j a b : Nat) (hi : i < 1024) (hj : j < 1024)
    (ha : a < 4) (hb : b < 4) (different : p≠q ∨ i≠j) :
    base+4*(p*1024+i)+a ≠ base+4*(q*1024+j)+b := by
  intro eq
  have offsets : p*1024+i = q*1024+j := by omega
  have both := unique_lane p q i j hi hj offsets
  rcases different with h|h <;> simp_all

/-- Fresh output storage separates every write from both read-only inputs,
regardless of overlap between the caller's original logical input views. -/
theorem separated_accesses (input output : Buffer) (n i j : Nat)
    (vi : input.Valid n) (vo : output.Valid n) (fresh : input.Disjoint output)
    (hi : i<n) (hj : j<n) :
    input.base+4*i+4 ≤ output.base+4*j ∨ output.base+4*j+4 ≤ input.base+4*i := by
  rcases vi with ⟨ci,_,_⟩; rcases vo with ⟨co,_,_⟩
  rcases fresh with h|h <;> omega

/-- Previously failing endpoint is admitted and has its first lane active. -/
theorem endpoint_repaired : Supported (2^31) ∧ mask (2^31) 0 0 = Bool.true := by
  constructor
  · constructor <;> norm_num [grid]
  · decide +kernel

#axiomsClean widen32_exact
#axiomsClean source_control_exact
#axiomsClean active_address_exact
#axiomsClean active_byte_bound
#axiomsClean owner
#axiomsClean unique_lane
#axiomsClean distinct_store_bytes
#axiomsClean separated_accesses
#axiomsClean endpoint_repaired
end VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native

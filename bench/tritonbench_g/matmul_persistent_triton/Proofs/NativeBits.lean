import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeDomain

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeBits
open NativeDomain
set_option maxHeartbeats 2000000

def Fits (value : Int) : Prop := -(2147483648 : Int) ≤ value ∧ value < 2147483648
abbrev bits (value : Int) : BitVec 32 := BitVec.ofInt 32 value
def Exact (value : BitVec 32) (math : Int) : Prop := value.toInt = math

theorem nat_fits {n : Nat} (h : n < limit) : Fits (n : Int) := by
  unfold Fits limit at *
  omega

theorem bits_exact {z : Int} (h : Fits z) : Exact (bits z) z := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 h.2

theorem nat_exact {n : Nat} (h : n < limit) : Exact (bits (n : Int)) n :=
  bits_exact (nat_fits h)

theorem bmod_exact {z : Int} (h : Fits z) : z.bmod (2^32) = z := by
  have he := bits_exact h
  simpa only [Exact,BitVec.toInt_ofInt] using he

theorem add_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) (hr : Fits (a+b)) : Exact (x+y) (a+b) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  change (x+y).toInt = a+b
  rw [BitVec.toInt_add,hx,hy,bmod_exact hr]

theorem sub_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) (hr : Fits (a-b)) : Exact (x-y) (a-b) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  change (x-y).toInt = a-b
  rw [BitVec.toInt_sub,hx,hy,bmod_exact hr]

theorem mul_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) (hr : Fits (a*b)) : Exact (x*y) (a*b) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  change (x*y).toInt = a*b
  rw [BitVec.toInt_mul,hx,hy,bmod_exact hr]

/-- Signed machine division agrees with natural division for the nonnegative
operands of the actual source. Negative initial registers are never divided. -/
theorem div_exact {x y : BitVec 32} {a b : Nat}
    (hx : Exact x a) (hy : Exact y b) (ha : a < limit) :
    Exact (x.sdiv y) ((a/b : Nat) : Int) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  change (x.sdiv y).toInt = ((a/b : Nat) : Int)
  rw [BitVec.toInt_sdiv,hx,hy,← Int.ofNat_tdiv]
  exact bmod_exact (nat_fits (lt_of_le_of_lt (Nat.div_le_self _ _) ha))

theorem rem_exact {x y : BitVec 32} {a b : Nat}
    (hx : Exact x a) (hy : Exact y b) : Exact (x.srem y) ((a%b : Nat) : Int) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  change (x.srem y).toInt = ((a%b : Nat) : Int)
  rw [BitVec.toInt_srem,hx,hy,← Int.ofNat_tmod]

theorem slt_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) : x.slt y = decide (a<b) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  simp only [BitVec.slt_eq_decide,hx,hy]

theorem eq_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) : (x == y) = decide (a=b) := by
  change x.toInt = a at hx
  change y.toInt = b at hy
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq,← BitVec.toInt_inj,hx,hy,decide_eq_true_eq]

theorem select_exact {x y : BitVec 32} {a b : Int} (condition : Bool)
    (hx : Exact x a) (hy : Exact y b) :
    Exact (if condition then x else y) (if condition then a else b) := by
  cases condition <;> assumption

def smin (x y : BitVec 32) := if x.slt y then x else y

theorem min_exact {x y : BitVec 32} {a b : Int}
    (hx : Exact x a) (hy : Exact y b) : Exact (smin x y) (min a b) := by
  unfold smin
  rw [slt_exact hx hy]
  by_cases h : a < b
  · simpa [h,min_eq_left h.le] using hx
  · have hb : b ≤ a := by omega
    simpa [h,min_eq_right hb] using hy

theorem nat_add_exact {x y : BitVec 32} {a b : Nat}
    (hx : Exact x a) (hy : Exact y b) (hr : a+b < limit) :
    Exact (x+y) ((a+b : Nat) : Int) := by
  exact add_exact hx hy (by simpa only [Int.natCast_add] using nat_fits hr)

theorem nat_mul_exact {x y : BitVec 32} {a b : Nat}
    (hx : Exact x a) (hy : Exact y b) (hr : a*b < limit) :
    Exact (x*y) ((a*b : Nat) : Int) := by
  exact mul_exact hx hy (by simpa only [Int.natCast_mul] using nat_fits hr)

theorem nat_sub_exact {x y : BitVec 32} {a b : Nat}
    (hx : Exact x a) (hy : Exact y b) (hba : b ≤ a) (ha : a < limit) :
    Exact (x-y) ((a-b : Nat) : Int) := by
  rw [Int.natCast_sub hba]
  apply sub_exact hx hy
  have hd := nat_fits (lt_of_le_of_lt (Nat.sub_le a b) ha)
  simpa only [Int.natCast_sub hba] using hd

/-- LLVM widens global-memory indices by sign-extension, preserving the
already proved source i32 integer value before pointer arithmetic. -/
theorem sign_extend_exact {x : BitVec 32} {a : Int} (hx : Exact x a) :
    (x.signExtend 64).toInt = a := by
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  exact hx

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeBits

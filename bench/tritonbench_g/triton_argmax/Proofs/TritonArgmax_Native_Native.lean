import bench.tritonbench_g.triton_argmax.Proofs.Tree

/-! Repaired fixed-width control/address arithmetic. Exactness is proved for
active addresses. Inactive row products are allowed to wrap modulo 64 bits;
their masks depend only on separately bounded row/column coordinates. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Native

def modulus : Nat := 2^64
def signedLimit : Nat := 2^63
def wrap (n : Nat) : Nat := n%modulus
def signed (n : Nat) : Int :=
  let bits := wrap n
  if bits < signedLimit then Int.ofNat bits else Int.ofNat bits-Int.ofNat modulus

theorem limits : signedLimit < modulus := by norm_num [signedLimit,modulus]

theorem signed_exact (n : Nat) (small : n < signedLimit) : signed n=Int.ofNat n := by
  simp [signed,wrap,Nat.mod_eq_of_lt (small.trans limits),small]

theorem wrap_add (a b : Nat) : wrap (wrap a+wrap b)=wrap (a+b) := by
  simp [wrap,Nat.add_mod]

theorem wrap_mul (a b : Nat) : wrap (wrap a*wrap b)=wrap (a*b) := by
  simp [wrap,Nat.mul_mod]

def dimOffset (row col inner n k : Nat) : Nat :=
  wrap (wrap (wrap row*wrap n)*wrap k + wrap (wrap col*wrap k)+wrap inner)

theorem dim_offset_modular (row col inner n k : Nat) :
    dimOffset row col inner n k=wrap (row*n*k+col*k+inner) := by
  simp [dimOffset,wrap,Nat.add_mod,Nat.mul_mod]

theorem dim_offset_bound (m n k row col inner : Nat)
    (rowValid : row < m) (colValid : col < n) (innerValid : inner < k) :
    row*n*k+col*k+inner < m*n*k := by
  have colNext := Nat.mul_le_mul_right k (show col+1 ≤ n by omega)
  have rowNext := Nat.mul_le_mul_right (n*k) (show row+1 ≤ m by omega)
  nlinarith

theorem dim_output_bound (m k row inner : Nat)
    (rowValid : row < m) (innerValid : inner < k) : row*k+inner < m*k := by
  have rowNext := Nat.mul_le_mul_right k (show row+1 ≤ m by omega)
  nlinarith

theorem active_dim_address (m n k row col inner : Nat)
    (rowValid : row < m) (colValid : col < n) (innerValid : inner < k)
    (allocation : 4*(m*n*k) < signedLimit) :
    dimOffset row col inner n k=row*n*k+col*k+inner ∧
      signed (row*n*k+col*k+inner)=Int.ofNat (row*n*k+col*k+inner) := by
  have bound := dim_offset_bound m n k row col inner rowValid colValid innerValid
  have small : row*n*k+col*k+inner < signedLimit := by omega
  refine ⟨?_,signed_exact _ small⟩
  rw [dim_offset_modular]
  exact Nat.mod_eq_of_lt (small.trans limits)

theorem active_output_address (m k row inner : Nat)
    (rowValid : row < m) (innerValid : inner < k)
    (allocation : 8*(m*k) < signedLimit) :
    signed (row*k+inner)=Int.ofNat (row*k+inner) := by
  have bound := dim_output_bound m k row inner rowValid innerValid
  apply signed_exact
  omega

/-- Row/column bounds are sufficient for exact signed guards even if the
subsequent inactive row's three-factor pointer expression overflows. -/
theorem exact_mask (m n row col : Nat)
    (hm : m < signedLimit) (hn : n < signedLimit)
    (hr : row < signedLimit) (hc : col < signedLimit) :
    (signed row < signed m ∧ signed col < signed n) ↔ (row < m ∧ col < n) := by
  rw [signed_exact _ hr,signed_exact _ hm,signed_exact _ hc,signed_exact _ hn]
  exact and_congr Int.ofNat_lt Int.ofNat_lt

theorem padded_coordinate_small (n coordinate : Nat)
    (allocation : 4*n < signedLimit) (tail : coordinate < n+128) : coordinate < signedLimit := by
  norm_num [signedLimit] at *
  omega

theorem stage1_index_bound (size block pid lane : Nat)
    (activeBlock : pid*block < size) (laneBound : lane < block) :
    lane+pid*block < size+block := by omega

theorem stage1_index_exact (size block pid lane : Nat)
    (activeBlock : pid*block < size) (laneBound : lane < block)
    (allocation : 4*size < signedLimit) (acceptedBlock : block ≤ 1048576) :
    signed (lane+pid*block)=Int.ofNat (lane+pid*block) := by
  apply signed_exact
  have bound := stage1_index_bound size block pid lane activeBlock laneBound
  norm_num [signedLimit] at *
  omega

theorem grid_row_coverage (m row : Nat) (valid : row < m) :
    row/128 < (m+127)/128 ∧ row/128*128+row%128=row ∧ row%128 < 128 := by
  have d := Nat.div_add_mod row 128
  have r := Nat.mod_lt row (by omega : 0 < 128)
  have mmod := Nat.mod_lt (m+127) (by omega : 0 < 128)
  have md := Nat.div_add_mod (m+127) 128
  constructor
  · omega
  constructor
  · omega
  · exact r

theorem grid_row_tail (m pid lane : Nat) (pidBound : pid < (m+127)/128)
    (laneBound : lane < 128) : pid*128+lane < m+128 := by
  have md := Nat.div_add_mod (m+127) 128
  omega

theorem output_unique (k row inner row' inner' : Nat)
    (hi : inner < k) (hi' : inner' < k) (same : row*k+inner=row'*k+inner') :
    row=row' ∧ inner=inner' := by
  have rows : row=row' := by
    by_cases less : row < row'
    · have bound := Nat.mul_le_mul_right k (show row+1 ≤ row' by omega)
      nlinarith
    · by_cases greater : row' < row
      · have bound := Nat.mul_le_mul_right k (show row'+1 ≤ row by omega)
        nlinarith
      · omega
  exact ⟨rows,by subst row'; omega⟩

#axiomsClean dim_offset_modular
#axiomsClean active_dim_address
#axiomsClean exact_mask
#axiomsClean stage1_index_exact
#axiomsClean grid_row_coverage
#axiomsClean output_unique
end VeriTile.Bench.Repairs.TritonArgmax.Native

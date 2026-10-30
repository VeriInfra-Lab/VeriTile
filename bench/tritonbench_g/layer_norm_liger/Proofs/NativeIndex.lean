import bench.tritonbench_g.layer_norm_liger.Proofs.Contract
import bench.tritonbench_g.layer_norm_liger.Proofs.Settings
import bench.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage

namespace VeriTile.Bench.Repairs.LayerNormLiger.NativeIndex

def i64 (n : Int) : Int := (BitVec.ofInt 64 n).toInt

theorem exact_nat (n : Nat) (h : n < 2^63) : i64 n = n := by
  unfold i64
  apply BitVec.toInt_ofInt_eq_self (by decide)
  all_goals norm_num at * <;> omega

/-- These two element bounds follow from ordinary successful FP32 allocation:
the input and the actual [programs,columns] partial-gradient allocation. -/
structure Capacity (rows columns programs : Nat) : Prop where
  input : rows * columns ≤ 2^62
  partials : programs * columns ≤ 2^62

structure Safe (rows columns programs block : Nat) : Prop where
  rowsFit : rows < 2^63
  columnsFit : columns < 2^63
  programsFit : programs < 2^63
  group : Launch.rowsPerProgram rows programs < 2^63
  grouping : ∀ pid, pid < programs →
    pid + 1 < 2^63 ∧
    pid * Launch.rowsPerProgram rows programs < 2^63 ∧
    (pid + 1) * Launch.rowsPerProgram rows programs < 2^63 ∧
    pid * Launch.rowsPerProgram rows programs * columns < 2^63
  rowOffsets : ∀ row, row < rows → ∀ lane, lane < block →
    row * columns < 2^63 ∧ row * columns + lane < 2^63 ∧
    (row + 1) * columns < 2^63 ∧ row + 1 < 2^63
  partialOffsets : ∀ pid, pid < programs → ∀ lane, lane < block →
    pid * columns < 2^63 ∧ pid * columns + lane < 2^63

theorem ceil_bound (rows programs : Nat) (hp : 0 < programs) :
    programs * Launch.rowsPerProgram rows programs < rows + programs := by
  have bound := Nat.div_mul_le_self (rows + programs - 1) programs
  unfold Launch.rowsPerProgram
  rw [Nat.mul_comm] at bound
  have add : rows + programs - 1 + 1 = rows + programs := by omega
  omega

theorem safe (rows columns programs block : Nat) (hr : 0 < rows) (hc : 0 < columns)
    (hp : 0 < programs) (hb : block ≤ 65536) (capacity : Capacity rows columns programs) :
    Safe rows columns programs block := by
  have rle : rows ≤ rows * columns := by nlinarith
  have ple : programs ≤ programs * columns := by nlinarith
  have cle : columns ≤ rows * columns := by nlinarith
  have cr := capacity.input
  have cp := capacity.partials
  have ceil := ceil_bound rows programs hp
  have group := Launch.ceil_positive rows programs hr hp
  have gUpper : Launch.rowsPerProgram rows programs ≤ programs * Launch.rowsPerProgram rows programs := by nlinarith
  have rounded : programs * Launch.rowsPerProgram rows programs * columns < 2^63 := by
    have product := Nat.mul_lt_mul_of_pos_right ceil hc
    nlinarith
  refine ⟨by omega,by omega,by omega,by omega,?_,?_,?_⟩
  · intro pid hi
    have bp := Nat.mul_le_mul_right (Launch.rowsPerProgram rows programs) hi
    simp only [Nat.succ_eq_add_one] at bp
    have scaled := Nat.mul_le_mul_right columns bp
    have start : pid * Launch.rowsPerProgram rows programs ≤
        (pid+1)*Launch.rowsPerProgram rows programs := by nlinarith
    have scaledStart := Nat.mul_le_mul_right columns start
    refine ⟨by omega,by omega,by omega,by omega⟩
  · intro row hrow lane hlane
    have rowNext := Nat.mul_le_mul_right columns hrow
    simp only [Nat.succ_eq_add_one] at rowNext
    refine ⟨by nlinarith,by nlinarith,by omega,by omega⟩
  · intro pid hi lane hlane
    have partialBound := Nat.mul_le_mul_right columns hi
    simp only [Nat.succ_eq_add_one] at partialBound
    refine ⟨by nlinarith,by nlinarith⟩

/-- Integer casts, left-associated products and pointer-lane additions are
exact. This applies to masked lanes as well as dereferenced active columns. -/
theorem source_row (rows columns programs block row lane : Nat)
    (h : Safe rows columns programs block) (hr : row < rows) (hl : lane < block) :
    i64 (i64 row * i64 columns) = (row * columns : Nat) ∧
    i64 (i64 (i64 row * i64 columns) + lane) = (row * columns + lane : Nat) ∧
    i64 (i64 (i64 row * i64 columns) + i64 columns) = ((row+1)*columns : Nat) := by
  have b := h.rowOffsets row hr lane hl
  rw [exact_nat row (lt_trans hr h.rowsFit), exact_nat columns h.columnsFit]
  have product : i64 ((row : Int)*columns) = (row*columns : Nat) := by simpa using exact_nat _ b.1
  rw [product]
  refine ⟨rfl,?_,?_⟩
  · simpa using exact_nat _ b.2.1
  · have eq : ((row*columns : Nat) : Int) + columns = ((row+1)*columns : Nat) := by push_cast; ring
    rw [eq,exact_nat _ b.2.2.1]

theorem source_group (rows columns programs block pid : Nat)
    (h : Safe rows columns programs block) (hp : pid < programs) :
    i64 (i64 pid * i64 (Launch.rowsPerProgram rows programs)) =
      (pid * Launch.rowsPerProgram rows programs : Nat) ∧
    i64 (i64 (i64 pid + 1) * i64 (Launch.rowsPerProgram rows programs)) =
      ((pid+1) * Launch.rowsPerProgram rows programs : Nat) ∧
    i64 (i64 (i64 pid * i64 (Launch.rowsPerProgram rows programs)) * i64 columns) =
      (pid * Launch.rowsPerProgram rows programs * columns : Nat) := by
  have b := h.grouping pid hp
  rw [exact_nat pid (lt_trans hp h.programsFit),exact_nat _ h.group,exact_nat _ h.columnsFit]
  have next : i64 ((pid : Int)+1) = (pid+1 : Nat) := by simpa using exact_nat _ b.1
  have first : i64 ((pid : Int)*Launch.rowsPerProgram rows programs) =
      (pid*Launch.rowsPerProgram rows programs : Nat) := by simpa using exact_nat _ b.2.1
  rw [next,first]
  refine ⟨rfl,?_,?_⟩
  · simpa using exact_nat _ b.2.2.1
  · simpa using exact_nat _ b.2.2.2

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.NativeIndex

import bench.tritonbench_g.fused_layernorm_triton.Proofs.Binding

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.NativeIndex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole

def i64 (n : Int) : Int := (BitVec.ofInt 64 n).toInt

theorem i64_exact (n : Nat) (h : n < 2^63) : i64 n = n := by
  unfold i64
  apply BitVec.toInt_ofInt_eq_self (by decide)
  all_goals norm_num at * <;> omega

/-- Covers padded lanes too, plus the increment after the last loop iteration.
Only active pointers are dereferenced; inactive arithmetic still fits int64. -/
structure Bounds (S D B row k lane : Nat) : Prop where
  rowBound : row < 2^63
  width : D < 2^63
  start : k * B < 2^63
  next : k * B + B < 2^63
  column : k * B + lane < 2^63
  rowStart : D * row < 2^63
  offset : k * B + lane + D * row < 2^63

theorem bounds (S D B row k lane : Nat) (positive : 0 < D)
    (width : D ≤ 2^62) (count : S * D ≤ 2^62)
    (config : B = 1024 ∨ B = 2048) (hr : row < S)
    (hk : k * B < D) (hl : lane < B) : Bounds S D B row k lane := by
  have rb : D * (row + 1) ≤ S * D := by
    simpa [Nat.mul_comm] using Nat.mul_le_mul_right D hr
  have rs : row ≤ D * row := by nlinarith
  rcases config with rfl | rfl <;> constructor <;> nlinarith

theorem source_offsets (S D B row k lane : Nat) (h : Bounds S D B row k lane) :
    i64 (i64 (k * B) + lane) = (k * B + lane : Nat) ∧
    i64 (i64 D * i64 row) = (D * row : Nat) ∧
    i64 (i64 (i64 (k * B) + lane) + i64 (i64 D * i64 row)) =
      (k * B + lane + D * row : Nat) ∧
    i64 (i64 (k * B) + B) = (k * B + B : Nat) := by
  have ks : i64 ((k : Int) * B) = (k * B : Nat) := by simpa using i64_exact _ h.start
  rw [ks, i64_exact _ h.width, i64_exact _ h.rowBound]
  have col : i64 ((k * B : Nat) + (lane : Int)) = (k * B + lane : Nat) := by
    simpa using i64_exact _ h.column
  have rs : i64 ((D : Int) * row) = (D * row : Nat) := by simpa using i64_exact _ h.rowStart
  rw [col, rs]
  refine ⟨rfl,rfl,?_,?_⟩
  · simpa using i64_exact _ h.offset
  · simpa using i64_exact _ h.next

theorem loop_entry (D B k : Nat) (hD : 0 < D) (hB : 0 < B) (hk : k < blockCount D B) :
    k * B < D := by
  have divbound := Nat.div_mul_le_self (D + B - 1) B
  have lower : k + 1 ≤ (D + B - 1) / B := hk
  have prod := Nat.mul_le_mul_right B lower
  have sub : D + B - 1 + 1 = D + B := by omega
  nlinarith

/-- Exact active byte pointer, including checked scalar alignment. -/
theorem active_address (base elements i bytes : Nat)
    (span : base + bytes * elements ≤ 2^64) (hi : i < elements)
    (positive : 0 < bytes) (align : base % bytes = 0) :
    base + bytes * i + bytes ≤ 2^64 ∧ (base + bytes * i) % bytes = 0 ∧
    (BitVec.ofNat 64 (base + bytes * i)).toNat = base + bytes * i := by
  have bound : base + bytes * i + bytes ≤ 2^64 := by nlinarith
  refine ⟨bound, by simp [Nat.add_mod, align], ?_⟩
  exact Nat.mod_eq_of_lt (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedLayernormTriton.NativeIndex

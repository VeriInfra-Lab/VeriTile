import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ByteMemory
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.KldivOps.Native
open VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory

/-- Ordinary successful contiguous FP32 allocation metadata. The native bound
is inherited from its real allocation; no caller-imposed size cap is introduced. -/
structure Dense where
  view : View
  nonnull : 0<view.words → 0<view.backing.base

theorem count_bound (d : Dense) : d.view.words < 2^62 := by
  have h := d.view.byte_count_lt d.nonnull
  norm_num at h ⊢
  omega

theorem matrix_bound (d : Dense) (rows cols : Nat) (shape : d.view.words=rows*cols)
    (row col : Nat) (hr : row<rows) (hc : col<cols) :
    row*cols+col < d.view.words ∧ row*cols+col < 2^62 := by
  have hmul := Nat.mul_le_mul_right cols (show row+1≤rows by omega)
  have hn := count_bound d
  constructor <;> simp only [Nat.add_mul,Nat.one_mul] at hmul <;> omega

/-- i64 loop start, active and padded lanes, and the next induction value all
fit. Allocation supplies the column bound whenever a row is actually launched. -/
theorem loop_intermediates (d : Dense) (rows cols row start block lane : Nat)
    (shape : d.view.words=rows*cols) (hr : row<rows) (hs : start<cols)
    (hb : block≤16384) (hl : lane<block) :
    cols<2^62 ∧ start+lane<2^63 ∧ start+block<2^63 := by
  have hrows : 1≤rows := by omega
  have hcols := Nat.mul_le_mul_right cols hrows
  have hn := count_bound d
  simp only [Nat.one_mul] at hcols
  norm_num at hn ⊢
  omega

theorem signed_exact (n : Nat) (h : n<2^63) :
    (BitVec.ofNat 64 n).toInt=(n : Int) := by
  have hmod : (BitVec.ofNat 64 n).toNat=n := by
    rw [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hmod]; omega),hmod]

theorem pointer_exact (d : Dense) (i : Nat) (hi : i<d.view.words) (k : Fin 4) :
    (BitVec.ofNat 64 (d.view.address i+k.val)).toNat=d.view.address i+k.val := by
  have h := d.view.address_native i hi
  rw [BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by omega)

theorem no_empty_dereference (cols start lane : Nat) (empty : cols=0) :
    ¬start+lane<cols := by omega

theorem reduction_output (d : Dense) (rows row : Nat) (shape : d.view.words=rows)
    (hr : row<rows) : row<2^62 ∧ d.view.address row+4≤2^64 := by
  constructor
  · have h := count_bound d; omega
  · apply d.view.address_native; omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivOps.Native

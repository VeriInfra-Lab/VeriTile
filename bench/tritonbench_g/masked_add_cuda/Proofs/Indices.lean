import bench.tritonbench_g.masked_add_cuda.Proofs.Contract
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteIndices

namespace VeriTile.Bench.Repairs.MaskedAddCuda.Indices
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite

def gridCount (n : Nat) : Nat := (n+1023)/1024

/-- All launched offsets, including the masked tail, fit signed int32. -/
theorem no_wrap (n pid lane : Nat) (hn : n < 2^31)
    (hp : pid < gridCount n) (hl : lane < 1024) :
    (start32 pid).toInt = (pid*1024 : Nat) ∧
    (offset32 pid lane).toInt = (pid*1024+lane : Nat) ∧
    mask32 n pid lane = decide (pid*1024+lane<n) :=
  source_index_arithmetic n pid lane hn hp hl

def pointer (base width pid lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + (offset32 pid lane).signExtend 64 * BitVec.ofNat 64 width

/-- Only active addresses need to fit storage: masked tail pointers are never
dereferenced. Width is 4 for floats and 4/8 for signed masks. -/
theorem pointer_exact (base width n pid lane : Nat) (hw : 0 < width)
    (hn : n < 2^31) (hp : pid < gridCount n) (hl : lane < 1024)
    (active : pid*1024+lane < n) (valid : base + width*n ≤ 2^64) :
    (pointer base width pid lane).toNat = base + width*(pid*1024+lane) ∧
    (pointer base width pid lane).toNat + width ≤ base + width*n := by
  have bound : base + width*(pid*1024+lane) < 2^64 := by
    have := Nat.mul_lt_mul_of_pos_left active hw
    omega
  have word : (pointer base width pid lane).toNat = base + width*(pid*1024+lane) := by
    unfold pointer
    rw [source_index_sign_extend n pid lane hn hp hl,
      BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
    rw [Nat.mul_comm (pid*1024+lane), Nat.mod_eq_of_lt bound]
  refine ⟨word, ?_⟩
  rw [word]
  have := Nat.mul_le_mul_left width (show pid*1024+lane+1 ≤ n by omega)
  rw [Nat.mul_succ] at this
  omega

theorem owner (n i : Nat) (hi : i < n) :
    i/1024 < gridCount n ∧ i%1024 < 1024 ∧ (i/1024)*1024+i%1024=i := by
  unfold gridCount
  omega

/-- Distinct logical coordinates write disjoint four-byte intervals. -/
theorem distinct_grad_bytes (base i j u v : Nat) (hu : u < 4) (hv : v < 4)
    (different : i ≠ j) : base+4*i+u ≠ base+4*j+v := by omega

theorem input_not_written (gBase inputBase width n i j u v : Nat)
    (separate : Separate gBase (4*n) inputBase (width*n))
    (hi : i < n) (hj : j < n) (hu : u < 4) (hv : v < width) :
    gBase+4*i+u ≠ inputBase+width*j+v := by
  have a := Nat.mul_le_mul_left width (show j+1 ≤ n by omega)
  simp only [Nat.mul_add, Nat.mul_one] at a
  rcases separate with h | h <;> omega

#axiomsClean no_wrap
#axiomsClean pointer_exact
#axiomsClean owner
#axiomsClean distinct_grad_bytes
#axiomsClean input_not_written
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaskedAddCuda.Indices

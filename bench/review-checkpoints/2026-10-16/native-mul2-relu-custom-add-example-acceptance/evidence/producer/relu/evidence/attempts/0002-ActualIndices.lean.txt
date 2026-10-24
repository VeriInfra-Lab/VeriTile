import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteIndices
import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteCodec

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual
open VeriTile.Triton FiniteAddress
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def grid (n : Nat) : Nat := (n+1023)/1024

/-- Native constexpr promotion, including unsigned and widened comparison.
N>=2^64 is rejected by the frontend before this expression executes. -/
def mask (n : Nat) (offset : BitVec 32) : Bool :=
  if n<2^31 then offset.slt (BitVec.ofNat 32 n)
  else if n<2^32 then offset.ult (BitVec.ofNat 32 n)
  else if n<2^63 then (offset.signExtend 64).slt (BitVec.ofNat 64 n)
  else (offset.signExtend 64).ult (BitVec.ofNat 64 n)

def active (n pid lane : Nat) : Bool := mask n (offset32 pid lane)
def lanes (n pid : Nat) : List Nat := (List.range 1024).filter (active n pid)

def address (pointer : BitVec 64) (width : Nat) (offset : BitVec 32) : Nat :=
  (pointer + offset.signExtend 64 * BitVec.ofNat 64 width).toNat

def laneAddress (pointer : BitVec 64) (width pid lane : Nat) : Nat :=
  address pointer width (offset32 pid lane)

theorem old_mask_refinement (n pid lane : Nat) (hn : n≤2^31) :
    active n pid lane=FiniteIndices.sourceMask n pid lane := by
  unfold active mask FiniteIndices.sourceMask indexLimit
  by_cases h : n<2^31
  · simp only [if_pos h,mask32]
  · have h32 : n<2^32 := by omega
    simp only [if_neg h,if_pos h32]

theorem mask_widened_all_active (n pid lane : Nat) (lo : 2^32≤n) (hi : n<2^63) :
    active n pid lane=Bool.true := by
  have hn : (BitVec.ofNat 64 n).toInt=(n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)
  have bound := BitVec.toInt_lt (x:=offset32 pid lane)
  unfold active mask
  rw [if_neg (by omega),if_neg (by omega),if_pos hi,BitVec.slt_eq_decide,
    BitVec.toInt_signExtend_of_le (by decide),hn]
  norm_num at bound
  simp only [decide_eq_true_eq]
  omega

theorem zero_program_offsets (lane : Nat) (hl : lane<1024) :
    (offset32 0 lane).toNat=lane ∧ (offset32 0 lane).toInt=(lane : Int) := by
  rw [offset32_word]
  simp only [offset,Nat.zero_mul,Nat.zero_add]
  exact ⟨by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)],nat32_signed lane (by unfold indexLimit;omega)⟩

theorem zero_program_mask (n lane : Nat) (hn : n<2^63) (hl : lane<1024) :
    active n 0 lane=decide (lane<n) := by
  have words := zero_program_offsets lane hl
  unfold active mask
  by_cases h31 : n<2^31
  · simp only [if_pos h31,BitVec.slt_eq_decide,words.2,nat32_signed n h31,Int.ofNat_lt]
  · by_cases h32 : n<2^32
    · simp only [if_neg h31,if_pos h32,BitVec.ult_eq_decide,words.1,BitVec.toNat_ofNat,Nat.mod_eq_of_lt h32]
    · have ha := mask_widened_all_active n 0 lane (by omega) hn
      change mask n (offset32 0 lane)=_ at ha
      change mask n (offset32 0 lane)=_
      rw [ha]
      simp [show lane<n by omega]

theorem boundary_read_examples :
    active (2^31+1) (2^21) 0=Bool.true ∧
    (offset32 (2^21) 0).toInt=-(2^31 : Int) ∧
    active (2^32) (2^21) 1023=Bool.true ∧
    active (2^63) (2^21) 0=Bool.false := by decide +kernel

theorem constexpr_zero_empty : lanes 0 0=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro lane hm
  have hl := List.mem_range.mp (List.mem_filter.mp hm).1
  have ha := (List.mem_filter.mp hm).2
  rw [zero_program_mask 0 lane (by decide) hl] at ha
  simp at ha

#axiomsClean old_mask_refinement
#axiomsClean mask_widened_all_active
#axiomsClean zero_program_mask
#axiomsClean boundary_read_examples
#axiomsClean constexpr_zero_empty
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual

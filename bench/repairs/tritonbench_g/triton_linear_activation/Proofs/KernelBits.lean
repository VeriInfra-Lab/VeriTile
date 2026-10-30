import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelNativeModel

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelBits
open VeriTile.Triton FP.Structural KernelSource KernelNativeModel
noncomputable section
set_option maxHeartbeats 3000000

/-- Little-endian actual storage words. Half storage occupies exactly two
bytes in the low 16 bits of the FP32-sized carrier; it is never a rounded
FP32 subnormal standing in for a half operand. -/
def wordAt (mode : Mode) (bytes : Nat → BitVec 8) (address : Nat) : Word :=
  let low := (bytes address).toNat+256*(bytes (address+1)).toNat
  FP.Value.ofBits .fp32 (match mode with
    | .fp32 => low+65536*(bytes (address+2)).toNat+16777216*(bytes (address+3)).toNat
    | .autocastFP16 => low)

theorem half_width (bytes : Nat → BitVec 8) (address : Nat) :
    (wordAt .autocastFP16 bytes address).raw<65536 := by
  have lo := (bytes address).isLt
  have hi := (bytes (address+1)).isLt
  change (bytes address).toNat<256 at lo
  change (bytes (address+1)).toNat<256 at hi
  have bound : (bytes address).toNat+256*(bytes (address+1)).toNat<65536 := by omega
  simp only [wordAt,FP.Value.raw,FP.Value.ofBits,BitVec.toNat_ofNat]
  exact lt_of_le_of_lt (Nat.mod_le _ _) bound

def storedByte (word : Word) (index : Nat) : BitVec 8 := BitVec.ofNat 8 (word.raw/2^(8*index))

/-- Input-only ideal scalar projection. Exceptional native bit patterns are
mapped to zero in the ideal interpretation; no numerical equality between
native execution and this Real interpretation is asserted. -/
def ideal (word : Word) : ℝ :=
  match word.decode with
  | .finite negative magnitude => if negative then -(magnitude : ℚ) else (magnitude : ℚ)
  | _ => 0

def realAt (mode : Mode) (bytes : Nat → BitVec 8) (address : Nat) : ℝ :=
  ideal (match mode with | .fp32 => wordAt mode bytes address | .autocastFP16 => unpackHalf (wordAt mode bytes address))

#axiomsClean half_width
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelBits

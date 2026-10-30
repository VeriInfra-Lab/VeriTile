import bench.tritonbench_g.diag_ssm_triton.Proofs.ScalarGraphs
import bench.tritonbench_g.diag_ssm_triton.Proofs.ByteMemory

/-! Concrete little-endian four-byte FP32 storage. All bit patterns are
represented, including signed zero and every NaN payload. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec
open Kernels ByteMemory
set_option autoImplicit false

def encode (word : Word) (byte : Fin 4) : Byte :=
  ⟨word.raw / (256 ^ byte.val) % 256, Nat.mod_lt _ (by decide)⟩

def decode (bytes : Fin 4 → Byte) : Word :=
  VeriTile.Triton.FP.Value.ofBits .fp32
    ((bytes 0).val + 256 * (bytes 1).val + 65536 * (bytes 2).val + 16777216 * (bytes 3).val)

theorem roundtrip (word : Word) : decode (encode word) = word := by
  cases word with
  | mk bits =>
    unfold decode VeriTile.Triton.FP.Value.ofBits
    congr 1
    apply BitVec.eq_of_toNat_eq
    have bound := bits.isLt
    simp only [decode, encode, VeriTile.Triton.FP.Value.ofBits, VeriTile.Triton.FP.Value.raw,
      BitVec.toNat_ofNat]
    norm_num [VeriTile.Triton.FP.Format.width, VeriTile.Triton.FP.Format.fractionBits] at *
    omega

def littleEndian : Codec Word := ⟨encode, decode, roundtrip⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec

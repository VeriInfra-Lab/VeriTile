import Mathlib.Tactic.Linarith
import bench.tritonbench_g.streamk_matmul.Proofs.Host
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16
namespace VeriTile.Bench.TritonBenchG.StreamK
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Dot32 := (m n k : Nat)→(Fin (m*k)→Word)→(Fin (k*n)→Word)→Fin (m*n)→Word

/-- Genuine fp32→binary16→fp32 conversion. C is fp32 in the original provided
four tests; the source full-wave half conversion is not the identity. -/
def halfRound (x : Word) : Word :=
 let h : MatmulDequantInt4.Binary16.Value := match x.decode with
  | .nan .. => MatmulDequantInt4.Binary16.Value.qNaN
  | .infinity sign => MatmulDequantInt4.Binary16.Value.infinity sign
  | .finite sign magnitude => MatmulDequantInt4.Binary16.round .ieeeValue
       (if sign then -magnitude else magnitude) sign
 match h.decode with
 | .nan .. => FP.Value.qNaN .fp32
 | .infinity sign => FP.Value.infinity .fp32 sign
 | .finite sign magnitude => FP.round .fp32 .ieeeValue (if sign then -magnitude else magnitude) sign

/-- Explicit fp32-storage profile. Arithmetic uses RNE/gradual underflow and
canonical arithmetic NaNs. dot retains default source options {}, but its
native interpretation is NOT assumed to be full fp32 input multiplication. -/
def fp32 (dot : Dot32) : Backend Word where
 zero:=FP.Value.zero .fp32
 dot:=dot
 add:=FP.add .ieeeValue
 half:=halfRound
 output:=id
 atomicAdd:=FP.add .ieeeValue

/-- A fully executable *software* dot instance: multiply then ascending fp32
sum, then the source performs a separate accumulator add. This is not a TF32,
MMA or fused dot-accumulator refinement theorem. -/
private theorem dotLeftIndex {m n k : Nat} (ij : Fin (m*n)) (t : Fin k) :
    ij.val/n*k+t.val<m*k := by
  have hm : ij.val/n<m := Nat.div_lt_of_lt_mul (by simpa only [Nat.mul_comm] using ij.isLt)
  nlinarith [t.isLt]
private theorem dotRightIndex {m n k : Nat} (ij : Fin (m*n)) (t : Fin k) :
    t.val*n+ij.val%n<k*n := by
  have hn : 0<n := by nlinarith [ij.isLt]
  have hi := Nat.mod_lt ij.val hn
  nlinarith [t.isLt]
def sequentialDot : Dot32 := fun m n k a b ij=>
 (List.finRange k).foldl (fun acc t=>FP.add .ieeeValue acc
   (FP.mul .ieeeValue
    (a ⟨(ij.val/n)*k+t.val,dotLeftIndex ij t⟩)
    (b ⟨t.val*n+ij.val%n,dotRightIndex ij t⟩)))
   (FP.Value.zero .fp32)

def software : Backend Word := fp32 sequentialDot

theorem half_loss_is_real : halfRound ⟨0x3f800001⟩=⟨0x3f800000⟩ ∧
 halfRound ⟨0x3f800001⟩≠(⟨0x3f800001⟩:Word) := by decide +kernel

theorem atomic_order_changes_bits :
 FP.add .ieeeValue (FP.add .ieeeValue (⟨0x4b800000⟩:Word) ⟨0xcb800000⟩) ⟨0x3f800000⟩=⟨0x3f800000⟩ ∧
 FP.add .ieeeValue (FP.add .ieeeValue (⟨0x4b800000⟩:Word) ⟨0x3f800000⟩) ⟨0xcb800000⟩=⟨0⟩ := by decide +kernel

#axiomsClean half_loss_is_real
#axiomsClean atomic_order_changes_bits
end VeriTile.Bench.TritonBenchG.StreamK

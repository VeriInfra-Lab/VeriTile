import VeriTile.Triton.Random.Values
import VeriTile.Triton.Float.DecoderCompatibility

/-! Optional bridge above both the random foundation and Core.Ast.
Core.Ast must not import this module. The pure random definitions and Values
module do not depend on this bridge. -/
namespace VeriTile.Triton.Random
open Dyadic

theorem canonical_rand_core_decode (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    Float32Bits.decodeRat ⟨(rand seed offset width nRounds).bits⟩ =
      some (uniformDyadic (foldSign (randint seed offset width nRounds))) := by
  rw [FP.DecoderCompatibility.equals_software_decoder]
  exact canonical_rand_toRat seed offset width nRounds

theorem rand_core_decode (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    Float32Bits.decodeRat ⟨(rand seed offset width nRounds).bits⟩ =
      some (randRat seed offset width nRounds) := by
  rw [FP.DecoderCompatibility.equals_software_decoder]
  exact rand_toRat_randRat seed offset width nRounds

end VeriTile.Triton.Random

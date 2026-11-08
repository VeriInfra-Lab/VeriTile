import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NaturalBitwiseLiterals
open VeriTile.Triton

def source (i : Nat) : ComputeKernel := triton {
  index = $(i)
  mask = 3 << (2 * index)
  result = mask & 3
}

def expected (i : Nat) : ComputeKernel := .mk [] [] [
  .assign .nat [] "index" (.alg (.constNat i)),
  .assign .nat [] "mask" (.alg (.shiftLeft .nil (.constNat 3)
    (.mul .nat .nil (.constNat 2) (.ref .nat [] "index")))),
  .assign .nat [] "result" (.alg (.bitAnd .nil (.ref .nat [] "mask") (.constNat 3)))]

theorem complete_program (i : Nat) : source i=expected i := rfl

/-- Every original packed-field mask fits the physical word. -/
theorem packed_masks : ∀ i : Fin 4,
    ((BitVec.ofNat 32 3) <<< (2*i.val)).toNat=3*2^(2*i.val) := by decide

/-- error: `<<`: dtype mismatch -/
#guard_msgs in
example (i : Nat) : ComputeKernel := triton {
  mask = 3.0 << (2 * $(i))
}

#axiomsClean complete_program
#axiomsClean packed_masks
end VeriTile.Tests.NaturalBitwiseLiterals

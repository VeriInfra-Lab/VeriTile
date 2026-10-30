import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantProfile
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.NativeProfile

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantNative
open VeriTile.Triton NativeProfile
noncomputable section

/-- Fixed lowering-specific reductions over the complete ordered lane vector.
The same callbacks are used by both source versions, including NaNs and ties. -/
structure Reductions where
  low : (n : Nat) → (Fin n → Word) → Word
  high : (n : Nat) → (Fin n → Word) → Word

/-- Exactly the five literals used by the source; the fallback is unused. -/
def sourceLiteral (x : ℝ) : Word :=
  if x = 0 then literal 0
  else if x = 1 then literal 1
  else if x = 15 then literal 15
  else if x = 255 then literal 255
  else if x = (1/2 : ℝ) then half
  else literal 0

def ops (a : Arithmetic) (reductions : Reductions) : QuantProfile.Ops Word where
  literal := sourceLiteral
  add := a.add
  sub := a.sub
  divide := a.divFull
  minimum := a.minNum
  maximum := a.maxNum
  equal := a.equal
  byte := PtxCast.byte
  low := reductions.low
  high := reductions.high

/-- Source cast and division are the existing selected native profile. -/
theorem native_primitives (a : Arithmetic) (r : Reductions) (x y : Word) :
    (QuantProfile.algebra (ops a r)).binary (some .fp32) .real .div x y = a.divFull x y ∧
    (QuantProfile.algebra (ops a r)).floatToNat (some .fp32) .real 8 =
      some (fun x => some (PtxCast.byte x).toNat) := by
  simp [QuantProfile.algebra,ops]

#axiomsClean native_primitives
end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantNative

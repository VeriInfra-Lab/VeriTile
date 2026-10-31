import VeriTile.Triton.DSL
/-! Pure source parameter types and clamp mathematics. The executable repaired
body and integer-address candidate live in the task's Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open Lean VeriTile.Triton

inductive BiasStorage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

/-- Finite upper limit or the host default +∞. -/
inductive UpperBound where
  | finite (value : ℝ)
  | posInf

instance : Coe ℝ UpperBound := ⟨UpperBound.finite⟩
instance (n : Nat) : OfNat UpperBound n := ⟨.finite n⟩

/-- Independent real meaning of the upper clamp. -/
noncomputable def UpperBound.cap (upper : UpperBound) (x : ℝ) : ℝ :=
  match upper with
  | .finite value => min x value
  | .posInf => x

end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd

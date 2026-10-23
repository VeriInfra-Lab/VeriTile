import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.IOBindings

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- The selected FP32 coefficient-first FMA backend. Observing a value removes
only internal product provenance, preserving every FP32 word, including NaNs.
All initial Tagged values are admitted, exactly as in the retained raw theorem. -/
def nativeIO (cfg : FP.Config) (r : Request) : ReadProgramIO Numerics.Tagged Numerics.Tagged :=
  PublicIO.sourceIO (Numerics.operations (KernelContract.nativePrimitives cfg))
    (fun x => Numerics.leaf x.word) r

/-- Independent native graph: low=fma(c,a,mul(s,b)),
high=fma(c,b,neg(mul(s,a))). Untouched odd tails retain their input word. -/
def nativeFormula (cfg : FP.Config) (r : Request) (xs : PublicIO.Inputs Numerics.Tagged r)
    (key : Bool) (row head col : Nat) : Numerics.Tagged :=
  Numerics.leaf (if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    Numerics.pair (KernelContract.nativePrimitives cfg) (decide (r.half≤col))
      (PublicIO.q xs key row head j).word
      (PublicIO.q xs key row head (j+r.half)).word
      (PublicIO.coefficient xs Bool.false (row%r.sequence) j).word
      (PublicIO.coefficient xs Bool.true (row%r.sequence) j).word
  else (PublicIO.q xs key row head col).word)

/-- Source-to-native observation, with complete Q/K outputs and exact frame.
The primitive configuration and selected FMA lowering are model boundaries;
this theorem uses no numerical transformation assumption or Real exactness. -/
theorem rope_backward_kernel_fp32 (cfg : FP.Config) (r : Request) :
    (nativeIO cfg r).Implements (fun xs => PublicIO.result (nativeFormula cfg r xs)) := by
  apply PublicIO.implements_formula
  intro xs key row head col
  simp only [PublicIO.logicalResult,nativeFormula]
  split
  · rw [Numerics.source_pair]
  · rfl

#specNonCircular nativeFormula avoiding [Source.kernel,grid,scalarResult,FP.Structural.exec]
#axiomsClean rope_backward_kernel_fp32
end
end VeriTile.Bench.Repairs.RopeBackwardTransform

import bench.tritonbench_g.int_scaled_matmul.Kernels
import bench.tritonbench_g.int_scaled_matmul.Proofs.CheckedNative

namespace VeriTile.Bench.Repairs.IntScaledMatmul.KernelModel
open VeriTile.Triton FP.Structural
open Revision001 FP.IntegerConversion
noncomputable section

abbrev Word := FP.Value .fp32

/-- Undefined float-to-integer conversions retain a local 32-bit result.
No admitted scale is excluded, and no output formula is a premise. -/
def signed32 (undefined : Word → BitVec 32) (x : Word) : BitVec 32 :=
  match toSigned 32 x with
  | some z => BitVec.ofInt 32 z
  | none => undefined x

/-- Concrete primitive semantics for the observed non-FTZ FP32 source profile.
These are operation definitions, not numerical-transformation assumptions. -/
def instructions (undefined : Word → BitVec 32) : CheckedNative.Instructions where
  promote32 := fun z => FP.round .fp32 Native.sourceConfig (z : Rat)
  multiply32 := FP.mul Native.sourceConfig
  signed32 := signed32 undefined

theorem signed32_defined (undefined : Word → BitVec 32) (x : Word) (z : Int)
    (defined : toSigned 32 x=some z) : (signed32 undefined x).toInt=z := by
  obtain ⟨q,hq,rounding,range⟩ := (toSigned_spec 32 x z).mp defined
  simp only [signed32,defined]
  exact BitVec.toInt_ofInt_eq_self range.1 range.2.1 range.2.2

/-- The former scalar-law interface is discharged from the selected operation
definitions; it is not a public experimental obligation. -/
theorem correspondence (undefined : Word → BitVec 32) :
    Native.PrimitiveCorrespondence (CheckedNative.ports (instructions undefined)) := by
  constructor
  · intro z hz; rfl
  · intro a b; rfl
  · intro x z hz
    exact signed32_defined undefined x z hz

def ports (undefined : Word → BitVec 32) := CheckedNative.ports (instructions undefined)
def fallback (undefined : Word → BitVec 32) := Native.fallback (ports undefined)

theorem conversion (undefined : Word → BitVec 32) (x : Word) :
    (signed32 undefined x).toInt=totalSigned (ports undefined).signed32 32 x :=
  Native.signed_eq_total (ports undefined) (correspondence undefined) x

#axiomsClean signed32_defined
#axiomsClean correspondence
#axiomsClean conversion
end
end VeriTile.Bench.Repairs.IntScaledMatmul.KernelModel

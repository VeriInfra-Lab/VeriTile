import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.PublicFP

namespace VeriTile.Bench.Repairs.F8ConversionUtils.FPEquiv
open VeriTile.Triton FP.Structural Source.BitAlgebra
open Checked
noncomputable section

/-- Actual source storage channels, byte widths, masks and FP32 default profile.
The scalar carrier preserves every raw input byte, including NaN payloads.
The configured rounding/NaN cast policy defines the machine primitive. -/
abbrev conversionIO (policy : Numerics.Policy) (p : Source.Profile) (Y X : RegionName)
    (N B pid : Nat) : ProgramIO Scalar Scalar where
  inputs := [⟨p.input.toTileDType,[B],X⟩]
  outputs := [⟨p.output.toTileDType,[B],Y⟩]
  scratch := []
  buffers := [X,Y]
  read := fun i => Fin.cases (fun _ j => pid*B+j.1.val) (fun z => Fin.elim0 z) i
  inputMask := fun i => Fin.cases (fun _ j => pid*B+j.1.val<N) (fun z => Fin.elim0 z) i
  write := fun o => Fin.cases (fun _ j => pid*B+j.1.val) (fun z => Fin.elim0 z) o
  outputMask := fun o => Fin.cases (fun _ j => pid*B+j.1.val<N) (fun z => Fin.elim0 z) o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [{
    kernel := Source.kernel p Y X N B
    backend := (KernelContract.model policy p).withDefaultPrecision .fp32
    pids := fun ax => if ax=0 then pid else 0 }]
  encode := id
  observe := fun _ _ => id

/-- Independent scalar bit formula; its definition does not mention IO or execution. -/
def bitFormula (policy : Numerics.Policy) (p : Source.Profile) (x : Scalar) : Scalar :=
  pack (conversionBits policy p (unpack p.inputWidth x))

/-- Independent conversion on the source-width bytes of each active input. -/
def conversionFormula (policy : Numerics.Policy) (p : Source.Profile)
    (Y X : RegionName) (N B pid : Nat)
    (xs : (conversionIO policy p Y X N B pid).Inputs) :
    (conversionIO policy p Y X N B pid).Outputs :=
  fun o => Fin.cases
    (fun j => ofFloat p.output
      (bitFormula policy p (toFloat p.input (xs 0 j))))
    (fun z => Fin.elim0 z) o

/-- Whole typed source, source-access safety, independent output words and
complete exterior frame. Both decoder stores are retained. -/
theorem complete_kernel_bit_formula (policy : Numerics.Policy) (p : Source.Profile)
    (Y X : RegionName) (N B pid : Nat) :
    (conversionIO policy p Y X N B pid).Implements
      (conversionFormula policy p Y X N B pid) :=
  PublicFP.implements policy p Y X N B pid

/-- No floating transformation is used; casts are defined bit-machine primitives. -/
def deferredNumericalObligations : List String := []
#specNonCircular bitFormula avoiding [Source.kernel,FP.Structural.exec]
#axiomsClean complete_kernel_bit_formula
#stmtSurfaceSubset complete_kernel_bit_formula ⊆
  [Numerics.Policy, Source.Profile, RegionName, Scalar, conversionIO,
   conversionFormula, ProgramIO.Implements]

end
end VeriTile.Bench.Repairs.F8ConversionUtils.FPEquiv

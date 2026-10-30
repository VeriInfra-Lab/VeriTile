import bench.tritonbench_g.f8_conversion_utils.Proofs.RealProjection
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.F8ConversionUtils.ConversionIO
open VeriTile.Triton
noncomputable section

def real (p : Source.Profile) (Y X : RegionName) (N B : Nat) : MaskedKernelIO₁ where
  kernel := (Source.kernel p Y X N B).eraseDType
  projection := PublicReal.source_projection p Y X N B
  inp := X
  out := Y
  B := B
  read := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < N
  scratch := []

end
end VeriTile.Bench.Repairs.F8ConversionUtils.ConversionIO

namespace VeriTile.Bench.Repairs.F8ConversionUtils.PublicFP
open VeriTile.Triton FP.Structural Source.BitAlgebra
open Checked
noncomputable section

/-- Independent byte-level conversion: only the source input width is read,
and exactly the destination word is produced. -/
def formula (policy : Numerics.Policy) (p : Source.Profile) (x : Scalar) : Scalar :=
  pack (conversionBits policy p (unpack p.inputWidth x))

/-- This helper names the structural interface used by the public reader.
Storage dtype, active windows and original source are all explicit data. -/
abbrev io (policy : Numerics.Policy) (p : Source.Profile) (Y X : RegionName)
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

def target (policy : Numerics.Policy) (p : Source.Profile) (Y X : RegionName)
    (N B pid : Nat) (xs : (io policy p Y X N B pid).Inputs) :
    (io policy p Y X N B pid).Outputs :=
  fun o => Fin.cases (fun j => ofFloat p.output (formula policy p (toFloat p.input (xs 0 j)))) (fun z => Fin.elim0 z) o

end
end VeriTile.Bench.Repairs.F8ConversionUtils.PublicFP

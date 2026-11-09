import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

/-! Numerical guards read actual typed-load registers after the complete
source prefix. The numerical rewrite must still match these operands. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SourceSites
open VeriTile.Triton FP.ProfiledRewrite

def lastFP32Load (code : List ComputeStmt) : Option RegName :=
  match code with
  | []=>none
  | statement::rest=>
    match lastFP32Load rest with
    | some name=>some name
    | none=>match statement with
      | .assign _ _ name (.compute (.load .fp32 _ _))=>some name
      | _=>none

def loadedAddition (BC : Nat) (code : List ComputeStmt) (out : RegName) : Option Site := do
  let before:=Source.prefixBefore code out
  let input ← lastFP32Load before
  return .addFP32 before [BC,BC] (.ref .real [BC,BC] out) (.ref .real [BC,BC] input)
end VeriTile.Bench.Repairs.ChunkGlaFwd.SourceSites

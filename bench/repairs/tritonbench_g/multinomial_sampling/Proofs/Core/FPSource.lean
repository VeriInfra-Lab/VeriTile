import bench.repairs.tritonbench_g.multinomial_sampling.Kernels
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast

namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open VeriTile.Triton Lean

/-! Proof-side configurations instantiate the single canonical source graph.
Only FP32 belongs to the selected repaired Python contract. Other storage
instantiations remain available to the preserved historical helpers. -/
def original := Source.original
def original16 := Source.original16
def originalBfloat16 := Source.originalBfloat16
def original64 := Source.original64

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

def Storage.dtype : Storage → FloatDType
  | .fp32 => .real | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .real

structure Parameters where
  scores : RegionName
  seeds : RegionName
  offsets : RegionName
  indices : RegionName
  output : RegionName
  strideScoreBatch : Nat
  strideScoreToken : Nat
  strideIndexBatch : Nat
  strideIndexToken : Nat
  batches : Nat
  tokens : Nat
  block : Nat
  blockTokens : Nat
  blockDomain : Random.ArangeSizeDomain block
  tokensDomain : Random.ArangeSizeDomain blockTokens

def source (d : Storage) (p : Parameters) : ComputeKernel :=
  let kernel := match d with
    | .fp32 => original | .fp16 => original16 | .bf16 => originalBfloat16 | .fp64 => original64
  kernel p.scores p.seeds p.offsets p.indices p.output
    p.strideScoreBatch p.strideScoreToken p.strideIndexBatch p.strideIndexToken
    p.batches p.tokens p.block p.blockTokens p.blockDomain p.tokensDomain

def prologue (d : Storage) (p : Parameters) : List ComputeStmt :=
  (source d p).surfaceBody.take 9

def body (d : Storage) (p : Parameters) : List ComputeStmt :=
  match ((source d p).surfaceBody.drop 9).head? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []

def epilogue (d : Storage) (p : Parameters) : List ComputeStmt :=
  (source d p).surfaceBody.drop 10

theorem source_decomposition (d : Storage) (p : Parameters) :
    (source d p).surfaceBody = prologue d p ++
      [.forRange "b_idx" 0 p.tokens p.blockTokens (body d p)] ++ epilogue d p := by
  cases d <;> rfl

#axiomsClean source_decomposition
#axiomsClean source
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue

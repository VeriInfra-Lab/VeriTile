import bench.tritonbench_g.chunk_gated_attention.Proofs.HProjection
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSourceFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
open scoped VeriTile.Spec
set_option maxRecDepth 32000
set_option maxHeartbeats 1200000

/-- The actual typed dot operand consumed by the fp32 carry addition. -/
def blockDot (dtype : FloatDType) (BK BT BV : Nat) : Op .real [BK,BV] :=
  .castFloat .fp32 .real (.dotTyped dtype dtype .fp32 (batch:=[])
    (.ref dtype.toTileDType [BK,BT] "b_k") (.ref dtype.toTileDType [BT,BV] "b_v")
    {allowTF32:=some Bool.false})

def requirements (source : ComputeKernel) (dtype : FloatDType) (BK BT BV : Nat) : List Site :=
  (do
    let loop←Source.intRange? source.surfaceBody "i_t"
    let branch←Source.conditional? loop.body (fun _=>Bool.true)
    let selected←match branch.condition with
      | .constBool value=>some value
      | _=>none
    let arm:=if selected then branch.yes else branch.no
    return Site.intRange loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.branch branch.before branch.condition selected
        (.addFP32 (Source.prefixBefore arm "b_h" 1) [BK,BV]
          (.ref .real [BK,BV] "b_h") (blockDot dtype BK BT BV)))).toList

def original (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) : Program where
  kernel:=kernel {p with swapped:=Bool.false} r dot gate history seed final
  domain:=requirements (kernel {p with swapped:=Bool.false} r dot gate history seed final) dot p.BK p.BT p.BV

def candidate (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) : Program :=
  {original p r dot gate history seed final with
    kernel:=kernel {p with swapped:=Bool.true} r dot gate history seed final}

theorem domain_present (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (original p r dot gate history seed final).domain.length=1 := by rfl

theorem equivalent (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) (R : FP.ScalarArithmetic.Rules) :
    original p r dot gate history seed final ≡[R] candidate p r dot gate history seed final := by
  cases p with
  | mk layout base gateK hasSeed storeFinal swapped =>
    cases gateK <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#axiomsClean domain_present
#axiomsClean equivalent
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSourceFP

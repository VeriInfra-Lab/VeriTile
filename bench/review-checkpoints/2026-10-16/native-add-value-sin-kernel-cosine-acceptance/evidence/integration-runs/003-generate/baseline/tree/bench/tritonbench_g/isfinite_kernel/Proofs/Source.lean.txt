import bench.tritonbench_g.isfinite_kernel.Kernels
import VeriTile.Meta.StatementAudit

/-! Parameterized views of the already audited complete source. All eleven
source parameters remain explicit, including num_tasks and both order fields.
Only the proven rank-one order-zero domain is selected. No source statements
are replaced by a mathematical helper or by a prepared register state. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton

def arms (k : ComputeKernel) : List ComputeStmt × List ComputeStmt :=
  match k.surfaceBody.drop 2 with
  | [.ifThenElse _ yes no] => (yes, no)
  | _ => ([], [])

def loopBody (k : ComputeKernel) : List ComputeStmt :=
  match (arms k).2 with
  | [_, .forRangeDyn _ _ _ _ body] => body
  | _ => []

def before (p : Parameters) : List ComputeStmt := (original32 p).surfaceBody.take 2
def oneBody32 (p : Parameters) : List ComputeStmt := (arms (original32 p)).1
def oneBody64 (p : Parameters) : List ComputeStmt := (arms (original64 p)).1
def loopBody32 (p : Parameters) : List ComputeStmt := loopBody (original32 p)
def loopBody64 (p : Parameters) : List ComputeStmt := loopBody (original64 p)
def numCtas : ComputeStmt := .assign .nat [] "num_ctas" (.alg (.numPrograms 0))
def header (p : Parameters) : List ComputeStmt := before p ++ [numCtas]

def countOp (p : Parameters) : Op .nat [] :=
  .div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat p.size) (.constNat p.block)) (.constNat 1))
    (.constNat p.block)

theorem before_exact (p : Parameters) : before p =
    [.assign .nat [] "pid" (.alg (.programId 0)),
     .assign .nat [] "num_tiles0" (.alg (countOp p))] := rfl

theorem source32_split (p : Parameters) :
    (original32 p).surfaceBody = before p ++
      [.ifThenElse (.constBool p.oneTile) (oneBody32 p)
        [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
          (.constNat 1) (loopBody32 p)]] := rfl

theorem source64_split (p : Parameters) :
    (original64 p).surfaceBody = before p ++
      [.ifThenElse (.constBool p.oneTile) (oneBody64 p)
        [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
          (.constNat 1) (loopBody64 p)]] := rfl

theorem one32_length (p : Parameters) : (oneBody32 p).length = 8 := rfl
theorem one64_length (p : Parameters) : (oneBody64 p).length = 8 := rfl
theorem loop32_length (p : Parameters) : (loopBody32 p).length = 8 := rfl
theorem loop64_length (p : Parameters) : (loopBody64 p).length = 8 := rfl

#axiomsClean before_exact
#axiomsClean source32_split
#axiomsClean source64_split
#axiomsClean one32_length
#axiomsClean one64_length
#axiomsClean loop32_length
#axiomsClean loop64_length
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric

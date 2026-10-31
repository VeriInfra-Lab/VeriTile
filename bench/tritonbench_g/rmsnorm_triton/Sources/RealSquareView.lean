import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Core.Integer

/-! Proof-only local Real rewrite on a view of the shared source. The source
and FP executions retain libdevicePowi. The complete caller connection is
proved by variance_assignment, variance_run and original_run.
Only the two flat forRange blocks in this source are traversed. -/
namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.RealSquareView
open VeriTile.Triton
def square {sh : TileShape} (e : Op .real sh) : Op .real sh :=
  match e with
  | .libdevicePowi b x exponent =>
    match b with
    | .scalarR => match exponent with
      | .constInt n => if n=2 then
        .mul .real (IntegerExpr.sameShape _) (.castFloat .real .real x) (.castFloat .real .real x)
        else e
      | _ => e
    | _ => e
  | _ => e

def sum {sh : TileShape} : Op .real sh → Op .real sh
  | .add .real b a rhs => .add .real b a (square rhs)
  | e => e

def assignment : Stmt → Stmt
  | .assign .real sh name value => .assign .real sh name (sum value)
  | statement => statement

def binding : Stmt → Stmt
  | .forRange idx a b c body => .forRange idx a b c (body.map assignment)
  | statement => assignment statement

def bindings (statements : List Stmt) : List Stmt := statements.map binding

def ofKernel (source : Kernel) : ComputeKernel :=
  ComputeKernel.fromKernelBody source.inputs source.outputs (bindings source.body)
end VeriTile.Bench.TritonBenchG.RmsnormTriton.RealSquareView

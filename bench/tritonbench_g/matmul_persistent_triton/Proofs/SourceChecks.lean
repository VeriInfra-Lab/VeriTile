import bench.tritonbench_g.matmul_persistent_triton.Proofs.Parameters

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.SourceChecks
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeDyn _ _ _ _ body => body
  | _ => []

def thenBody : ComputeStmt → List ComputeStmt
  | .ifThen _ body | .ifThenElse _ body _ => body
  | _ => []

/-- Independent typed fused-dot expectation. There is one opaque dotAcc,
with the existing accumulator as its third operand, under fp32 computation.
The fp16/f8e4 input channels are not replaced by fp32 casts or by dot-plus-add. -/
def expectedDot (storage : TypedSource.Storage) (BM BN BK : Nat) : ComputeStmt :=
  match storage with
  | .fp16 => .assign .real [BM, BN] "accumulator" (.compute (.alg .fp32
      (.dotAcc .fp16 .fp16 .real (batch := [])
        (.ref .fp16 [BM, BK] "a") (.ref .fp16 [BK, BN] "b")
        (.ref .real [BM, BN] "accumulator"))))
  | .f8e4 => .assign .real [BM, BN] "accumulator" (.compute (.alg .fp32
      (.dotAcc .f8e4 .f8e4 .real (batch := [])
        (.ref .f8e4 [BM, BK] "a") (.ref .f8e4 [BK, BN] "b")
        (.ref .real [BM, BN] "accumulator"))))

/-- The negative initial values and data-dependent persistent trip count
are source AST facts for every dimension/configuration and both profiles. -/
theorem persistent_structure (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :
    let body := (p.source storage v).surfaceBody
    let loop := loopBody body[16]!
    body.length = 17 ∧
    body[7]! = .assign .int [] "tile_id" (.alg (.sub .int .nil
      (.castNatToInt (.ref .nat [] "start_pid")) (.constInt (p.SMS : Int)))) ∧
    body[8]! = .assign .int [] "ki" (.alg (.constInt (-1))) ∧
    body[16]! = .forRangeDyn "_i" (.constNat 0)
      (.mul .nat .nil (.ref .nat [] "k_tiles") (.ref .nat [] "tiles_per_SM"))
      (.constNat 1) loop ∧
    loop.length = 9 ∧ loop[7]! = expectedDot storage p.BM p.BN p.BK ∧
    (thenBody loop[8]!).length = 7 := by
  cases storage <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Input/output region names are the three source arguments, for both arms. -/
theorem io_ports (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :
    (p.source storage v).inputs = [p.A, p.B] ∧
      (p.source storage v).outputs = [p.C] := by
  cases storage <;> cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.SourceChecks

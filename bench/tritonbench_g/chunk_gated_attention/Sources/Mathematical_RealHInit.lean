import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHLoop

/-! The original initialization, including zero-plus-seed rather than a seed
assignment. Optional seed storage is read before any output write, so it need
not be disjoint from output storage. Launch order is value, key, head. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHInit
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def starting (p : Layout) (l : Launch) (a padding : Nat → ℝ) (hasSeed : Bool) : StateValue p :=
  if hasSeed then seedData p l a padding else fun _ _ => 0

theorem if_true (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.true) body) u = stepStmts body u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,↓reduceIte]

theorem if_false (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.false) body) u = some u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,Bool.false_eq_true,↓reduceIte]

theorem starting_expected (p : Layout) (l : Launch) (a : Inputs) (gateK hasSeed : Bool) :
    starting p l a.seed (fun _ => 0) hasSeed = expected p l a gateK hasSeed 0 := by
  funext e j
  cases hasSeed <;> simp [starting,expected,seedData,RealHMath.seed]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHInit

import bench.tritonbench_g.embedding_triton_kernel.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.SourceTypes
open VeriTile.Triton
open VeriTile.Bench.Repairs.EmbeddingTritonKernel
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Both programs are the current repair and its integer-index candidate. -/
def selected (swapped : Bool) : ComputeKernel :=
  let p : Parameters := ⟨"weight","ids","out",-2,1,2,2,4,2,4⟩
  if swapped then indexCommutedSource .int32 p else source .int32 p

def loads {d : TileDType} {sh : TileShape} : Op d sh → List (TileDType × Bool × Bool)
  | .load dt _ (.mask _) _ => [(dt,Bool.true,Bool.false)]
  | .load dt _ (.maskOther _ _) _ => [(dt,Bool.true,Bool.true)]
  | .load dt _ _ _ => [(dt,Bool.false,Bool.false)]
  | .castFloat _ _ e => loads e
  | _ => []

def accesses : List ComputeStmt → List (String × TileDType × Bool × Bool)
  | [] => []
  | .assign _ _ _ e :: rest =>
      ((e.toAlgorithm?.toOption.map loads).getD []).map (fun (d,m,o) => ("load",d,m,o)) ++ accesses rest
  | .store d _ _ _ .none :: rest => ("store",d,Bool.false,Bool.false) :: accesses rest
  | .store d _ _ _ _ :: rest => ("store",d,Bool.true,Bool.false) :: accesses rest
  | .forRange _ _ _ _ body :: rest => accesses body ++ accesses rest
  | _ :: rest => accesses rest

/-- Inspect the actual full typed loop: signed int32 token IDs, matching
payload/output precision, both explicit load defaults and the output mask. -/
theorem source_accesses (swapped : Bool) :
    accesses (selected swapped).surfaceBody =
      [("load",.int,Bool.true,Bool.true),
       ("load",.real,Bool.true,Bool.true),
       ("store",.real,Bool.true,Bool.false)] := by
  cases swapped
  all_goals simp [selected,source,indexCommutedSource,Region.cast,ComputeKernel.surfaceBody,accesses,loads,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

/-- Inspect the signed subtraction/comparisons rather than a Nat surrogate. -/
def signedOps {d : TileDType} {sh : TileShape} : Op d sh → List (String × TileDType)
  | .sub _ _ _ _ => [("sub",d)]
  | .ge .int _ _ _ => [("ge",.int)]
  | .lt .int _ _ _ => [("lt",.int)]
  | .boolAnd _ a b => signedOps a ++ signedOps b
  | _ => []

def assignments : List ComputeStmt → List (String × List (String × TileDType))
  | [] => []
  | .assign _ _ name e :: rest =>
      (name,(e.toAlgorithm?.toOption.map signedOps).getD []) :: assignments rest
  | .forRange _ _ _ _ body :: rest => assignments body ++ assignments rest
  | _ :: rest => assignments rest

theorem signed_id_operations (swapped : Bool) :
    (assignments (selected swapped).surfaceBody).filter
      (fun x => x.1=="id_mask" || x.1=="token_ids") =
      [("token_ids",[]),("id_mask",[("ge",.int),("lt",.int)]),("token_ids",[("sub",.int)])] := by
  cases swapped
  all_goals simp [selected,source,indexCommutedSource,Region.cast,ComputeKernel.surfaceBody,assignments,signedOps,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.toOption]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.SourceTypes

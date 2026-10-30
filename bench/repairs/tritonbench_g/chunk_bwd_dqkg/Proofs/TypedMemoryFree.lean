import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedMemoryFree
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety TypedSafetyBasics
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Source slices, obtained from the one authoritative quotation. -/
def body (c : Config) : List ComputeStmt :=
  match ((kernel c).surfaceBody[14]? : Option ComputeStmt) with
  | some (.forRangeDyn _ _ _ _ code)=>code
  | _=>[]
theorem body_eq (c : Config) : body c=(parts c).body := rfl

theorem entry_suffix (M : Algebra α) (dot : Dot α) (c : Config)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds (((kernel c).surfaceBody.drop 9).take 5) s := by
  apply pure_trace
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.take,List.mem_cons,List.not_mem_nil,or_false,
    forall_eq_or_imp,forall_eq]
  simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  try simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold]

theorem body_suffix (M : Algebra α) (dot : Dot α) (c : Config)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds ((body c).drop 8) s := by
  apply pure_trace
  simp only [body,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.getElem?_cons_zero,List.getElem?_cons_succ,
    List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]
  simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  try simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold]

theorem post_suffix (M : Algebra α) (dot : Dot α) (c : Config)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds (((kernel c).surfaceBody.drop 19).take 13) s := by
  apply pure_trace
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.take,List.mem_cons,List.not_mem_nil,or_false,
    forall_eq_or_imp,forall_eq]
  simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  try simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold]

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedMemoryFree

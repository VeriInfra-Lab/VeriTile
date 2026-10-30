import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourcePreservation
open VeriTile.Triton
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue (Parameters)
noncomputable section

theorem complete_original (p : Parameters) : (source p).surfaceBody.length = 14 := rfl
theorem complete_candidate (p : Parameters) : (reuseSource p).surfaceBody.length = 14 := rfl

/-- The new quotation keeps the Python integer select and signed loop. The
historical source used natural loop registers and added explicit load casts;
its AST is not the original Python body. Actual memory/IO proofs are separate. -/
theorem differs_from_historical (p : Parameters) :
    source p ≠ VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source p := by
  intro h
  have tag := congrArg (fun k : ComputeKernel =>
    match k.surfaceBody[13]? with
    | some (.forRangeInt _ _ _ _ _) => Bool.true
    | _ => Bool.false) h
  change Bool.true = Bool.false at tag
  cases tag

#axiomsClean complete_original
#axiomsClean complete_candidate
#axiomsClean differs_from_historical
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourcePreservation

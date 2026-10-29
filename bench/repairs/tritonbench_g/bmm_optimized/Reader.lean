import bench.repairs.tritonbench_g.bmm_optimized.Correct
import bench.repairs.tritonbench_g.bmm_optimized.FPEquiv
import bench.repairs.tritonbench_g.bmm_optimized.Proofs.Ownership
namespace BmmOptimizedReader
open Lean Elab Command
open VeriTile.Bench.Repairs.BmmOptimized
open Native Host
example (call : Call) (domain : Valid call) (c : Config) (p : Instructions)
    (callerDevice : Nat) (initial : Bytes) :
    Public.NativeClaim call c p p callerDevice initial (invoke call c p callerDevice initial) :=
  Public.whole_native call domain c p callerDevice initial
example (call : Call) (domain : Valid call) (c : Config) (actual target : Instructions)
    (ports : Obligations actual target) (callerDevice : Nat) (initial : Bytes) :
    Public.NativeClaim call c actual target callerDevice initial (invoke call c actual callerDevice initial) :=
  Public.whole_fp call domain c actual target ports callerDevice initial
example (call : Call) (domain : Valid call) (c : Config)
    (callerDevice : Nat) (initial : Memory ℝ) :
    Public.RealClaim call c callerDevice initial (invokeReal call c callerDevice initial) :=
  Public.whole_real call domain c callerDevice initial
#print Public.whole_real
#print Public.whole_native
#print Public.whole_fp
#print Host.Valid
#print IndexABI
#print Native.Instructions
#print Native.Obligations
#auditStmt Public.whole_real
#auditStmt Public.whole_native
#auditStmt Public.whole_fp
#specNonCircular Host.product avoiding [Host.prepare,Host.prepareBytes,Host.invoke,Host.invokeReal,source,sourceValue,Native.selected]
#specNonCircular Host.expected avoiding [Host.prepare,Host.prepareBytes,Host.invoke,Host.invokeReal,source,sourceValue,Native.selected]
#specNonCircular Host.Valid avoiding [Host.invoke,Host.invokeReal,source,sourceValue,Native.selected]
#axiomsClean config_bounds
#axiomsClean laneLoop_add
#axiomsClean blockLoop_flat
#axiomsClean source_ordered
#axiomsClean source_real
#axiomsClean sourcePrograms_inputs
#axiomsClean sourcePrograms_commit
#axiomsClean address_injective
#axiomsClean address_bound
#axiomsClean swizzle_injective_nat
#axiomsClean swizzle_covers
#axiomsClean source_grid_covers
#axiomsClean source_masks
#axiomsClean active_a_bound
#axiomsClean active_b_bound
#axiomsClean formed_addresses
#axiomsClean source_group_intermediates
#axiomsClean output_byte_signed64
#axiomsClean source_ownership
#axiomsClean output_owner
#axiomsClean any_program_order
#axiomsClean allWrites_covers
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean source_grid_ordered
#axiomsClean source_grid_real
#axiomsClean Native.load_store
#axiomsClean Native.load_selected
#axiomsClean Native.selected_read
#axiomsClean Native.selected_frame
#axiomsClean Native.selected_equiv
#axiomsClean Host.view_address_bound
#axiomsClean Host.packedRead_address
#axiomsClean Host.load_prepare
#axiomsClean Host.whole_real
#axiomsClean Host.whole_native
#axiomsClean Host.whole_frame
#axiomsClean Host.returned_tensor
#axiomsClean correct_real
#axiomsClean correct_native
#axiomsClean fp_equivalent
#axiomsClean Public.whole_real
#axiomsClean Public.whole_native
#axiomsClean Public.whole_fp
run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"BMM_READER_MODULES_JSON {(Json.arr names).compress}"
run_cmd do
  for short in #[`Public.whole_real,`Public.whole_native,`Public.whole_fp,`formed_addresses,
      `source_masks,`output_owner,`any_program_order,`source_group_intermediates] do
    let n := `VeriTile.Bench.Repairs.BmmOptimized ++ short
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"BMM_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
end BmmOptimizedReader

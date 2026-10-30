import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.HostCorrect
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.HostFPEquiv
namespace MatmulDequantizeWhole001Reader
open Lean Elab Command
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
#check @Correct.host_gptq
#check @Correct.host_s2
#check @Correct.host_dequant32
#check @Correct.host_dequant16
#check @Correct.s1
#check @FPEquiv.host_gptq
#check @FPEquiv.host_s2
#check @FPEquiv.host_dequant32
#check @FPEquiv.host_dequant16
#check @FPEquiv.s1
#print Host.Admissible
#print Host.Tensor.Valid
#print Host.first
#print Host.second
#print Host.dequant16
#print Host.dequant32
#print Host.returned
#print Host.initialization
#print Host.Initialized
#print Native.InputWords
#print TorchMM.RealAPI
#print TorchMM.NativeAPI
#print ABI.IndexABI
#print ABI.SourceExact
#print ABI.matmulRoots
#print ABI.dequantRoots
#print ABI.accesses
#print Native.WordISA
#print Native.Plan
#print Native.lower
#print Real.rawDequant
#print Real.rawMatmul
#axiomsClean ABI.word_exact
#axiomsClean ABI.expression_exact
#axiomsClean ABI.source_exact
#axiomsClean ABI.difference_exact
#axiomsClean ABI.rejects_masked_stride_overflow
#axiomsClean Host.finite_address
#axiomsClean Host.first_exhaustive
#axiomsClean Host.first_valid
#axiomsClean Host.distinct
#axiomsClean Host.first_return
#axiomsClean Host.second_return
#axiomsClean Host.helper_return
#axiomsClean Host.s1_return
#axiomsClean Grid.injective
#axiomsClean Grid.post_of_lanes
#axiomsClean FirstGrid.whole
#axiomsClean SecondGrid.whole
#axiomsClean DequantGrid16.whole
#axiomsClean DequantGrid32.whole
#axiomsClean RealFirst.output_formula
#axiomsClean RealSecond.output_formula
#axiomsClean RealDequant.D16.formula
#axiomsClean RealDequant.D32.formula
#axiomsClean Native.dot_feature_covered
#axiomsClean Native.dot_feature_unique
#axiomsClean Native.preserves_default_half_fma
#axiomsClean Native.preserves_live_seed
#axiomsClean Correct.host_gptq
#axiomsClean Correct.host_s2
#axiomsClean Correct.host_dequant32
#axiomsClean Correct.host_dequant16
#axiomsClean Correct.s1
#axiomsClean FPEquiv.host_gptq
#axiomsClean FPEquiv.host_s2
#axiomsClean FPEquiv.host_dequant32
#axiomsClean FPEquiv.host_dequant16
#axiomsClean FPEquiv.s1
#auditStmt Correct.host_gptq
#auditStmt Correct.host_s2
#auditStmt Correct.host_dequant32
#auditStmt Correct.host_dequant16
#auditStmt Correct.s1
#auditStmt FPEquiv.host_gptq
#auditStmt FPEquiv.host_s2
#auditStmt FPEquiv.host_dequant32
#auditStmt FPEquiv.host_dequant16
#auditStmt FPEquiv.s1
#specNonCircular Real.rawMatmul avoiding [FP.Structural.exec,Grid.run]
#specNonCircular Real.dequant avoiding [FP.Structural.exec,Grid.run]
#specNonCircular Native.lower avoiding [FP.Structural.exec,Grid.run]
run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"MATMUL_READER_MODULES_JSON {(Json.arr names).compress}"
run_cmd do
  let names := #[`Correct.host_gptq,`Correct.host_s2,`Correct.host_dequant32,`Correct.host_dequant16,`Correct.s1,`FPEquiv.host_gptq,`FPEquiv.host_s2,`FPEquiv.host_dequant32,`FPEquiv.host_dequant16,`FPEquiv.s1]
  for short in names do
    let n := `VeriTile.Bench.Repairs.MatmulDequantizeWhole001 ++ short
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"MATMUL_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
end MatmulDequantizeWhole001Reader

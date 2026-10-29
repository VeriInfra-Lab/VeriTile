import bench.repairs.tritonbench_g.matmul_dequant_int4.Correct
import bench.repairs.tritonbench_g.matmul_dequant_int4.FPEquiv
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Witnesses

open VeriTile.Bench.Repairs.MatmulDequantInt4
#axiomsClean Direct.matmul_dequant_int4_real
#axiomsClean Direct.matmul_dequant_int4_native
#axiomsClean Direct.matmul_dequant_int4_if_local
#axiomsClean Direct.checkPacked_iff
#axiomsClean Direct.checkDequant_iff
#axiomsClean Direct.checkS1_iff
#axiomsClean Direct.quant16_checked
#axiomsClean Direct.quant32_checked
#axiomsClean Direct.formed_pointer
#axiomsClean Direct.launchFacts
#axiomsClean SourceExecution.complete_run_eq
#axiomsClean Grid.schedule_nodup
#axiomsClean Grid.formed
#axiomsClean Physical.halfCodec
#axiomsClean Physical.wordCodec
#axiomsClean Host.context_exit_restores
#axiomsClean quantize_real
#axiomsClean whole_fp
#guard_msgs (drop info) in
#auditModuleAxioms

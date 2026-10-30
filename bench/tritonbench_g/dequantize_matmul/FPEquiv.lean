import bench.tritonbench_g.dequantize_matmul.PairIO
import bench.tritonbench_g.dequantize_matmul.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.DequantizeMatmul.FPEquiv
open VeriTile Triton FP.Structural KernelSource KernelModels
open scoped VeriTile.Spec
noncomputable section

abbrev original := PairIO.original
abbrev indexCommuted := PairIO.indexCommuted

/-- Both complete INT8/FP32 source grids differ only by commuting the exact
integer N-mask product. The signed-byte domain, source precision, multiplication
order, all output cells, safe executions and complete exterior frames remain. -/
specification kernel_equiv (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p) :
    original multiply rest p ≡[[]] indexCommuted multiply rest p :=
  PairEquivalence.equivalent multiply rest p hK hN sep

#print_fp_assumptions kernel_equiv
#axiomsClean kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.FPEquiv

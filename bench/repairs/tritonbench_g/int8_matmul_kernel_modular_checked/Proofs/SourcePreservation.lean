import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Kernels
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.LegacySource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/-- The direct ordinary quotation has exactly the prior full program AST.
Only the bounded source ASM notation has been replaced by its already used
strict modulo32 instruction semantics; the actual Python barrier is retained. -/
theorem direct_source_preserved (a : Region .fp32) (b : Region .nat) (c : Region .int)
    (M N K sam sak sbk sbn scm scn BM BN BK G : Nat) :
    kernel a b c M N K sam sak sbk sbn scm scn BM BN BK G =
      LegacySource.kernel a b c M N K sam sak sbk sbn scm scn BM BN BK G := rfl

#axiomsClean direct_source_preserved
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked

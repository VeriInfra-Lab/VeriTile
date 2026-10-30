import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.LegacyQuotedSource
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.LegacySource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/-- Historical cleanup result for the pre-body-copy quotation: its complete
AST equals the prior bounded expansion. The current raw source uses counted/
static loops and is proved directly in TileSource/TileComplete and AccessSafety.
The following equality is not a claim about that current source.
The pre-body-copy quotation has exactly the prior full program AST.
Only the bounded source ASM notation has been replaced by its already used
strict modulo32 instruction semantics; the actual Python barrier is retained. -/
theorem prior_direct_source_preserved (a : Region .fp32) (b : Region .nat) (c : Region .int)
    (M N K sam sak sbk sbn scm scn BM BN BK G : Nat) :
    LegacyQuotedSource.kernel a b c M N K sam sak sbk sbn scm scn BM BN BK G =
      LegacySource.kernel a b c M N K sam sak sbk sbn scm scn BM BN BK G := rfl

#axiomsClean prior_direct_source_preserved
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked

import bench.repairs.tritonbench_g.vector_addition_custom.Correct
import bench.repairs.tritonbench_g.vector_addition_custom.Misc.SourceBinding
import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.MaskedDomain

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous

/-- The optional flat IO and public FP proof use the identical canonical source.
The primary regional IO names `originalKernel` directly. -/
theorem public_source_identity (n : Nat) :
    (Correct.addIO n).kernel = (FPEquiv.original n).kernel := rfl

theorem public_candidate_identity (n : Nat) :
    (FPEquiv.commuted n).kernel = commutedKernel n := rfl

#axiomsClean public_source_identity
#axiomsClean public_candidate_identity
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous

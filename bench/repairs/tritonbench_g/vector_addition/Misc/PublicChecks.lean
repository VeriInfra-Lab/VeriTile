import bench.repairs.tritonbench_g.vector_addition.Correct
import bench.repairs.tritonbench_g.vector_addition.Misc.SourceBinding
import bench.repairs.tritonbench_g.add_example.Proofs.MaskedDomain

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous

/-- The regional/flat IO and public FP proof use the identical canonical source.
This reader integration audit is separate from the reusable proof modules. -/
theorem public_source_identity (n : Nat) :
    (Correct.addIO n).kernel = (FPEquiv.original n).kernel := rfl

theorem public_candidate_identity (n : Nat) :
    (FPEquiv.commuted n).kernel = commutedKernel n := rfl

#axiomsClean public_source_identity
#axiomsClean public_candidate_identity
end VeriTile.Bench.Repairs.VectorAddition.Contiguous

import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PairIO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.Equivalence
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat) :=
  ProgramIO.Implementation.original (PublicIO.nativeIO ops p batches visits)
    (PublicIO.Admissible p batches visits)
abbrev quotientRemainder (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate ops p batches visits)
    (PublicIO.Admissible p batches visits)

/-- Exact head-index CSE on two complete selected native source programs.
Runtime FP32 scale, all CSR occurrences, feature blocks, tagged outputs,
successful safe execution and both frames retain their full original domain. -/
specification quotient_remainder_equiv (ops : PhysicalKernel.NativePrimitives)
    (p : Parameters) (batches visits : Nat) (geometry : PublicIO.Geometry p) :
    original ops p batches visits ≡[[]] quotientRemainder ops p batches visits := by
  exact ProgramIO.Implementation.of_equivalentOn (PairIO.candidate ops p batches visits)
    (PairIO.equivalent ops p batches visits geometry)

#axiomsClean quotient_remainder_equiv
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.Equivalence

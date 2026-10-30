import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.PairResult

/-! Complete original/sum-left source equivalence on the retained kernel IO.
The original recognizes a left product and the candidate recognizes a right
product; each lowers to the identical FMA with the same multiplicand order.
Every raw FP32 word is allowed, including NaNs, infinities and signed zeros.
The ISA ports and DG reduction plan remain explicit model data. This is a
source/lowering theorem, without an additional hardware-conformance claim. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv
open VeriTile Triton FP.Structural KernelProgram KernelCarrier PairIO
open scoped VeriTile.Spec
noncomputable section

/-- Both full time loops, backward DG outputs and final DL are retained.
The four repaired seed strides and legal PID are unchanged. The empty
experimental rule set follows from identical native instruction applications,
without a finite-domain restriction or split-multiply/add-to-FMA assumption. -/
specification chunk_gate_recurrence_sumLeft_equiv (isa : Instructions)
    (plan : Reduction.ProductPlan 1024) (kind : Kind) (c : Shape)
    (s : SeedStrides) (seed : Bool) (p : Program c) (nonempty : 0<c.N) :
    original isa plan kind c s seed p ≡[[]] sumLeft isa plan kind c s seed p :=
  ProgramIO.Implementation.of_equivalent (sumLeftIO isa plan kind c s seed p)
    (ProgramPair.equivalent isa plan kind c s seed p nonempty)

#axiomsClean chunk_gate_recurrence_sumLeft_equiv
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv

import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteOutcomes

/-! Finite source acceptance candidate. All seventeen source-target dtype
profiles, the exact producer/consumer wrapper and first-error paths are covered
under explicit trusted primitive contracts and valid views. Numerical native
correspondences remain precisely registrable obligations, not proved facts.
The asserted output is the as-written physical prefix; N>1024 is not full ReLU. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteCorrect
open VeriTile.Triton HostWrapper GatedGrid FP64Host FiniteBoundary FiniteHost
noncomputable section

specification relu_finite_source_outcomes
    {α : Type} [Inhabited α] (views : TensorSemantics)
    (P : Primitives (Runtime α)) (C : PrimitivesContract views P)
    (M : FP.Structural.Algebra α) (cmp16 cmp32 : α → α → Bool)
    (num : FP64Source.Numerics) (undef : Nat → BitVec 64) (order : Nat → List Nat)
    (fault : Option (Error × Runtime α)) (x : Tensor) (s : Runtime α) (trace : List Event)
    (domain : SourceDomain x) (ready : Ready views x s)
    (complete : (order (gridCount (numel x) 1024)).Perm (List.range (gridCount (numel x) 1024))) :
    Outcome views P M cmp16 cmp32 num fault x s trace
      (relu (FiniteHost.withLaunch P M cmp16 cmp32 num undef order fault) x ⟨s, trace⟩) :=
  all_outcomes views P C M cmp16 cmp32 num undef order fault x s trace domain ready complete

#axiomsClean relu_finite_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteCorrect

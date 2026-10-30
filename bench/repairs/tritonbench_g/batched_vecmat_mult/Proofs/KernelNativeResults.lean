import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelCore
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelProfile
import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativeContract

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelNativeResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- Raw FP32 words determine the complete contracted native graph. The
profile consumes payload-erased structure; all exterior cells are preserved. -/
theorem batched_vecmat_grid_native (cfg : FP.Config)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (before : State ValueProofs.Word) (index : Whole.IntegerAgreement p)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) :
    ∃ final, SourceGraph.runGrid rest (KernelProfile.toPlans profile) p
        (SourceGraph.Native.liftState before before.mem)=some final ∧
      FiniteExecution.grid (KernelProfile.toPlans profile) p
        (SourceGraph.Native.liftState before before.mem)=some final.mem ∧
      (∀ m : Fin p.M, ∀ n : Fin p.N,
        SourceGraph.Native.decodeMemory cfg final.mem p.output (m.val*p.N+n.val)=
          .mk .real (SourceGraph.Native.expected cfg (KernelProfile.toPlans profile) p before.mem m.val n.val)) ∧
      (∀ r i, SourceGraph.GridOutside p r i →
        SourceGraph.Native.decodeMemory cfg final.mem r i=before.mem r i) := by
  obtain ⟨final,he,hfinite,hv,hf⟩ := KernelCore.graph_complete rest
    (KernelProfile.toPlans profile) p (SourceGraph.Native.liftState before before.mem)
    index separate hBM hBN hM hN
  refine ⟨final,he,hfinite,?_,?_⟩
  · intro m n
    unfold SourceGraph.Native.decodeMemory
    rw [hv m n]
    rfl
  · intro r i outside
    unfold SourceGraph.Native.decodeMemory
    rw [hf r i outside]
    exact congrFun (congrFun (SourceGraph.Native.decode_lift cfg before.mem) r) i

#specNonCircular SourceGraph.Native.expected avoiding [source,FP.Structural.exec,SourceGraph.runGrid]
#axiomsClean batched_vecmat_grid_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelNativeResults

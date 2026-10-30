import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Correct
open VeriTile Triton PublicIO
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- Every active output of the complete original three-axis grid is Doutᵀ @ A
plus optional residual. The job axis retains independent CTA read replicas,
exact strides, shortened last chunks, source access safety and exterior frame. -/
specification bmm_chunk_bwd_real (h : HostCall) (choice : Fin 9)
    (_ha : h.aStorage = .fp32) (_hd : h.doutStorage = .fp32)
    (distinct : RealViews.OutputDistinct (tuning choice).BM (tuning choice).BN (HostContract.parameters h))
    (separate : HostGrid.InputsSeparate (HostContract.parameters h))
    (writes : HostGrid.SeparateWrites (tuning choice).BM (tuning choice).BN
      (HostContract.parameters h) (jobs h (tuning choice))) :
    Spec.Real (logicalIO h (tuning choice) ⊨ fun xs q row col =>
      (∑ k : Fin h.chunk, if k.val < chunkLength h (tuning choice) q
        then xs.dout q row k * xs.a q k col else 0) +
        if h.hasResidual then xs.residual q row col else 0) :=
  logical_implements h choice distinct separate writes

#axiomsClean bmm_chunk_bwd_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Correct

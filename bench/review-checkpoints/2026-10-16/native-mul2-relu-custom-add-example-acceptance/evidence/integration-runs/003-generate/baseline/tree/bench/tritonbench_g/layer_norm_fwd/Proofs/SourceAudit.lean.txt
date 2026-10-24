import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.SourceChecks
import bench.tritonbench_g.layer_norm_fwd.Proofs.SeedSource
import bench.tritonbench_g.layer_norm_fwd.FPEquiv

/-! Axiom audit for source lowering, full execution equivalence and the
explicit complete-source typed consistency witness. Real IO has its own audit. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean SeedSource.original_signed_seed
#axiomsClean SeedSource.original_second_signed_seed
#axiomsClean original_toAlgorithm_supported
#axiomsClean candidate_toAlgorithm_supported
#axiomsClean FPProof.prefix_run
#axiomsClean FPProof.suffix
#axiomsClean FPProof.whole_state
#axiomsClean Equivalence.layer_norm_fwd_fp32_equiv
#axiomsClean Equivalence.layer_norm_fwd_fp32_nonempty

end VeriTile.Bench.TritonBenchG.LayerNormFwd

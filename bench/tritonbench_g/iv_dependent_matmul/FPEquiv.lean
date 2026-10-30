import bench.tritonbench_g.iv_dependent_matmul.Kernels
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Host
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Safety
import bench.tritonbench_g.iv_dependent_matmul.Proofs.IndexOverflow
import bench.tritonbench_g.iv_dependent_matmul.Proofs.LegacyFP

/-! Public FP contract. Source precision is FP32 inputs/accumulator/output,
with the original terminal FP16 narrowing followed by FP32 storage. Every
schedule, ceil-K tail, active output edge, grid instance and returned tensor
is included. `Graph.source` is Dot+Add; `Graph.accumulator` is a separate
compiled DotAcc graph. No cross-graph or hardware numerical equivalence is
assumed. Dot primitives remain explicit; software instances are selectable.

The original wrapper has no index-size guard. IndexOverflow proves a valid
large-output active store underflows signed i32 arithmetic. Thus the following
explicitly bounded contracts earn no unrestricted whole-task acceptance;
classification is source_defect_pending_repair. Original Python is unchanged. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPEquiv
open VeriTile.Triton FP.Structural FPValue Kernels Whole
noncomputable section

specification fp32_whole_host_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (indices : IndexDomain r s allocation) :
    ∃ final, host rest cfg D F graph r s=some (outputMetadata r,final) ∧
      Result cfg D F graph r s final ∧ SourceSafety r s allocation := by
  obtain ⟨final,he,hp⟩ := host_correct rest cfg D F graph r s allocation
  exact ⟨final,he,hp,source_safety r s allocation indices⟩

/-- Same-graph schedule equivalence is structural and needs no numerical
associativity, reassociation, FMA fusion, TF32 or rounding assumption. -/
specification same_graph_schedule_equivalence [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (graph : Graph) (r : Request) (s : State Word)
    (allocation : Allocation r s) (_indices : IndexDomain r s allocation) (a b : Schedule) :
    ∃ x y, host rest cfg D F graph {r with schedule:=a} s=some (outputMetadata r,x) ∧
      host rest cfg D F graph {r with schedule:=b} s=some (outputMetadata r,y) ∧ x.mem=y.mem :=
  host_schedule_equivalence rest cfg D F graph r s allocation a b

/-- Concrete sequential software arithmetic, retaining both half conversions;
this is not an assertion that a tensor-core instruction uses this reduction. -/
specification sequential_software_fp32_host [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (graph : Graph) (r : Request) (s : State Word) (allocation : Allocation r s)
    (indices : IndexDomain r s allocation) :
    ∃ final, host rest cfg (softwareDot cfg) (softwareDotAcc cfg) graph r s=some (outputMetadata r,final) ∧
      Result cfg (softwareDot cfg) (softwareDotAcc cfg) graph r s final ∧ SourceSafety r s allocation :=
  fp32_whole_host_values rest cfg (softwareDot cfg) (softwareDotAcc cfg) graph r s allocation indices

#axiomsClean fp32_whole_host_values
#axiomsClean same_graph_schedule_equivalence
#axiomsClean sequential_software_fp32_host
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPEquiv

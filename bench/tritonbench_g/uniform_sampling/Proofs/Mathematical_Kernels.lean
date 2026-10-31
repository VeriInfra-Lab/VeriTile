import VeriTile.Triton.DSL
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_SourceDomain

/-! Shared storage tags and the proved legal-block adapter. The executable
repair source is maintained only in Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedSource
open Lean VeriTile.Triton

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq

theorem arange_domain {B : Nat} (hB : SourceDomain.LegalBlock B) :
    Random.ArangeSizeDomain B := by
  have hf := SourceDomain.legalBlock_fields hB
  obtain ⟨k, _, hk⟩ := hB
  exact ⟨hf.1, hf.2.1, k, hk⟩

#axiomsClean arange_domain

end VeriTile.Bench.TritonBenchG.UniformSampling.TypedSource

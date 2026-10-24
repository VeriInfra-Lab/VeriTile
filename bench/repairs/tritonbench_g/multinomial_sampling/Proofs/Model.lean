import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Native
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource

namespace VeriTile.Bench.Repairs.MultinomialSampling
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

abbrev entry := @Host.invoke

/-- Input-only finite-width address domain for this fixed source. It is an
explicit native ABI precondition, not a numerical callback or a general claim
about overflowing/huge grids. It includes inactive formed pointers. -/
def IndexABI {α Store Byte : Type} [Inhabited α] (api : Physical.API α Store Byte)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Native.Preparation api p a s) : Prop :=
  let q := Host.params p a
  p.batches < 2^31 ∧ p.tokens < 2^31 ∧
  p.strideScoreBatch < 2^31 ∧ p.strideScoreToken < 2^31 ∧
  q.strideIndexBatch < 2^31 ∧ q.strideIndexToken < 2^31 ∧
  call.scores.base%4=0 ∧ call.indices.base%8=0 ∧
  call.seeds.base%8=0 ∧ call.offsets.base%8=0 ∧ call.output.base%8=0 ∧
  (∀ r, r < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid q*q.block → r < 2^31 ∧
    call.seeds.base+8*r < 2^64 ∧ call.offsets.base+8*r < 2^64 ∧ call.output.base+8*r < 2^64 ∧
    ∀ t, t < count q*q.blockTokens → t < 2^31 ∧
      r*p.strideScoreBatch+t*p.strideScoreToken < 2^31 ∧
      r*q.strideIndexBatch+t*q.strideIndexToken < 2^31 ∧
      call.scores.base+4*(r*p.strideScoreBatch+t*p.strideScoreToken) < 2^64 ∧
      call.indices.base+8*(r*q.strideIndexBatch+t*q.strideIndexToken) < 2^64)

theorem source_exact : Source.original = original := rfl

theorem kernel_source_checked : Source.original = original := rfl
#axiomsClean kernel_source_checked
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling

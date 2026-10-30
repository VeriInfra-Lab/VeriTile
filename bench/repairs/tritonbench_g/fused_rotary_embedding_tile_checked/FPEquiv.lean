import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.FPEquiv
open VeriTile Triton FP.Structural FusedRotaryEmbedding
open scoped VeriTile.Spec
noncomputable section

def original (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config) :=
  ProgramIO.Implementation.original (IOPair.io rest p c) (PublicIO.metadataDomain c)
def candidate (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant rest p c) (PublicIO.metadataDomain c)

/-- Reuse Q's first-half address in the second half of the complete Q/K/V
source grid. All FP32 input words, distinct Q/K floating orders, in-place Q,
conditional K/V work, indirect caches and both exterior frames are retained. -/
specification kernel_equivalent (rest : Algebra NativeSource.Payload) (p : Native.Instructions)
    (c : Config) (valid : c.Valid) :
    original rest p c ≡[[]] candidate rest p c := by
  apply ProgramIO.Implementation.of_equivalentOn
  exact IOPair.equivalent rest p c valid

#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.FPEquiv

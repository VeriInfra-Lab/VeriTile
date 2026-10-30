import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.UncachedInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.CachedInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.RealRaw

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section

theorem uncached_real (rest : Algebra ℝ) (g : Geometry) (p q : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    (uncachedIO rest realOps g p q).Implements (uTarget realOps g) :=
  uncached_implements rest realOps g p q heads

theorem cached_real (rest : Algebra ℝ) (g : Geometry) (p q : Nat) (token : q<g.tokens) :
    (cachedIO rest realOps g p q).ImplementsOn (MetadataDomain g) (cTarget realOps g) :=
  cached_implements rest realOps g p q token

#axiomsClean uncached_real
#axiomsClean cached_real
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO

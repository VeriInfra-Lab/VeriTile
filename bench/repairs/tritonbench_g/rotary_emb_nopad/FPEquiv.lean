import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.ProgramPair
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Native

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv
open VeriTile Triton FP.Structural KernelInterface
open scoped VeriTile.Spec
noncomputable section

/-- Actual complete uncached source pair, including both Q stores, shared
coefficient reads and the conditional GQA-leader K branch. All original native
payloads are admitted; only exact integer Q-address CSE is used. -/
specification uncached_source_candidate_equivalence (rest : Algebra Native.Node)
    (op : Native.Instructions) (g : Geometry) (p q : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    PairIO.uncachedOriginal rest (Native.ops op Bool.false) g p q ≡[[]]
    PairIO.uncachedShared rest (Native.ops op Bool.false) g p q := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.uncached _ _ g p q)
    (PairIO.uncached_equivalent _ _ g p q heads)

/-- Actual complete cached source pair, including early head return, signed
metadata, all Q/cache outputs and K/exterior frames on the unchanged metadata
domain. Numeric operations and cache-before-Q store order are identical. -/
specification cached_source_candidate_equivalence (rest : Algebra Native.Node)
    (op : Native.Instructions) (g : Geometry) (p q : Nat) (token : q<g.tokens) :
    PairIO.cachedOriginal rest (Native.ops op Bool.true) g p q ≡[[]]
    PairIO.cachedShared rest (Native.ops op Bool.true) g p q := by
  exact ProgramIO.Implementation.of_equivalentOn (PairIO.cached _ _ g p q)
    (PairIO.cached_equivalent _ _ g p q token)

#axiomsClean uncached_source_candidate_equivalence
#axiomsClean cached_source_candidate_equivalence
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv

import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPTotalBackend
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeSM90
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Numerical

namespace VeriTile.Bench.Repairs.ContextAttnLlama.NativeInstructions
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
abbrev Expr := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.Expr
abbrev Word := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.Word
abbrev ScalarOps := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.ScalarOps
abbrev ScalarLaws := VeriTile.Bench.Repairs.ContextAttnFwd.Numerical.ScalarLaws
abbrev evaluate := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.evaluate
abbrev plan := VeriTile.Bench.Repairs.ContextAttnFwd.NativeSM90.plan
abbrev owner := VeriTile.Bench.Repairs.ContextAttnFwd.NativeSM90.storedOwner

/-- The page lookup changes discrete operands, not the shared SM90 floating
instruction plan. Both BM branches, every width, replica owner, ordered tree,
FMA and WGMMA scale-d selection are retained by this same expression algebra. -/
def backend (c : Host.Call) : Algebra Expr :=
  VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.backend (plan c.layout.BM c.layout.D)

def support (c : Host.Call) : FPSuccess.DotSupport c.layout (backend c) where
  qk := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.dots.dot (some .fp32)
    (batch := []) (m := c.layout.BM) (k := c.layout.D) (n := c.layout.BN)
  pv := VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.dots.dotAcc (some .fp32) .real .real .real {}
    (batch := []) (m := c.layout.BM) (k := c.layout.BN) (n := c.layout.D)
  qk_bound := fun _ _ => rfl
  pv_bound := fun _ _ _ => rfl

theorem evaluated_if_local (actual reference : ScalarOps) (laws : ScalarLaws actual reference)
    (c : Host.Call) (expression : Expr) (consumer : Nat) :
    evaluate actual (plan c.layout.BM c.layout.D) expression consumer=
      evaluate reference (plan c.layout.BM c.layout.D) expression consumer :=
  VeriTile.Bench.Repairs.ContextAttnFwd.Numerical.evaluated_equiv actual reference laws _ expression consumer

#axiomsClean evaluated_if_local
end
end VeriTile.Bench.Repairs.ContextAttnLlama.NativeInstructions

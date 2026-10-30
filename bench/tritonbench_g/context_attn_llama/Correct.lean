import bench.tritonbench_g.context_attn_llama.Proofs.LogicalIO
import VeriTile.Meta.Specification

/-! Complete ordinary-grid IO of the unchanged FP32 source under Real
interpretation. RealIO.lean exposes all named arrays, capacity/stride geometry,
metadata, masks and the independent gathered finite-sentinel formula.
The original canonical flat-allocation transport remains in Proofs. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile Triton
open scoped VeriTile.Triton.LogicalDomainIO
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ContextAttnLlama.Formula
def score (p : Layout) (l : Launch) (a : RealInterface.Logical.Arrays) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt (batch p l) then
    (∑ e : Fin p.D, a.query l i e.val * a.key l j e.val) * scale
  else -100000000

def denominator (p : Layout) (l : Launch) (a : RealInterface.Logical.Arrays) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (a.blocks l*p.BN), if j.val < c*p.BN then pow2 (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : RealInterface.Logical.Arrays) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (a.blocks l*p.BN),
    if j.val < c*p.BN then pow2 (score p l a scale i j.val)*a.value l j.val d else 0

def expected (p : Layout) (l : Launch) (a : RealInterface.Logical.Arrays) (scale : ℝ) (i d : Nat) : ℝ :=
  numerator p l a scale i d (a.blocks l) / denominator p l a scale i (a.blocks l)
end Mathematics

def mathematical (c : Host.Call) (a : RealInterface.Logical.Arrays) (k : RealInterface.Lane c) : ℝ :=
  Mathematics.expected c.layout (c.launchAt k.1) a
    (Host.scale c.layout.D) k.2.1.val k.2.2.1.val


specification context_attn_llama_correctness (c : Host.Call) (capacity : RegionBounds)
    (geometry : RealInterface.Geometry c capacity) :
    Spec.Real (RealInterface.Logical.io c capacity ⊨{RealInterface.Logical.domain c capacity}
      (fun values => mathematical c values.arrays)) :=
  RealInterface.Logical.correct_arrays c capacity geometry (mathematical c) (by intros; rfl)

#stmtSurfaceSubset context_attn_llama_correctness ⊆
  [Spec.Real,Host.Call,RegionBounds,RealInterface.Geometry,RealInterface.Logical.io,
   LogicalDomainIO.ImplementsOn,RealInterface.Logical.domain,mathematical,
   RealInterface.Inputs,RealInterface.laneCount,RealInterface.Logical.Inputs,RealInterface.Logical.Inputs.arrays,RealInterface.Logical.Arrays,RealInterface.Lane,
   AlgorithmProgramIO.ImplementsOn,RealInterface.programIO]
#axiomsClean context_attn_llama_correctness
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly

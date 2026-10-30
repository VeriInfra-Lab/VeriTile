import bench.tritonbench_g.context_attn_bloom.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalDomainIO
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Math
def score (p : Layout) (l : Launch) (a : RealKernelIO.Logical.Arrays) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt l.batch then
    scale * ∑ e : Fin p.D, a.query l i e.val * a.key l j e.val
  else -100000000

def denominator (p : Layout) (l : Launch) (a : RealKernelIO.Logical.Arrays) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (a.blocks l*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : RealKernelIO.Logical.Arrays) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (a.blocks l*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*a.value l j.val d else 0

def expected (p : Layout) (l : Launch) (a : RealKernelIO.Logical.Arrays) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 =>
    expected p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) /
        denominator p l a scale i (c+1)
end Mathematics


/-- Logical named arrays and grid/lane coordinates retain every original
finite storage cell, signed metadata role and the complete serial schedule. -/
def realIO := RealKernelIO.Logical.io

def mathematical (g : Raw.Geometry) (a : RealKernelIO.Logical.Arrays)
    (c : KernelIO.Coordinate g) : ℝ :=
  Mathematics.expected (Host.layout g.arguments) (Host.launchOf g.arguments c.1)
    a (Host.scale g.arguments) c.2.1.val c.2.2.1.val
    (a.blocks (Host.launchOf g.arguments c.1))

def realDomain := RealKernelIO.Logical.domain

/-- Independent normalized-block values on the entire original input domain.
The shared IO judgment includes actual source access safety, successful full
grid execution, every active output and the complete exterior cell frame. -/
specification correct_real (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) :
    Spec.Real (realIO g r extent ⊨{realDomain g r extent} (fun values => mathematical g values.arrays)) :=
  RealKernelIO.Logical.correct_recurrence g r extent Mathematics.expected
    (by intros; rfl) (by intros; rfl)

#stmtSurfaceSubset correct_real ⊆
  [Spec.Real,realIO,realDomain,mathematical,LogicalDomainIO.ImplementsOn,Raw.Geometry,Regions,RegionBounds,
   RealKernelIO.Values,KernelIO.cells,RealKernelIO.Logical.Inputs,RealKernelIO.Logical.Inputs.arrays,RealKernelIO.Logical.Arrays,KernelIO.Coordinate,
   AlgorithmProgramIO.ImplementsOn,RealKernelIO.io]
#specNonCircular mathematical avoiding [exec,execR,AlgorithmStage.Runs,BlockState.mem]
#axiomsClean correct_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel

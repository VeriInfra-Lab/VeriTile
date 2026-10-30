import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Proofs.LogicalIO

/-! Public finite IO specifications for the two distinct selected FP32 kernels.
RealIO.lean declares source projections, full ordinary-grid plan lists, input
roles, logical output windows and padded scratch. Python staging stays an input
contract; these theorems prove kernel execution and independent mathematics. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealInterface
open scoped VeriTile.Triton.LogicalDomainIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open Host (InnerInputs endpoint)
def innerWeight (a : InnerInputs) (T BT : Nat) (gateK : Bool) (c k v : Nat) : ℝ :=
  Real.exp (a.cumulativeGate (endpoint T BT c) (if gateK then k else v))

def innerContribution (a : InnerInputs) (T BT : Nat) (gateK : Bool) (c k v : Nat) : ℝ :=
  ∑ lane : Fin BT,
    let t := c*BT+lane.val
    if t < T then
      let f := if gateK then k else v
      let w := Real.exp (a.cumulativeGate (endpoint T BT c) f - a.cumulativeGate t f)
      if gateK then (a.key t k*w)*a.value t v else a.key t k*(a.value t v*w)
    else 0

def innerState (a : InnerInputs) (T BT : Nat) (gateK hasSeed : Bool) (c k v : Nat) : ℝ :=
  (if hasSeed then a.seed k v else 0) *
      (∏ u ∈ Finset.range c, innerWeight a T BT gateK u k v) +
    ∑ t ∈ Finset.range c, innerContribution a T BT gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) c, innerWeight a T BT gateK u k v

def innerResult (p : RealHMath.Layout) (heads T K V : Nat) (gateK hasSeed : Bool)
    (x : Nat → Host.InnerInputs) : Logical.InnerOutputs p heads K V where
  history := fun head chunk key value =>
    innerState (x head.val) T p.BT gateK hasSeed chunk.val key.val value.val
  final := fun head key value =>
    innerState (x head.val) T p.BT gateK hasSeed p.NT key.val value.val

end Mathematics


/-- Complete recurrent kernel: every logical history cell and optional final
state equals the independent gated contraction. Seed/final channels follow the
source switches; output padding has an explicit writable scratch footprint. -/
specification inner_correct (p : RealHMath.Layout) (heads T K V : Nat)
    (r : RealHProgram.Regions) (gateK hasSeed storeFinal : Bool)
    (x : Nat → Host.InnerInputs) (hp : StagedLayout.Canonical p)
    (hk : K ≤ p.K) (hv : V ≤ p.V)
    (hs : DirectRealSequence.InputSeparate r storeFinal)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    Spec.Real (Logical.innerIO p heads K V r gateK hasSeed storeFinal ⊨{Logical.innerDomain p heads T K V gateK hasSeed x}
      fun _ => Mathematics.innerResult p heads T K V gateK hasSeed x) :=
  Logical.inner_correct p heads T K V r gateK hasSeed storeFinal x hp hk hv hs hfsep

/-- Complete prefix kernel for every retained BS choice, including exhausted
autotune programs. The independent target is the within-chunk prefix sum. -/
specification prefix_correct (p : RealCumProgram.Layout) (heads T S : Nat)
    (input output : RegionName) (x : Nat → Nat → Nat → ℝ)
    (hp : CumulativeHost.Canonical p) (hbs : p.BS = 16 ∨ p.BS = 32 ∨ p.BS = 64)
    (hT : T ≤ p.T) (hS : S ≤ p.S) (hsep : input ≠ output) :
    Spec.Real (Logical.prefixIO p heads T S input output ⊨{Logical.prefixDomain p heads T S x}
      fun _ head time feature => ∑ u ∈ Finset.range (time.val+1),
        if time.val / p.BT * p.BT ≤ u then x head.val u feature.val else 0) := by
  apply (Logical.prefix_full_judgment_iff p heads T S input output x _).mpr
  exact prefix_implements p heads T S input output x hp hbs hT hS hsep

#axiomsClean inner_correct
#axiomsClean prefix_correct
#stmtSurfaceSubset inner_correct ⊆ [Spec.Real,LogicalDomainIO.ImplementsOn,AlgorithmProgramIO.ImplementsOn,AlgorithmProgramIO.Ctx,Logical.innerIO,Logical.innerDomain,
  Logical.InnerInputs,Logical.InnerOutputs,innerInputArity,innerOutputArity,innerIO,innerDomain,Mathematics.innerResult,RealHMath.Layout,RealHProgram.Regions,RegionName,
  Host.InnerInputs,StagedLayout.Canonical,DirectRealSequence.InputSeparate,
  RealHMath.Layout.K,RealHMath.Layout.V,RealHProgram.Regions.history,RealHProgram.Regions.final]
#stmtSurfaceSubset prefix_correct ⊆ [Spec.Real,LogicalDomainIO.ImplementsOn,AlgorithmProgramIO.ImplementsOn,AlgorithmProgramIO.Ctx,Logical.prefixIO,Logical.prefixDomain,
  prefixCount,prefixIO,prefixDomain,RealCumProgram.Layout,RegionName,
  CumulativeHost.Canonical,VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram.Layout.BT,VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram.Layout.BS,
  VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram.Layout.T,
  VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram.Layout.S]
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly

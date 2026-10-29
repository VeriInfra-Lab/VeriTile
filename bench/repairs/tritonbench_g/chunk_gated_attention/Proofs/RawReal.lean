import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.DirectRealWhole
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.DirectCumulativeWhole

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawReal
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section

/-- Complete FP32 recurrent source under the ideal Real interpretation.
Raw input reads describe the existing zero padding and repeated final gate;
the result includes independent history/final values and every exterior cell. -/
specification inner_correct (p : RealHMath.Layout) (heads T K V : Nat)
    (r : RealHProgram.Regions) (gateK hasSeed storeFinal : Bool)
    (x : Nat → Host.InnerInputs) (hp : StagedLayout.Canonical p)
    (hk : K ≤ p.K) (hv : V ≤ p.V)
    (hs : DirectRealSequence.InputSeparate r storeFinal)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (before : BlockState)
    (hkey : ∀ off, before.readMem r.key off = (Materialize.inputs p T K V gateK x).key off)
    (hvalue : ∀ off, before.readMem r.value off = (Materialize.inputs p T K V gateK x).value off)
    (hgate : ∀ off, before.readMem r.gate off = (Materialize.inputs p T K V gateK x).gate off)
    (hseed : ∀ off, before.readMem r.seed off = (Materialize.inputs p T K V gateK x).seed off) :
    Spec.Real (∃ after,
      DirectRealSequence.Runs p r .fp32 .fp32 .fp32 .fp32 .fp32
        gateK hasSeed storeFinal Bool.false
        (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV)))
        {before with undef := fun _ _ => 0} after ∧
      DirectRealWhole.Outputs p heads T K V r x gateK hasSeed storeFinal after ∧
      BlockState.WriteWithin
        (DirectRealSequence.allWrites p r
          (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV))) storeFinal before) before after) :=
  DirectRealWhole.executes p heads T K V r .fp32 .fp32 .fp32 .fp32 .fp32
    gateK hasSeed storeFinal Bool.false x hp hk hv hs hfsep before hkey hvalue hgate hseed

/-- Complete FP32 prefix source for every retained BS choice, independent
within-chunk sums at valid logical coordinates, and the complete memory frame. -/
specification prefix_correct (p : RealCumProgram.Layout) (heads T S : Nat)
    (input output : RegionName) (x : Nat → Nat → Nat → ℝ)
    (hp : CumulativeHost.Canonical p) (hbs : p.BS = 16 ∨ p.BS = 32 ∨ p.BS = 64)
    (hT : T ≤ p.T) (hS : S ≤ p.S) (hsep : input ≠ output)
    (before : BlockState)
    (hx : ∀ off, before.readMem input off = CumulativeHost.materialized p T S x off) :
    Spec.Real (∃ after,
      DirectCumulativeSequence.Runs p heads input output Bool.false .fp32 .fp32
        (DirectCumulativeWhole.plans p heads) {before with undef := fun _ _ => 0} after ∧
      (∀ h t f, h < heads → t < T → f < S →
        after.readMem output (h*p.headStride+t*p.timeStride+f*p.channelStride) =
          Host.chunkPrefix p.BT (x h) t f) ∧
      BlockState.WriteWithin
        (DirectCumulativeSequence.allWrites p heads (DirectCumulativeWhole.plans p heads)
          output before) before after) :=
  DirectCumulativeWhole.executes p heads T S input output Bool.false .fp32 .fp32
    x hp hbs hT hS hsep before hx

#specNonCircular Host.innerState avoiding [DirectRealSequence.Runs,VeriTile.Triton.exec]
#specNonCircular Host.chunkPrefix avoiding [DirectCumulativeSequence.Runs,VeriTile.Triton.exec]
#axiomsClean inner_correct
#axiomsClean prefix_correct
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawReal

import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.InnerIO
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.PrefixIO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface.Logical
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

theorem inner_full_judgment_iff (p : RealHMath.Layout) (heads T K V : Nat)
    (r : RealHProgram.Regions) (gateK hasSeed storeFinal : Bool) (x : Nat → Host.InnerInputs)
    (f : InnerInputs p heads gateK → InnerOutputs p heads K V) :
    (innerIO p heads K V r gateK hasSeed storeFinal ⊨{innerDomain p heads T K V gateK hasSeed x} f) ↔
      (RealInterface.innerIO p heads K V r gateK hasSeed storeFinal).ImplementsOn
        (RealInterface.innerDomain p heads T K V gateK hasSeed x)
        (fun raw => (innerOutputs p heads K V).symm (f (innerInputs p heads gateK raw))) := by
  change (RealInterface.innerIO p heads K V r gateK hasSeed storeFinal).ImplementsOn
    (fun raw => RealInterface.innerDomain p heads T K V gateK hasSeed x
      ((innerInputs p heads gateK).symm (innerInputs p heads gateK raw))) _ ↔ _
  simp only [Equiv.symm_apply_apply]
  rfl

theorem prefix_full_judgment_iff (p : RealCumProgram.Layout) (heads T S : Nat)
    (input output : RegionName) (x : Nat → Nat → Nat → ℝ)
    (f : (Fin heads → Fin p.T → Fin p.S → ℝ) → Fin heads → Fin T → Fin S → ℝ) :
    (prefixIO p heads T S input output ⊨{prefixDomain p heads T S x} f) ↔
      (RealInterface.prefixIO p heads T S input output).ImplementsOn
        (RealInterface.prefixDomain p heads T S x)
        (fun raw => (prefixOutputs heads T S).symm (f (prefixInputs p heads raw))) := by
  change (RealInterface.prefixIO p heads T S input output).ImplementsOn
    (fun raw => RealInterface.prefixDomain p heads T S x
      ((prefixInputs p heads).symm (prefixInputs p heads raw))) _ ↔ _
  simp only [Equiv.symm_apply_apply]
  rfl

/-- Canonical scalar history/final target, with all flat output routing
eliminated by the already proved coordinate equivalence. -/
theorem inner_correct (p : RealHMath.Layout) (heads T K V : Nat)
    (r : RealHProgram.Regions) (gateK hasSeed storeFinal : Bool)
    (x : Nat → Host.InnerInputs) (hp : StagedLayout.Canonical p)
    (hk : K ≤ p.K) (hv : V ≤ p.V)
    (hs : DirectRealSequence.InputSeparate r storeFinal)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    innerIO p heads K V r gateK hasSeed storeFinal ⊨{innerDomain p heads T K V gateK hasSeed x}
      (fun _ =>
        { history := fun head chunk key value =>
          Host.innerState (x head.val) T p.BT gateK hasSeed chunk.val key.val value.val
          final := fun head key value =>
            Host.innerState (x head.val) T p.BT gateK hasSeed p.NT key.val value.val }) := by
  apply (inner_full_judgment_iff p heads T K V r gateK hasSeed storeFinal x _).mpr
  have target_eq : (fun raw => (innerOutputs p heads K V).symm
      { history := fun head chunk key value =>
          Host.innerState (x head.val) T p.BT gateK hasSeed chunk.val key.val value.val
        final := fun head key value =>
          Host.innerState (x head.val) T p.BT gateK hasSeed p.NT key.val value.val }) =
      innerFormula p heads T K V gateK hasSeed x := by
    funext raw channel j
    fin_cases channel <;> rfl
  exact (congrArg ((RealInterface.innerIO p heads K V r gateK hasSeed storeFinal).ImplementsOn
    (RealInterface.innerDomain p heads T K V gateK hasSeed x)) target_eq).mpr
      (inner_implements p heads T K V r gateK hasSeed storeFinal x hp hk hv hs hfsep)


#axiomsClean inner_full_judgment_iff
#axiomsClean prefix_full_judgment_iff
#axiomsClean inner_correct
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface.Logical

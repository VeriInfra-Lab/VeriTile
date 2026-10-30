import bench.tritonbench_g.max_reduction.IO
import bench.tritonbench_g.max_reduction.Proofs.RealIO
import bench.tritonbench_g.max_reduction.Proofs.Mathematical_StageIO

namespace VeriTile.Bench.Repairs.MaxReduction.LogicalIOProof
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

def blockMaximum (N B pid : Nat) (xs : Fin B → ℝ) : ℝ :=
  WithBot.unbotD 0 ((Finset.univ : Finset (Fin B)).sup
    (fun i => if pid*B+i.val<N then (some (xs i) : WithBot ℝ) else ⊥))

theorem stage1_headline_iff (inp mid : RegionName) (N B : Nat) :
    (PublicIO.logicalStage1IO inp mid N B ⊨ fun ⟨pid,xs⟩ _ => blockMaximum N B pid xs) ↔
    (PublicIO.stage1Real inp mid N B).Implements (fun pid xs _ =>
      VeriTile.Bench.TritonBenchG.MaxReduction.blockMaximum N B pid xs) := Iff.rfl

theorem stage2_headline_iff (mid out : RegionName) (N B : Nat) :
    (PublicIO.logicalStage2IO mid out N B ⊨ fun ⟨_,xs⟩ _ => blockMaximum N B 0 xs) ↔
    (PublicIO.stage2Real mid out N B).Implements (fun _ xs _ =>
      VeriTile.Bench.TritonBenchG.MaxReduction.blockMaximum N B 0 xs) := Iff.rfl

theorem stage1_implements (inp mid : RegionName) (N B : Nat) (hB : 0<B) :
    (PublicIO.logicalStage1IO inp mid N B ⊨ fun ⟨pid,xs⟩ _ => blockMaximum N B pid xs) :=
  (stage1_headline_iff inp mid N B).2
    (VeriTile.Bench.TritonBenchG.MaxReduction.StageIOProof.stage1_real_correct .fp32 Bool.false inp mid N B hB)

theorem stage2_implements (mid out : RegionName) (N B : Nat) (hB : 0<B) :
    (PublicIO.logicalStage2IO mid out N B ⊨ fun ⟨_,xs⟩ _ => blockMaximum N B 0 xs) :=
  (stage2_headline_iff mid out N B).2
    (VeriTile.Bench.TritonBenchG.MaxReduction.StageIOProof.stage2_real_correct .fp32 Bool.false mid out N B hB)

theorem dimension_headline_iff (inp ov oi : RegionName) (M N K BM p k : Nat)
    (plan : ReductionPlan (nextPower N))
    (relation : (Fin BM → Fin (nextPower N) → ℝ) → ((Fin BM → ℝ) × (Fin BM → Int)) → Prop)
    (same : ∀ xs ys, relation (PublicIO.dimensionInputs inp BM (nextPower N) xs)
      (PublicIO.dimensionOutputs ov oi BM ys) = DimensionMath.result inp ov oi M N BM (nextPower N) p xs ys) :
    (PublicIO.logicalDimensionIO (KernelMath.engine Compute.real plan.tree) RealIO.observe
      inp ov oi M N K BM (nextPower N) p k).Satisfies relation ↔
    (PublicIO.dimensionIO (KernelMath.engine Compute.real plan.tree) (fun r : ℝ => (r : WithBot ℝ))
      RealIO.observe inp ov oi M N K BM (nextPower N) p k).Satisfies
      (DimensionMath.result inp ov oi M N BM (nextPower N) p) := by
  have target : (fun xs ys => relation (PublicIO.dimensionInputs inp BM (nextPower N) xs)
      (PublicIO.dimensionOutputs ov oi BM ys)) = DimensionMath.result inp ov oi M N BM (nextPower N) p := by
    funext xs ys
    exact same xs ys
  change (PublicIO.dimensionIO (KernelMath.engine Compute.real plan.tree)
    (fun r : ℝ => (r : WithBot ℝ)) RealIO.observe inp ov oi M N K BM (nextPower N) p k).Satisfies
      (fun xs ys => relation (PublicIO.dimensionInputs inp BM (nextPower N) xs)
        (PublicIO.dimensionOutputs ov oi BM ys)) ↔ _
  exact (congrArg
    (fun r => (PublicIO.dimensionIO (KernelMath.engine Compute.real plan.tree)
      (fun x : ℝ => (x : WithBot ℝ)) RealIO.observe inp ov oi M N K BM (nextPower N) p k).Satisfies r)
    target).to_iff

theorem dimension_implements (inp ov oi : RegionName) (M N K BM p k : Nat)
    (hRegions : ov≠oi) (hK : 0<K) (plan : ReductionPlan (nextPower N))
    (relation : (Fin BM → Fin (nextPower N) → ℝ) → ((Fin BM → ℝ) × (Fin BM → Int)) → Prop)
    (same : ∀ xs ys, relation (PublicIO.dimensionInputs inp BM (nextPower N) xs)
      (PublicIO.dimensionOutputs ov oi BM ys) = DimensionMath.result inp ov oi M N BM (nextPower N) p xs ys) :
    (PublicIO.logicalDimensionIO (KernelMath.engine Compute.real plan.tree) RealIO.observe
      inp ov oi M N K BM (nextPower N) p k).Satisfies relation :=
  (dimension_headline_iff inp ov oi M N K BM p k plan relation same).2
    (RealIO.satisfies inp ov oi M N K BM p k hK hRegions plan)

#axiomsClean stage1_headline_iff
#axiomsClean stage2_headline_iff
#axiomsClean dimension_headline_iff
#axiomsClean dimension_implements
end
end VeriTile.Bench.Repairs.MaxReduction.LogicalIOProof

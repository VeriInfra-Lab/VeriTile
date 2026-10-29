import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.PairSource
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.PairNumeric
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelGrid

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSemantics
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math PairSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- The strict software value model supplies only the two transformed ports.
All preparation ports, bias order, comparisons and runtime clamp words retain
the same explicit model data on both programs. -/
def concreteModel (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) :
    NativeModel (FP.Value .fp32) :=
  { M with mul := FP.mul cfg, fma := FP.fma cfg }

theorem reverse_prepared {α : Type} [Inhabited α] (M : NativeModel α)
    (c : Geometry) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : Term α) (s : State (Term α)) :
    logicalPrepared c (reversePrimitives c (primitives M c plan)) hasBias useSoftplus lower upper s =
      logicalPrepared c (primitives M c plan) hasBias useSoftplus lower upper s := by
  rfl

theorem reverse_rate {α : Type} [Inhabited α] (M : NativeModel α)
    (c : Geometry) (plan : PrefixPlan c.BH c.BC) (s : State (Term α)) :
    logicalRate c (reversePrimitives c (primitives M c plan)) s =
      logicalRate c (primitives M c plan) s := by
  rfl

/-- Every leaf retains a once-rounded multiply; every fused leaf retains a
once-rounded product-plus-add with its original addend and prefix topology. -/
theorem reversed_scan (cfg : FP.Config) (M : NativeModel (FP.Value .fp32))
    (prepared : Fin n → Term (FP.Value .fp32)) (rate : Term (FP.Value .fp32))
    (tree : SumTree n) :
    interpret (concreteModel cfg M)
      (tree.eval (Term.binary (some .fp32) .real .add) contract
        (fun i => .binary (some .fp32) .real .mul rate (prepared i))) =
      referenceTree (concreteModel cfg M)
        (fun i => interpret (concreteModel cfg M) (prepared i))
        (interpret (concreteModel cfg M) rate) tree := by
  induction tree with
  | leaf i => exact FP.ScalarOpsLaws.mul_comm cfg _ _
  | fused i r ih =>
    simp only [SumTree.eval, contract, interpret, referenceTree]
    rw [ih]
    exact FP.ScalarOpsLaws.fma_mul_comm cfg _ _ _
  | add l r il ir =>
    simp only [SumTree.eval, referenceTree]
    cases l <;> cases r <;> simp only [SumTree.eval, contract, interpret] at il ir ⊢ <;> rw [il, ir]

theorem reverse_cumulative (cfg : FP.Config) (M : NativeModel (FP.Value .fp32))
    (c : Geometry) (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : Term (FP.Value .fp32)) (s : State (Term (FP.Value .fp32)))
    (b h k : Nat) (j : Fin c.BC) :
    interpret (concreteModel cfg M)
      (logicalCumulative c (reversePrimitives c (primitives (concreteModel cfg M) c plan))
        hasBias useSoftplus lower upper s b h k j) =
      interpret (concreteModel cfg M)
      (logicalCumulative c (primitives (concreteModel cfg M) c plan)
        hasBias useSoftplus lower upper s b h k j) := by
  simp only [logicalCumulative, reverse_prepared, reverse_rate]
  change interpret (concreteModel cfg M)
      ((plan.tree _ j).eval (Term.binary (some .fp32) .real .add) contract
        (fun i => .binary (some .fp32) .real .mul
          (logicalRate c (primitives (concreteModel cfg M) c plan) s h)
          (logicalPrepared c (primitives (concreteModel cfg M) c plan)
            hasBias useSoftplus lower upper s b h k i.val))) = _
  rw [reversed_scan]
  exact (source_scan (concreteModel cfg M) _ _ _).symm

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairSemantics

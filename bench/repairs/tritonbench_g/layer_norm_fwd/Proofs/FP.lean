import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.SourceBinding
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WholeReduction
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LayerNormFwd.FPProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

/-- Complete typed FP execution is unchanged after the independently checked
integer-width erasure. Includes failure and every memory/register field. -/
theorem whole_state (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec A (repaired f r stride M N eps p block hBlock) u =
      FP.Structural.exec A (original f r stride M N eps p block hBlock) u := rfl

/-- Historical hypothetical exact-Real scalar interface. The exact add/mul/FMA
equations below are generally false for rounded FP32; they are not deferred
native-conformance obligations. FMA remains distinct from multiply then add. -/
structure ScalarOps (α : Type) where
  decode : α → ℝ
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α

def SumPorts (a : ScalarOps α) : Whole.SumTree block → (Fin block → α) → Prop
  | .leaf _, _ => True
  | .add l r, x => SumPorts a l x ∧ SumPorts a r x ∧
      a.decode (a.add (l.eval a.add x) (r.eval a.add x)) =
        a.decode (l.eval a.add x) + a.decode (r.eval a.add x)

def SquarePorts (a : ScalarOps α) : Whole.SquareTree block → (Fin block → α) → Prop
  | .square j, x => a.decode (a.mul (x j) (x j)) = a.decode (x j) * a.decode (x j)
  | .add l r, x => SquarePorts a l x ∧ SquarePorts a r x ∧
      a.decode (a.add (l.eval a.mul a.add a.fma x) (r.eval a.mul a.add a.fma x)) =
        a.decode (l.eval a.mul a.add a.fma x) + a.decode (r.eval a.mul a.add a.fma x)
  | .fmaSquare j r, x => SquarePorts a r x ∧
      a.decode (a.fma (x j) (x j) (r.eval a.mul a.add a.fma x)) =
        a.decode (x j) * a.decode (x j) + a.decode (r.eval a.mul a.add a.fma x)

theorem sum_ports_sound (a : ScalarOps α) (t : Whole.SumTree block) (x : Fin block → α)
    (h : SumPorts a t x) :
    a.decode (t.eval a.add x) = t.eval (·+·) (fun j => a.decode (x j)) := by
  induction t with
  | leaf j => rfl
  | add l r hl hr => exact h.2.2.trans (congrArg₂ (·+·) (hl h.1) (hr h.2.1))

theorem square_ports_sound (a : ScalarOps α) (t : Whole.SquareTree block) (x : Fin block → α)
    (h : SquarePorts a t x) :
    a.decode (t.eval a.mul a.add a.fma x) =
      t.eval (·*·) (·+·) (fun a b c => a*b+c) (fun j => a.decode (x j)) := by
  induction t with
  | square j => exact h
  | add l r hl hr => exact h.2.2.trans (congrArg₂ (·+·) (hl h.1) (hr h.2.1))
  | fmaSquare j r hr => exact h.2.trans (congrArg (fun z => a.decode (x j)*a.decode (x j)+z) (hr h.1))

/-- A derived independent sum theorem; only local primitive ports are inputs. -/
theorem mean_reduction_ports (a : ScalarOps α) (red : Whole.Reductions block)
    (x : Fin block → α) (h : SumPorts a red.mean x) :
    a.decode (red.mean.eval a.add x) = ∑ j : Fin block, a.decode (x j) := by
  rw [sum_ports_sound a red.mean x h, Whole.mean_reduction_real]

theorem variance_reduction_ports (a : ScalarOps α) (red : Whole.Reductions block)
    (x : Fin block → α) (h : SquarePorts a red.variance x) :
    a.decode (red.variance.eval a.mul a.add a.fma x) = ∑ j : Fin block, (a.decode (x j))^2 := by
  rw [square_ports_sound a red.variance x h, Whole.variance_reduction_real]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.FPProof

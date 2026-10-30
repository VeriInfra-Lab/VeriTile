import bench.repairs.tritonbench_g.context_attn_bloom.IO
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PublicReal

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Exact equivalence with the complete old judgment, for every logical
formula, including the unchanged original domain. -/
theorem full_judgment_iff (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (f : Inputs g r extent → KernelIO.Coordinate g → ℝ) :
    (io g r extent ⊨{domain g r extent} f) ↔
      (RealKernelIO.io g r extent).ImplementsOn (PublicDomain.real g r extent)
        (fun raw _ j => f (inputs g r extent raw) ((Fintype.equivFin (KernelIO.Coordinate g)).symm j)) := by
  change (RealKernelIO.io g r extent).ImplementsOn
    (fun raw => PublicDomain.real g r extent ((inputs g r extent).symm (inputs g r extent raw)))
    (fun raw _ j => f (inputs g r extent raw) ((Fintype.equivFin (KernelIO.Coordinate g)).symm j)) ↔ _
  simp only [Equiv.symm_apply_apply]

theorem correct_arrays (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (F : Arrays → KernelIO.Coordinate g → ℝ)
    (formula_eq : ∀ (a : Math.Inputs) (c : KernelIO.Coordinate g),
      F (mathematicalArrays (Host.layout g.arguments) a) c =
        Formula.expected (Host.layout g.arguments) (Host.launchOf g.arguments c.1) a
          (Host.scale g.arguments) c.2.1.val c.2.2.1.val
          (Math.count (Host.layout g.arguments) (Host.launchOf g.arguments c.1) a)) :
    io g r extent ⊨{domain g r extent} (fun xs => F xs.arrays) := by
  apply (full_judgment_iff g r extent _).mpr
  have target_eq : (fun raw (_ : Fin 1) j => F (inputs g r extent raw).arrays
      ((Fintype.equivFin (KernelIO.Coordinate g)).symm j)) = RealKernelIO.formula g r extent := by
    funext raw channel j
    exact formula_eq _ _
  exact (congrArg ((RealKernelIO.io g r extent).ImplementsOn
    (PublicDomain.real g r extent)) target_eq).mpr (RealKernelIO.public_implements g r extent)

/-- Connect an independently displayed recurrence on mathematical arrays.
The recursion argument is proved here, outside the reader specification. -/
theorem correct_recurrence (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (E : Layout → Launch → Arrays → ℝ → Nat → Nat → Nat → ℝ)
    (initial_equation : ∀ p l (a : Math.Inputs) scale i d,
      E p l (mathematicalArrays p a) scale i d 0 = 0)
    (step_equation : ∀ p l (a : Math.Inputs) scale i d n,
      E p l (mathematicalArrays p a) scale i d (n+1) =
        E p l (mathematicalArrays p a) scale i d n *
          (if ((n*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
            Formula.denominator p l a scale i n / Formula.denominator p l a scale i (n+1) else 1) +
          (Formula.numerator p l a scale i d (n+1)-Formula.numerator p l a scale i d n) /
            Formula.denominator p l a scale i (n+1)) :
    io g r extent ⊨{domain g r extent} (fun xs c =>
      E (Host.layout g.arguments) (Host.launchOf g.arguments c.1) xs.arrays
        (Host.scale g.arguments) c.2.1.val c.2.2.1.val
        (xs.arrays.blocks (Host.launchOf g.arguments c.1))) := by
  apply correct_arrays g r extent (fun a c =>
    E (Host.layout g.arguments) (Host.launchOf g.arguments c.1) a
      (Host.scale g.arguments) c.2.1.val c.2.2.1.val
      (a.blocks (Host.launchOf g.arguments c.1)))
  intro a c
  have recurrence_eq (p : Layout) (l : Launch) (a : Math.Inputs)
      (scale : ℝ) (i d n : Nat) :
      E p l (mathematicalArrays p a) scale i d n = Formula.expected p l a scale i d n := by
    induction n with
    | zero => exact initial_equation p l a scale i d
    | succ n ih => rw [step_equation, Formula.expected, ih]
  exact recurrence_eq _ _ _ _ _ _ _

#axiomsClean full_judgment_iff
#axiomsClean correct_arrays
#axiomsClean correct_recurrence
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO.Logical

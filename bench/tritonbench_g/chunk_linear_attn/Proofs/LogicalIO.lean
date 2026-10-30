import bench.tritonbench_g.chunk_linear_attn.IO
import bench.tritonbench_g.chunk_linear_attn.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.LogicalTensors
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Complete original judgment, in both directions, for the original
canonical formula after bijective input/output tensor coordinates. -/
theorem full_canonical_iff (c : KernelInterface.Config) (work : KernelSchedule.Work) :
    (io c work ⊨ fun values => outputs c ((Mathematical.real c) ((inputs c).symm values))) ↔
      (IO.real c work).Implements (Mathematical.real c) := by
  change (IO.real c work).Implements (fun raw => (outputs c).symm (outputs c ((Mathematical.real c) ((inputs c).symm (inputs c raw))))) ↔ (IO.real c work).Implements (Mathematical.real c)
  simp only [Equiv.symm_apply_apply]


/-- Universal mathematical side conditions are transported across the same
input bijection; no logical tensor or raw input assignment is omitted. -/
theorem inputs_forall_iff (c : KernelInterface.Config)
    (P : ChannelValues ℝ (IOLayout.inputs c.dims) → Prop) :
    (∀ values : Inputs c, P ((inputs c).symm values)) ↔ ∀ raw, P raw := by
  constructor
  · intro h raw
    have result := h (inputs c raw)
    rw [Equiv.symm_apply_apply] at result
    exact result
  · intro h values
    exact h ((inputs c).symm values)

/-- The complete conjunction from the original headline, including its
independent universal VJP clause, is equivalent in both directions. -/
theorem full_vjp_judgment_iff (c : KernelInterface.Config) (work : KernelSchedule.Work) :
    ((io c work ⊨ fun values => outputs c (Mathematical.real c ((inputs c).symm values))) ∧
      ∀ values : Inputs c, ∀ head,
        Real.IsVJP c.scale (realInputs c values head) (incoming c values head) (terminal c values head)
          (Mathematical.gradient c ((inputs c).symm values) head)) ↔
    ((IO.real c work).Implements (Mathematical.real c) ∧
      ∀ raw head, Real.IsVJP c.scale (Mathematical.inputs c raw head)
        (Mathematical.incoming c raw head) (Mathematical.terminal c raw head)
        (Mathematical.gradient c raw head)) :=
  and_congr (full_canonical_iff c work) (inputs_forall_iff c (fun raw => ∀ head,
    Real.IsVJP c.scale (Mathematical.inputs c raw head) (Mathematical.incoming c raw head)
      (Mathematical.terminal c raw head) (Mathematical.gradient c raw head)))

theorem correct (c : KernelInterface.Config) (work : KernelSchedule.Work) (valid : Host.Valid c.dims) (legal : KernelSchedule.Legal c.call work) :
    io c work ⊨ fun values => outputs c ((Mathematical.real c) ((inputs c).symm values)) :=
  (full_canonical_iff c work).mpr (PublicIO.real c valid work legal)

#axiomsClean full_canonical_iff
#axiomsClean correct
#axiomsClean full_vjp_judgment_iff
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.LogicalTensors

import bench.repairs.tritonbench_g.masked_select.Kernels
import bench.repairs.tritonbench_g.masked_select.Proofs.Kernel
import VeriTile.Meta.StatementAudit

/-! Equivalence of the actual autotune source configurations. No new kernel or
synthetic optimization is introduced. Different widths produce the same stable
FP32 payload sequence and entire memory, including the exterior frame. -/
namespace VeriTile.Bench.Repairs.MaskedSelect.ConfigurationEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect
open TypedCopy Whole
open scoped VeriTile.Spec
noncomputable section

structure Program where
  parameters : Parameters
  config : Config

def Program.source (p : Program) : ComputeKernel := kernel {p.parameters with width := p.config.width}

def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (A : Type) [Inhabited A] (rest : Algebra A) (input : Nat → A) (mask : Nat → Bool)
    (before : State A), KernelInputs lhs.parameters before input mask → KernelSeparate lhs.parameters →
    ∃ left right, Kernel.run rest lhs.parameters lhs.config before = some left ∧
      Kernel.run rest rhs.parameters rhs.config before = some right ∧
      left.mem = right.mem ∧ Kernel.Frame lhs.parameters mask before left

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := Parameters
  signature := Program.parameters
  body p := p.source.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs = rhs

def implementation (p : Parameters) (config : Config) : Program := ⟨p,config⟩

/-- Every pair of actual source configurations executes successfully from the
same admitted input memory and agrees on all cells. Opaque payloads include all
FP32 bit patterns. There is no floating arithmetic or numerical assumption. -/
theorem masked_select_configuration_equiv (p : Parameters) (left right : Config)
    (hl : left ∈ configs) (hr : right ∈ configs) :
    implementation p left ≡[([] : Spec.Assumptions ComputeStmt)] implementation p right := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := implementation p left) (rhs := implementation p right) rfl rfl
  intro A _ rest input mask before h sep
  exact Kernel.configurations_equivalent rest p left right hl hr input mask before h sep

theorem shared_source (p : Parameters) (config : Config) :
    (implementation p config).source = kernel {p with width := config.width} := rfl

#print_fp_assumptions masked_select_configuration_equiv
#axiomsClean masked_select_configuration_equiv
#axiomsClean shared_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.ConfigurationEquivalence

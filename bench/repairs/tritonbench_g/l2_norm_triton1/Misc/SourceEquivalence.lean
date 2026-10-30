import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Kernel
import bench.repairs.tritonbench_g.l2_norm_triton1.FPEquiv

namespace VeriTile.Bench.Repairs.L2NormTriton1.SourceEquivalence
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

structure Program where
  kernel : ComputeKernel

def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (A : Algebra α) (row : Nat) (eps : α) (before : State α),
    ∃ after,
      FP.Structural.exec A lhs.kernel (Tile.rowState row eps before) = some after ∧
      FP.Structural.exec A rhs.kernel (Tile.rowState row eps before) = some after

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := List RegionName × List RegionName
  signature p := (p.kernel.inputs,p.kernel.outputs)
  body p := p.kernel.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs = rhs

def original (N B : Nat) : Program := ⟨source N B⟩
def commuted (N B : Nat) : Program := ⟨indexCommuted N B⟩

/-- Both actual full kernels succeed with the identical complete state.
Only integer row-base products commute. FP32 operation/reduction order and
all raw input/epsilon words are retained, with no numerical transformation law. -/
theorem source_equivalence (N B : Nat) :
    original N B ≡[([] : Spec.Assumptions ComputeStmt)] commuted N B := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent) rfl
  · rfl
  intro α _ A row eps before
  obtain ⟨after,success⟩ := Kernel.executes A N B row eps before
  exact ⟨after,success,(Kernel.commuted_execution A N B row eps before).trans success⟩

#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.SourceEquivalence

namespace VeriTile.Bench.Repairs.L2NormTriton1.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- Retained shared IO judgment used by the public two-program conclusion. -/
theorem source_io_equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    (GridIO.gridIO A rows N B eps).Equivalent (GridIO.candidate A rows N B eps) :=
  GridIO.equivalence A rows N B eps

#axiomsClean source_io_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.FPEquiv

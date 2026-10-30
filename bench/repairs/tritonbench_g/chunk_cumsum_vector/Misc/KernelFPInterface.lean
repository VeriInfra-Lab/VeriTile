import bench.repairs.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Rewrite
import bench.repairs.tritonbench_g.chunk_cumsum_vector.FPEquiv

/-! Actual FP32 carry-add commutation, with source-ordered dot/reduction/casts.
Both kernels succeed and agree on all memory. Only active carry updates used
by a later output require the stated ADD-COMMUTE finite-operand domain. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPInterface
open VeriTile Triton FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

abbrev original := Kernels.source
abbrev sumLeft := Kernels.sumLeft
abbrev rewrite_domain := @Rewrite.FiniteCarry

theorem memory_equiv (M : Algebra α) (dot : DotBackend α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (chosen : c∈configurations)
    (p : Geometry.Program h) (u : State α) (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row)
    (finite : rewrite_domain M D h c p u) :
    ∃ a b, FP.Structural.exec (machine M dot) (original h c) u=some a ∧
      FP.Structural.exec (machine M dot) (sumLeft h c) u=some b ∧
      a.mem=b.mem ∧ FPMemory.Frame h p u a ∧ FPMemory.Frame h p u b :=
  Rewrite.successful_pair M dot D model h layout c
    (Kernels.configurations_positive c chosen).1 p u hf hr finite

theorem original_source (h : Call) (c : Config) : original h c=Kernels.source h c := rfl
theorem candidate_source (h : Call) (c : Config) : sumLeft h c=Kernels.sumLeft h c := rfl

#print_fp_assumptions Rewrite.carry_rule
#axiomsClean memory_equiv
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPInterface

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPEquiv
open VeriTile.Triton FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

/-- Retained complete shared IO equivalence, with the original carry domain. -/
theorem memory_io_equiv (M : Algebra α) (dot : DotBackend α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (chosen : c∈configurations)
    (p : Geometry.Program h) :
    (prefixNativeIO M dot h c p).EquivalentOn (sumLeftVariant M dot h c p)
      (fun xs => carryDomain M D h c p (xs (⟨0,by decide⟩ : Fin 1))) :=
  NativeIO.equivalent M dot D model h layout c chosen p

#axiomsClean memory_io_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPEquiv

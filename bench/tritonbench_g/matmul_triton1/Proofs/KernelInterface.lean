import bench.tritonbench_g.matmul_triton1.Proofs.FiniteSource
import bench.tritonbench_g.matmul_triton1.Proofs.Real

namespace VeriTile.Bench.Repairs.MatmulTriton1.RawKernel
open VeriTile.Triton FP.Structural
open Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

def Post {α : Type} (p : Core.Parameters) (before final : State α)
    (value : Core.LogicalIndex p → α) : Prop :=
  GridExecution.Post p.output (Geometry.address (Core.geometry p))
    (fun i => Cell.mk .real (value i)) (fun _ => True) before final

/-- Both complete raw source interpretations have identical memory, all outputs
and the full exterior frame. All premises describe numeric kernel geometry. -/
theorem execute {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input ≠ p.output) (hb : p.weight ≠ p.output) (initial : State α) :
    ∃ finite typed, FiniteSource.grid P p initial = some finite ∧
      Core.grid rest P p initial = some typed ∧ finite.mem = typed.mem ∧
      Post p initial finite (fun i => Core.logical P p initial i.1.val i.2.val) ∧
      Post p initial typed (fun i => Core.logical P p initial i.1.val i.2.val) := by
  have hBK : 0 < p.blockInner := Nat.lt_trans (by decide) ready.blockInner.1
  obtain ⟨finite,hf,fr,ff⟩ := FiniteSource.grid_correct rest P p canonical ready storage geometry ha hb initial
  obtain ⟨typed,ht,tr,tf⟩ := Core.grid_correct rest P p geometry hBK ha hb initial
  refine ⟨finite,typed,hf,ht,?_,⟨fr,ff⟩,⟨tr,tf⟩⟩
  funext R n
  by_cases region : R = p.output
  · subst R
    by_cases written : ∃ i : Core.LogicalIndex p, n = Geometry.address (Core.geometry p) i
    · obtain ⟨i,rfl⟩ := written
      exact (fr i trivial).trans (tr i trivial).symm
    · have outside : ∀ i : Core.LogicalIndex p, True → n ≠ Geometry.address (Core.geometry p) i :=
        fun i _ eq => written ⟨i,eq⟩
      exact (ff _ _ (Or.inr outside)).trans (tf _ _ (Or.inr outside)).symm
  · exact (ff _ _ (Or.inl region)).trans (tf _ _ (Or.inl region)).symm

#axiomsClean execute
end
end VeriTile.Bench.Repairs.MatmulTriton1.RawKernel

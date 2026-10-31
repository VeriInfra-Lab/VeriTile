import bench.tritonbench_g.matmul_triton1.Proofs.SourceIO
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.SourceIOProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
open VeriTile.Bench.Repairs.MatmulTriton1
open Revision001
open MatmulIO
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem complete (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (xs : ChannelValues α (MatmulIO.inputs p)) (initial : State (Word α))
    (hin : (io rest P p).InputBinding xs initial) :
    ∃ final, runSchedule ((io rest P p).schedule xs) initial=some final ∧
      (io rest P p).OutputBinding xs (MatmulIO.precisionTarget P p xs) final ∧
      (io rest P p).Frame xs initial final := by
  obtain ⟨final,ran,read,frame⟩ := grid_correct rest P p canonical ready geometry ha hb initial
  refine ⟨final,?_,?_,?_⟩
  · simpa only [io,schedule_eq,grid] using ran
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨r,c,u⟩
    cases u
    have result := read (r,c) trivial
    change final.mem p.output (r.val*p.outputRow+c.val*p.outputColumn) =
      Cell.mk .real (Core.logical (symbolicPrimitives α) p initial r.val c.val) at result
    change mapCell (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P)
        (final.mem p.output (r.val*p.outputRow+c.val*p.outputColumn)) =
      Cell.mk .real (MatmulIO.precisionAccum P p xs r c (Core.steps p))
    rw [result]
    change Cell.mk .real (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P
      (Core.logical (symbolicPrimitives α) p initial r.val c.val)) = _
    rw [observed_logical]
    exact congrArg (Cell.mk .real) (precision_target rest P p xs initial hin r c _)
  · intro r address outside
    apply frame r address
    by_cases region : r=p.output
    · right
      intro i _
      have h := outside.1 (0 : Fin 1) (i.1,i.2,()) trivial
      exact h.resolve_left (fun hn => hn region)
    · exact Or.inl region

#axiomsClean complete
end
end VeriTile.Bench.Repairs.MatmulTriton1.SourceIOProof

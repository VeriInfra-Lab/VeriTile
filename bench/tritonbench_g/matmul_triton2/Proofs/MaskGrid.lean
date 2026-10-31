import bench.tritonbench_g.matmul_triton2.Proofs.MaskDomain
import bench.tritonbench_g.matmul_triton2.Proofs.Grid
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem correctness (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (scope : SourceBounds p)
    (initial : State (Word α)) (pid : initial.pids 0 < Geometry.count (Core.geometry p))
    (hInj : ∀ i j, Core.active p initial i → Core.active p initial j →
      Core.cOffset p initial i=Core.cOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {})
        (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.maskHoisted p) initial=some final ∧
      Core.Post (symbolicPrimitives α) p initial final := by
  obtain ⟨final,ran,memory⟩ := repaired_source_execution rest P p scope initial pid
  refine ⟨final,ran,?_,?_⟩
  · intro i active
    rw [memory]
    exact masked_readback_active _ _ _ _ _ i active hInj
  · intro r o outside
    rw [memory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (outside.imp id (fun h i _ => h i)) _

def one (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) (pid : Nat)
    (s : State (Word α)) :=
  FP.Structural.exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {})
    (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.maskHoisted p) (atProgram s pid)
def grid (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) (s : State (Word α)) :=
  GridExecution.sequence (one rest P p) (List.range (Geometry.count (Core.geometry p))) s

theorem single_logical (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (scope : SourceBounds p)
    (hgeo : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State (Word α)) (hout : Outside p.output initial current)
    (pid : Nat) (hp : pid<Geometry.count (Core.geometry p)) :
    ∃ final, one rest P p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (Core.geometry p))
        (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical (symbolicPrimitives α) p initial x.1.val x.2.val))
        (Geometry.Owns (Core.geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, Core.active p s i → Core.active p s j →
      Core.cOffset p s i=Core.cOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (Core.geometry p) hgeo pid i j
      ((Core.active_at p current pid i).mp hi) ((Core.active_at p current pid j).mp hj)
    simpa only [s,Core.offset_at p current pid] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness rest P p scope s hp hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : Core.active p s i := (Core.active_at p current pid i).mpr
      (Geometry.owns_active (Core.geometry p) pid x i hr hc)
    have ho : Core.cOffset p s i=Geometry.address (Core.geometry p) x := by
      rw [show s=atProgram current pid from rfl,Core.offset_at p current pid]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,Core.accumulation_logical (symbolicPrimitives α) p s i hi (Core.steps p)]
    change Cell.mk .real (Core.logical (symbolicPrimitives α) p s (Geometry.outRow (Core.geometry p) pid i)
      (Geometry.outColumn (Core.geometry p) pid i))=_
    rw [hr,hc,Core.logical_preserved (symbolicPrimitives α) p initial s ha hb hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (Core.geometry p) pid i ((Core.active_at p current pid i).mp hi)
      have hx : Geometry.Owns (Core.geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : Core.cOffset p s i=Geometry.address (Core.geometry p) x :=
        Core.offset_at p current pid i
      rw [ho]
      exact h x hx

theorem grid_correct (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (scope : SourceBounds p)
    (hgeo : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State (Word α)) :
    ∃ final, grid rest P p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (Core.geometry p))
        (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical (symbolicPrimitives α) p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (Core.geometry p))
    hgeo.writable (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical (symbolicPrimitives α) p initial x.1.val x.2.val))
    (Geometry.Owns (Core.geometry p)) (one rest P p) initial (Geometry.count (Core.geometry p))
    (fun pid hp current hout => single_logical rest P p scope hgeo ha hb initial current hout pid hp)
    (List.range (Geometry.count (Core.geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (Core.geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean correctness
#axiomsClean single_logical
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime

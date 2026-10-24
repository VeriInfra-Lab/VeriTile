import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOPrograms
import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOReference

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem locality (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (domain : IO.domain x.shape o v) (base : RegisterState (SourceGraph.Node α))
    (bound : (IO.programIO x t o rest lanes).InputCells v base) :
    (IO.programIO x t o rest lanes).SourceLocality v base := by
  rintro p listed ⟨port,j⟩ _ left right same a b ra rb
  have hz : port=0 := Fin.eq_zero port
  subst port
  change a.mem "out" (j.1.val*x.shape.n+j.2.1.val)=b.mem "out" (j.1.val*x.shape.n+j.2.1.val)
  by_cases own : IO.owns x.shape t p j.1.val j.2.1.val
  · rw [source_value x geometry t member o rest lanes v domain base bound left p listed j own a ra,
      source_value x geometry t member o rest lanes v domain base bound right p listed j own b rb,same]
  · rw [source_missing x t (Host.all_tiles_valid t member) o rest lanes v base left p j own a ra,
      source_missing x t (Host.all_tiles_valid t member) o rest lanes v base right p j own b rb,same]

theorem coverage (x : Raw.Arguments) (t : Tile) (ht : t.Valid)
    (o : Ops α) (rest : Algebra α) (lanes : List Lane)
    (v : (IO.programIO x t o rest lanes).Inputs) (base : RegisterState (SourceGraph.Node α)) :
    (IO.programIO x t o rest lanes).SourceCoverage v base := by
  rintro ⟨port,j⟩ _ p listed missing current final ran
  have hz : port=0 := Fin.eq_zero port
  subst port
  have miss : ¬IO.owns x.shape t p j.1.val j.2.1.val := by
    intro own
    exact missing (List.mem_filter.mpr ⟨listed,decide_eq_true own⟩)
  exact source_missing x t ht o rest lanes v base current p j miss final ran

theorem trace (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs)
    (domain : IO.domain x.shape o v) (base : RegisterState (SourceGraph.Node α))
    (bound : (IO.programIO x t o rest lanes).InputCells v base)
    (initial : (IO.programIO x t o rest lanes).RawOutputs)
    (j : TileIndex [x.shape.m,x.shape.n]) (selected : List IO.Program)
    (listed : ∀ p∈selected,p∈IO.events x.shape t j.1.val j.2.1.val) (seed : α) :
    (IO.programIO x t o rest lanes).SourceCellTrace v base initial ⟨0,j⟩ selected (SourceGraph.Node.value seed)
      (SourceGraph.Node.value (selected.foldl (fun acc p=>if t.splits=1 then
        Formula.partialSum x.shape t o v j.1.val j.2.1.val p.2 else
        o.atomicAdd acc (Formula.partialSum x.shape t o v j.1.val j.2.1.val p.2)) seed)) := by
  induction selected generalizing seed with
  | nil =>
    exact PerAddressProgramIO.SourceCellTrace.nil
      (io:=IO.programIO x t o rest lanes) (vals:=v) (base:=base) (initial:=initial) (key:=⟨0,j⟩)
      (SourceGraph.Node.value seed)
  | cons p ps ih =>
    have chosen := List.mem_filter.mp (listed p (by simp))
    apply PerAddressProgramIO.SourceCellTrace.cons_of_programs _ _ _ _ _ _ _ _ _ _
      (programs x geometry t member o rest lanes v base) chosen.1
    · intro final ran
      have value := source_value x geometry t member o rest lanes v domain base bound
        ((IO.programIO x t o rest lanes).toMemoryIO.updateOutput initial ⟨0,j⟩ (SourceGraph.Node.value seed))
        p chosen.1 j (of_decide_eq_true chosen.2) final ran
      simpa only [MemoryIO.updateOutput_same,SourceGraph.observe] using value
    · exact ih (fun q hq=>listed q (by simp [hq])) _

def resultValues (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs) :
    (IO.programIO x t o rest lanes).RawOutputs := fun key => by
  rcases key with ⟨port,j⟩
  have hz : port=0 := Fin.eq_zero port
  subst port
  exact SourceGraph.Node.value (Formula.observed x.shape t o v j.1.val j.2.1.val
    (IO.order x.shape t lanes j.1.val j.2.1.val))

/-- Shared IO correctness of the complete unchanged source, with finite exact
input cells, all-CTA safety/success, locality, coverage, outputs and raw frame. -/
theorem implements (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t)) :
    (IO.programIO x t o rest lanes).ImplementsOnCells (IO.domain x.shape o)
      (Formula.nativeOutput x.shape t o lanes) := by
  intro v domain base bound initial output
  refine ⟨schedule_complete x geometry t member o rest lanes complete v,
    programs x geometry t member o rest lanes v base,
    locality x geometry t member o rest lanes v domain base bound,
    coverage x t (Host.all_tiles_valid t member) o rest lanes v base,
    resultValues x t o rest lanes v,?_,?_,?_⟩
  · rintro ⟨port,j⟩ _
    have hz : port=0 := Fin.eq_zero port
    subst port
    rw [initial_zero x t o rest lanes v domain base bound initial output j]
    exact trace x geometry t member o rest lanes v domain base bound initial j _
      (fun p hp=>(order_perm x.shape geometry.shape t (Host.all_tiles_valid t member) lanes complete _ _).mem_iff.mp hp) o.zero
  · intro port j _
    have hz : port=0 := Fin.eq_zero port
    subst port
    have value := (IO.programIO x t o rest lanes).toMemoryIO.patchOutputs_read v
      (output_unique x t o rest lanes v) (resultValues x t o rest lanes v) base ⟨0,j⟩ (by trivial)
    change ((IO.programIO x t o rest lanes).patchOutputs v (resultValues x t o rest lanes v) base).mem
      "out" (j.1.val*x.shape.n+j.2.1.val)=Cell.mk .real (resultValues x t o rest lanes v ⟨0,j⟩) at value
    change mapCell _ (((IO.programIO x t o rest lanes).patchOutputs v
      (resultValues x t o rest lanes v) base).mem "out" (j.1.val*x.shape.n+j.2.1.val))=_
    rw [value]
    rfl
  · exact (IO.programIO x t o rest lanes).toMemoryIO.patchOutputs_frame v _ base

theorem real_implements (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (rest : Algebra ℝ)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t)) :
    (IO.programIO x t realOps rest lanes).ImplementsOnCells (IO.domain x.shape realOps)
      (Formula.realOutput x.shape) := by
  have equal : Formula.nativeOutput x.shape t realOps lanes=Formula.realOutput x.shape := by
    funext v port j
    have hz : port=0 := Fin.eq_zero port
    subst port
    exact real_formula x geometry t member lanes complete v j.1.val j.2.1.val j.1.isLt j.2.1.isLt
  rw [←equal]
  exact implements x geometry t member realOps rest lanes complete

#axiomsClean implements
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof

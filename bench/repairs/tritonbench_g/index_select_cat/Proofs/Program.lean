import bench.repairs.tritonbench_g.index_select_cat.Proofs.TypedGather
import bench.repairs.tritonbench_g.index_select_cat.Proofs.Lift

namespace VeriTile.Bench.Repairs.IndexSelectCat.Program
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Lane := TileIndex [1,512]
def metadataRegion : RegionName := ⟨"index_select_private_indices"⟩

/-- One program from the actual two-dimensional gather launch. -/
structure Plan where
  (inputBase outputBase indexBase localRow colTile rows cols stride : Nat)
  indices : Nat → Nat

def layout (p : Plan) : Layout where
  source := ProgramGrid.sourceRegion
  output := ProgramGrid.outputRegion
  index := metadataRegion
  rows := p.rows
  cols := p.cols
  rowStride := p.stride
  colStride := 1
  BI := 1
  BC := 512

def initial {α : Type} [Inhabited α] (p : Plan) (input output : Nat → α) : State α where
  mem := fun r o => if r=ProgramGrid.sourceRegion then .mk .real (input (p.inputBase+o))
    else if r=metadataRegion then .mk .int (p.indices (p.indexBase+o))
    else .mk .real (output (p.outputBase+o))
  regs := fun _ _ _ => none
  pids := fun d => if d=0 then p.localRow else if d=1 then p.colTile else 0
  numPids := fun d => if d=0 then p.rows else if d=1 then (p.cols+511)/512 else 1
  undef := fun d _ _ => defaultValue d

def localOutput (p : Plan) (i : Lane) : Nat := p.localRow*p.stride+(p.colTile*512+i.2.1.val)

def key (p : Plan) (i : Lane) : Nat := p.outputBase+localOutput p i

def value {α : Type} (input : Nat → α) (p : Plan) (i : Lane) : α :=
  input (p.inputBase+(p.indices (p.indexBase+p.localRow)*p.stride+(p.colTile*512+i.2.1.val)))

def lanes (p : Plan) : List Lane :=
  (TileShape.allIndices [1,512]).filter (fun i => decide (p.localRow<p.rows ∧ p.colTile*512+i.2.1.val<p.cols))

@[irreducible] def actualStep {α : Type} [Inhabited α] (rest : Algebra α) (input : Nat → α)
    (p : Plan) (output : Nat → α) : Nat → α :=
  ProgramGrid.liftOutput p.outputBase output
    (TypedGather.execution rest (layout p) (initial p input output)).val

theorem initial_lift {α : Type} [Inhabited α] (p : Plan) (input output : Nat → α) :
    ProgramGrid.liftOutput p.outputBase output (initial p input output)=output := by
  funext q
  by_cases h : p.outputBase≤q
  · simp [ProgramGrid.liftOutput,initial,h,ProgramGrid.sourceRegion,
      ProgramGrid.outputRegion,metadataRegion,Nat.add_sub_of_le h]
  · simp [ProgramGrid.liftOutput,h]

theorem actualStep_eq_scatter {α : Type} [Inhabited α] (rest : Algebra α)
    (input : Nat → α) (p : Plan) (output : Nat → α) :
    actualStep rest input p output=Scatter.run (lanes p) (key p) (value input p) output := by
  unfold actualStep
  have hm := TypedGather.execution_memory rest (layout p) (initial p input output)
  have hl : ProgramGrid.liftOutput p.outputBase output
      (TypedGather.execution rest (layout p) (initial p input output)).val=
      ProgramGrid.liftOutput p.outputBase output (TypedGather.copied (layout p) (initial p input output)) := by
    funext q
    simp only [ProgramGrid.liftOutput,hm]
  rw [hl]
  unfold TypedGather.copied
  rw [State.masked_scatter_eq_filter]
  have hz (i : Lane) : i.1.val=0 := by have h := i.1.isLt; omega
  have hv (i : Lane) : TypedGather.loaded (layout p) (initial p input output) i=
      Cell.mk .real (value input p i) := by
    simp [TypedGather.loaded,TypedGather.inputOffset,TypedGather.row,
      TypedGather.indexOffset,TypedGather.column,layout,initial,value,hz i,
      ProgramGrid.sourceRegion,metadataRegion]
    norm_cast
  have ho (i : Lane) : TypedGather.outputOffset (layout p) (initial p input output) i=localOutput p i := by
    simp [TypedGather.outputOffset,TypedGather.indexOffset,TypedGather.column,layout,initial,localOutput,hz i]
  have ha (i : Lane) : TypedGather.active (layout p) (initial p input output) i ↔
      p.localRow<p.rows ∧ p.colTile*512+i.2.1.val<p.cols := by
    simp [TypedGather.active,TypedGather.indexOffset,TypedGather.column,layout,initial,hz i]
  simp only [hv,ho,ha]
  change ProgramGrid.liftOutput p.outputBase output
    ((lanes p).foldl (fun t i => t.write ProgramGrid.outputRegion (localOutput p i)
      (Cell.mk .real (value input p i))) (initial p input output)) = _
  rw [ProgramGrid.lift_scatter,initial_lift]
  rfl

def run {α : Type} [Inhabited α] (rest : Algebra α) (input : Nat → α)
    (plans : List Plan) (output : Nat → α) : Nat → α :=
  plans.foldl (fun state p => actualStep rest input p state) output

def expanded (plans : List Plan) : List (Plan × Lane) :=
  plans.flatMap (fun p => (lanes p).map (fun i => (p,i)))

theorem run_eq_scatter {α : Type} [Inhabited α] (rest : Algebra α) (input : Nat → α)
    (plans : List Plan) (output : Nat → α) :
    run rest input plans output=Scatter.run (expanded plans)
      (fun i => key i.1 i.2) (fun i => value input i.1 i.2) output := by
  induction plans generalizing output with
  | nil => rfl
  | cons p tail ih =>
    change run rest input tail (actualStep rest input p output) = _
    rw [ih,actualStep_eq_scatter]
    simp only [expanded,List.flatMap_cons,Scatter.run,List.foldl_append,List.foldl_map]

#axiomsClean actualStep_eq_scatter
#axiomsClean run_eq_scatter
end
end VeriTile.Bench.Repairs.IndexSelectCat.Program

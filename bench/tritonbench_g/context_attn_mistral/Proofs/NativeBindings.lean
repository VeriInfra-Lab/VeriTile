import bench.tritonbench_g.context_attn_mistral.NativeArrays
import bench.tritonbench_g.context_attn_mistral.Footprint

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeBindings
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

def snapshot (c : Host.Call) (s : State α) : NativeArrays.Inputs c α :=
  fun port off=>(s.mem (NativeArrays.ports c)[port].buf off).read (NativeArrays.ports c)[port].dtype

theorem metadata_iff (c : Host.Call) (s : State α) :
    NativeArrays.metadata c (snapshot c s) ↔ KernelFootprint.Metadata c s := Iff.rfl

theorem tileInput_snapshot (c : Host.Call) (idx : GridIndex c.grid) (s : State α) :
    NativeArrays.tileInput c idx (snapshot c s)=
      KernelReference.inputs c.layout c.regions (KernelReference.atIndex s idx) := rfl

theorem store_snapshot (c : Host.Call) (s : State α) (off : Nat) (value : α) :
    snapshot c (s.write c.regions.out off (.mk .real value))=
      NativeArrays.store c off value (snapshot c s) := by
  have read (region : RegionName) (address : Nat) (dtype : TileDType) :
      ((s.write c.regions.out off (.mk .real value)).mem region address).read dtype=
        if region=c.regions.out ∧ address=off then (Cell.mk .real value).read dtype
        else (s.mem region address).read dtype := by
    by_cases hit:region=c.regions.out ∧ address=off <;> simp [State.write,hit]
  funext port address
  fin_cases port
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.query address).read .real=
      if c.regions.query=c.regions.out ∧ address=off then value else (s.mem c.regions.query address).read .real
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.key address).read .real=
      if c.regions.key=c.regions.out ∧ address=off then value else (s.mem c.regions.key address).read .real
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.value address).read .real=
      if c.regions.value=c.regions.out ∧ address=off then value else (s.mem c.regions.value address).read .real
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.start address).read .int=
      if c.regions.start=c.regions.out ∧ address=off then (0 : Int) else (s.mem c.regions.start address).read .int
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.sequence address).read .int=
      if c.regions.sequence=c.regions.out ∧ address=off then (0 : Int) else (s.mem c.regions.sequence address).read .int
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.out address).read .real=
      if address=off then value else (s.mem c.regions.out address).read .real
    rw [read]
    simp [Cell.read,defaultValue]

theorem scatter_snapshot (c : Host.Call) (idx : GridIndex c.grid) (s t : State α)
    (metadata : KernelFootprint.Metadata c s) (values : Values α .real [c.layout.BM,c.layout.D]) :
    snapshot c (KernelReference.scatter c.layout c.regions (KernelReference.atIndex s idx) t values)=
      NativeArrays.scatter c idx values (snapshot c t) := by
  have active:=KernelFootprint.active_matches c s metadata idx
  have offset:=KernelFootprint.offset_matches c s metadata idx
  unfold KernelReference.scatter NativeArrays.scatter
  generalize TileShape.allIndices [c.layout.BM,c.layout.D]=lanes
  induction lanes generalizing t with
  | nil=>rfl
  | cons i rest ih=>
    simp only [List.foldl_cons,active i,offset i]
    by_cases on:c.active idx i
    · simp only [if_pos on]
      rw [ih,store_snapshot]
    · simp only [if_neg on]
      exact ih t

theorem tile_metadata (c : Host.Call) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s : State α) (metadata : KernelFootprint.Metadata c s)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    KernelFootprint.Metadata c (KernelReference.tile c o idx s) := by
  have untouched (rg : RegionName) (different : c.regions.out≠rg) (off : Nat) :
      (KernelReference.tile c o idx s).mem rg off=s.mem rg off := by
    apply KernelFootprint.scatter_frame
    intro i on same address
    exact different same.symm
  constructor
  · intro b hb
    rw [untouched _ separate.1]
    exact metadata.1 b hb
  · intro b hb
    rw [untouched _ separate.2]
    exact metadata.2 b hb

theorem tile_snapshot (c : Host.Call) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s : State α) (metadata : KernelFootprint.Metadata c s) :
    snapshot c (KernelReference.tile c o idx s)=
      NativeArrays.tile c o idx (snapshot c s) := by
  simp only [KernelReference.tile,NativeArrays.tile]
  rw [←tileInput_snapshot]
  exact scatter_snapshot c idx s (KernelReference.atIndex s idx) metadata _

theorem grid_snapshot (c : Host.Call) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : KernelFootprint.Metadata c s)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    snapshot c (KernelReference.grid c o order s)=
      NativeArrays.grid c o order (snapshot c s) := by
  induction order generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    rw [KernelReference.grid,NativeArrays.grid,
      ih _ (tile_metadata c o idx s metadata separate),tile_snapshot c o idx s metadata]

#axiomsClean store_snapshot
#axiomsClean scatter_snapshot
#axiomsClean grid_snapshot
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeBindings

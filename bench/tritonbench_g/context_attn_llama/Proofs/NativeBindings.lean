import bench.tritonbench_g.context_attn_llama.NativeArrays
import bench.tritonbench_g.context_attn_llama.KernelInterface

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

def snapshot (c : Host.Call) (s : State α) : NativeArrays.Inputs c α :=
  fun port off=>(s.mem (NativeArrays.ports c)[port].buf off).read (NativeArrays.ports c)[port].dtype

theorem metadata_iff (c : Host.Call) (s : State α) :
    NativeArrays.metadata c (snapshot c s) ↔ FPSuccess.Metadata c s := Iff.rfl

theorem tileInput_snapshot (c : Host.Call) (idx : GridIndex c.grid) (s : State α) :
    NativeArrays.tileInput c idx (snapshot c s)=
      FPReference.inputs c.layout c.regions (FPSuccess.atIndex s idx) := rfl

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
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.prompt address).read .int=
      if c.regions.prompt=c.regions.out ∧ address=off then (0 : Int) else (s.mem c.regions.prompt address).read .int
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.request address).read .int=
      if c.regions.request=c.regions.out ∧ address=off then (0 : Int) else (s.mem c.regions.request address).read .int
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.pages address).read .int=
      if c.regions.pages=c.regions.out ∧ address=off then (0 : Int) else (s.mem c.regions.pages address).read .int
    rw [read]
    simp [Cell.read,defaultValue]
  · change ((s.write c.regions.out off (.mk .real value)).mem c.regions.out address).read .real=
      if address=off then value else (s.mem c.regions.out address).read .real
    rw [read]
    simp [Cell.read,defaultValue]

theorem scatter_snapshot (c : Host.Call) (idx : GridIndex c.grid) (s t : State α)
    (metadata : FPSuccess.Metadata c s) (values : Values α .real [c.layout.BM,c.layout.D]) :
    snapshot c (FPSuccess.scatter c.layout c.regions (FPSuccess.atIndex s idx) t values)=
      NativeArrays.scatter c idx values (snapshot c t) := by
  have active:=FPSuccess.active_matches c s metadata idx
  have offset:=FPSuccess.offset_matches c s metadata idx
  unfold FPSuccess.scatter NativeArrays.scatter
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

theorem tile_metadata (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (idx : GridIndex c.grid) (s : State α) (metadata : FPSuccess.Metadata c s)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out) :
    FPSuccess.Metadata c (FPReference.tile c M dots idx s) := by
  refine ⟨?_,?_,?_⟩
  · intro b hb
    rw [RawKernel.tile_outside c M dots idx s _ _ separate.1]
    exact metadata.1 b hb
  · intro b hb
    rw [RawKernel.tile_outside c M dots idx s _ _ separate.2.1]
    exact metadata.2.1 b hb
  · intro b hb
    rw [RawKernel.tile_outside c M dots idx s _ _ separate.2.2]
    exact metadata.2.2 b hb

theorem tile_snapshot (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (idx : GridIndex c.grid) (s : State α) (metadata : FPSuccess.Metadata c s) :
    snapshot c (FPReference.tile c M dots idx s)=
      NativeArrays.tile c M dots idx (snapshot c s) := by
  simp only [FPReference.tile,NativeArrays.tile]
  rw [←tileInput_snapshot]
  exact scatter_snapshot c idx s (FPSuccess.atIndex s idx) metadata _

theorem grid_snapshot (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : FPSuccess.Metadata c s)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out) :
    snapshot c (FPReference.grid c M dots order s)=
      NativeArrays.grid c M dots order (snapshot c s) := by
  induction order generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    rw [FPReference.grid,NativeArrays.grid,
      ih _ (tile_metadata c M dots idx s metadata separate),tile_snapshot c M dots idx s metadata]

#axiomsClean store_snapshot
#axiomsClean scatter_snapshot
#axiomsClean grid_snapshot
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings

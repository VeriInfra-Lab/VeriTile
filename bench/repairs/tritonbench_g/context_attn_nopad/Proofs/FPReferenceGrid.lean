import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPReferenceProgram
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPHostSource

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPReference
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open FP.ProfiledRewrite (engine)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def tile (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid) (s : State α) : State α :=
  let launch := FPHost.atProgram c idx s
  scatter c.layout r (inputs r launch) launch (output c.layout o c.scale (inputs r launch))

def grid (c : Call) (r : Regions) (o : FPBackend.Ops α) : List (GridIndex c.grid) → State α → State α
  | [],s => s
  | idx::rest,s => grid c r o rest (tile c r o idx s)

def runGrid (c : Call) (r : Regions) (o : FPBackend.Ops α) : List (GridIndex c.grid) → State α → Option (State α)
  | [],s => some s
  | idx::rest,s => do
    let t ← FP.Structural.exec (engine o.algebra) (kernel c r) (FPHost.atProgram c idx s)
    runGrid c r o rest t

theorem tile_mem_congr (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s t : State α) (memory : s.mem=t.mem) :
    (tile c r o idx s).mem=(tile c r o idx t).mem := by
  have inp : inputs r (FPHost.atProgram c idx s)=inputs r (FPHost.atProgram c idx t) := by
    simp only [inputs,FPHost.atProgram,memory]
  dsimp only [tile]
  rw [inp]
  exact scatter_mem_congr c.layout r _ _ _ _ memory

theorem grid_mem_congr (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s t : State α) (memory : s.mem=t.mem) :
    (grid c r o order s).mem=(grid c r o order t).mem := by
  induction order generalizing s t with
  | nil => exact memory
  | cons idx rest ih => exact ih _ _ (tile_mem_congr c r o idx s t memory)

theorem whole (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) :
    ∃ t,runGrid c r o order s=some t ∧ t.mem=(grid c r o order s).mem := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,he,hm⟩ := program c.layout r c.scale o (by norm_num [Call.layout]) (FPHost.atProgram c idx s)
    obtain ⟨final,hfinal,hfm⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [runGrid,kernel,Call.source] at *
      rw [he]
      exact hfinal
    · exact hfm.trans (grid_mem_congr c r o rest mid (tile c r o idx s) hm)

def laneActive (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Prop :=
  (c.launch idx).block*128+i.1.val < c.length (c.launch idx).batch

def laneOffset (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  c.o.offset (c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val)) (c.launch idx).head i.2.1.val

theorem active_matches (c : Call) (r : Regions) (s : State α) (hm : FPHostSource.Metadata c r s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    active c.layout (inputs r (FPHost.atProgram c idx s)) i ↔ laneActive c idx i := by
  have hb := launch_batch_lt c idx
  change ((idx.toPids 2*c.layout.BM+i.1.val : Nat) : Int) <
    (s.mem r.sequence (idx.toPids 0)).read .int ↔ _
  have e := hm.length (c.launch idx).batch hb
  change (s.mem r.sequence (idx.toPids 0)).read .int=(c.length (c.launch idx).batch : Int) at e
  rw [e]
  exact_mod_cast (Iff.rfl : laneActive c idx i ↔ laneActive c idx i)

theorem offset_matches (c : Call) (r : Regions) (s : State α) (hm : FPHostSource.Metadata c r s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    offset c.layout (inputs r (FPHost.atProgram c idx s)) i=laneOffset c idx i := by
  have e := hm.start (c.launch idx).batch (launch_batch_lt c idx)
  unfold offset inputs FPHost.atProgram row
  simp only
  change (s.mem r.start (idx.toPids 0)).read .int=(c.start (c.launch idx).batch : Int) at e
  rw [e]
  simp only [laneOffset,Tensor3.offset,Call.layout,Call.launch,GridIndex.toPids,Call.grid,Grid.rank]
  norm_cast

def Writes (c : Call) (r : Regions) (order : List (GridIndex c.grid)) (rg : RegionName) (off : Nat) : Prop :=
  ∃ idx∈order,∃ i,laneActive c idx i ∧ rg=r.out ∧ off=laneOffset c idx i

theorem tile_frame (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s : State α) (metadata : FPHostSource.Metadata c r s) (rg : RegionName) (off : Nat)
    (outside : ¬∃ i,laneActive c idx i ∧ rg=r.out ∧ off=laneOffset c idx i) :
    (tile c r o idx s).mem rg off=s.mem rg off := by
  classical
  dsimp only [tile,scatter]
  apply State.masked_scatter_frame
  by_cases eq : rg=r.out
  · right
    intro i _ on address
    apply outside
    exact ⟨i,(active_matches c r s metadata idx i).mp on,eq,
      address.trans (offset_matches c r s metadata idx i)⟩
  · exact Or.inl eq

theorem grid_frame (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : FPHostSource.Metadata c r s)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence) :
    FPHostSource.Metadata c r (grid c r o order s) ∧
    ∀ rg off,¬Writes c r order rg off → (grid c r o order s).mem rg off=s.mem rg off := by
  induction order generalizing s with
  | nil => exact ⟨metadata,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    have unchanged (rg : RegionName) (different : r.out≠rg) (off : Nat) :
        (tile c r o idx s).mem rg off=s.mem rg off :=
      tile_frame c r o idx s metadata rg off (by rintro ⟨i,_,eq,_⟩; exact different eq.symm)
    have mid : FPHostSource.Metadata c r (tile c r o idx s) := by
      constructor
      · intro b hb; rw [unchanged r.start startLabel b]; exact metadata.start b hb
      · intro b hb; rw [unchanged r.sequence lengthLabel b]; exact metadata.length b hb
    obtain ⟨final,frame⟩ := ih _ mid
    refine ⟨final,?_⟩
    intro rg off missing
    rw [grid,frame rg off (by rintro ⟨j,hj,i,hi,hr,ho⟩; exact missing ⟨j,List.mem_cons_of_mem idx hj,i,hi,hr,ho⟩)]
    exact tile_frame c r o idx s metadata rg off (by rintro ⟨i,hi,hr,ho⟩; exact missing ⟨idx,List.mem_cons_self,i,hi,hr,ho⟩)

#specNonCircular grid avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,runGrid,kernel]
#axiomsClean whole
#axiomsClean grid_frame
end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPReference

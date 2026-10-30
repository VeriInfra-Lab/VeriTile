import bench.tritonbench_g.context_attn_mistral.Proofs.FPTotal
import bench.tritonbench_g.context_attn_mistral.Proofs.Copyback

namespace VeriTile.Bench.Repairs.ContextAttnMistral.FPHost
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
namespace Composition
export VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostFPComposition
  (prepare schedule allPrograms allPrograms_complete)
end Composition
noncomputable section
set_option maxHeartbeats 1000000

/-- The complete source, including both separate dot/add sites, all source
casts, sentinel guards and the actual rounded launch scale. -/
def source (c : Host.Call) (storage : Storage) : FP.ProfiledRewrite.Program where
  kernel := FPSymbolic.program c.layout c.regions storage (Host.scale c.layout.D)
  domain := []

/-- Every finite sequence of actual grid programs succeeds. Registers are
reset for each launch coordinate; memory is threaded through the sequence. -/
theorem successful_schedule {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Host.Call) (storage : Storage) (jobs : List (GridIndex c.grid)) (s : State α) :
    ∃ final, Composition.schedule o.algebra (source c storage) c.grid jobs s = some final ∧
      FPSymbolic.RegionFrame c.regions.out s final := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,hm,hframe⟩ := FPSymbolic.complete o c.layout c.regions storage
      (Host.scale c.layout.D) (Composition.prepare c.grid idx s) (by rw [c.blockN]; omega)
    obtain ⟨final,hf,hff⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · change (FP.Structural.exec (engine o.algebra)
        (FPSymbolic.program c.layout c.regions storage (Host.scale c.layout.D))
        (Composition.prepare c.grid idx s)).bind _ = some final
      rw [hm]
      exact hf
    · intro rg off hrg
      exact (hff rg off hrg).trans (hframe rg off hrg)

/-- Finite enumeration includes every program in the actual three-axis grid. -/
theorem successful_grid {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Host.Call) (storage : Storage) (s : State α) :
    ∃ final, Composition.schedule o.algebra (source c storage) c.grid
        (Composition.allPrograms c.grid) s = some final ∧
      FPSymbolic.RegionFrame c.regions.out s final :=
  successful_schedule o c storage _ s

abbrev Coord := Nat × Nat × Nat

structure Span where
  start : Nat
  length : Nat

/-- These descriptors are read together before copyback. Subsequent changes
to an aliased original metadata allocation cannot change this list. -/
def spans (c : Host.Call) : List Span :=
  (List.range c.batches).map fun b => ⟨c.start b,c.length b⟩

def slice (c : Host.Call) (span : Span) (i : Coord) : Prop :=
  i.2.1 < c.heads ∧ i.2.2 < c.layout.D ∧
    ∃ row, row < min span.length (c.blocks*128) ∧ i.1 = span.start+row

def active (c : Host.Call) : Coord → Prop := OutputView.SliceActive (slice c) (spans c)

/-- Exact membership in the Python slices, including an underlaunched grid
and zero-length/empty slices. No row beyond min(length, grid_z*128) is copied. -/
theorem active_iff (c : Host.Call) (i : Coord) : active c i ↔
    i.2.1 < c.heads ∧ i.2.2 < c.layout.D ∧
      ∃ b, b < c.batches ∧ ∃ row, row < c.length b ∧ row < c.blocks*128 ∧ i.1 = c.start b+row := by
  simp only [active,OutputView.SliceActive,spans,List.mem_map,List.mem_range,slice]
  constructor
  · rintro ⟨span,⟨b,hb,rfl⟩,hh,hd,row,hr,he⟩
    exact ⟨hh,hd,b,hb,row,(lt_min_iff.mp hr).1,(lt_min_iff.mp hr).2,he⟩
  · rintro ⟨hh,hd,b,hb,row,hr,hg,he⟩
    exact ⟨⟨c.start b,c.length b⟩,⟨b,hb,rfl⟩,hh,hd,row,lt_min hr hg,he⟩

/-- The slices are exactly the source grid's active row lanes. This is the
integer geometry of (batch, head, block), not a numerical/output premise. -/
theorem active_grid_iff (c : Host.Call) (i : Coord) : active c i ↔
    i.2.1 < c.heads ∧ i.2.2 < c.layout.D ∧
      ∃ b, b < c.batches ∧ ∃ block, block < c.blocks ∧ ∃ lane, lane < 128 ∧
        block*128+lane < c.length b ∧ i.1 = c.start b+(block*128+lane) := by
  rw [active_iff]
  constructor
  · rintro ⟨hh,hd,b,hb,row,hr,hg,he⟩
    have hm : row/128 < c.blocks := (Nat.div_lt_iff_lt_mul (by omega)).mpr hg
    have hl : row%128 < 128 := Nat.mod_lt _ (by omega)
    have hrow : row/128*128+row%128 = row := by omega
    exact ⟨hh,hd,b,hb,row/128,hm,row%128,hl,by simpa only [hrow] using hr,
      by simpa only [hrow] using he⟩
  · rintro ⟨hh,hd,b,hb,block,hm,lane,hl,hr,he⟩
    exact ⟨hh,hd,b,hb,block*128+lane,hr,by omega,he⟩

def offset (c : Host.Call) (i : Coord) : Nat :=
  i.1*c.layout.outRow+i.2.1*c.layout.outHead+i.2.2*c.layout.outDim

def readFloat {α : Type} [Inhabited α] (storage : Storage) (cell : Cell α) : α :=
  match storage with
  | .fp16 => cell.read .fp16 | .bf16 => cell.read .bf16 | .fp32 => cell.read .real

def values {α : Type} [Inhabited α] (c : Host.Call) (storage : Storage) (s : State α) (i : Coord) : α :=
  readFloat storage (s.mem c.regions.out (offset c i))

structure LogicalInputs (α : Type) where
  query : Nat → Nat → Nat → α
  key : Nat → Nat → Nat → α
  value : Nat → Nat → Nat → α

/-- Ordinary resolution/clone contracts on entry, independent of execution.
Only private scratch is separate; the original supplied output may alias any
input or metadata storage. `before` names that original allocation. -/
structure Prepared {α : Type} [Inhabited α] (c : Host.Call) (storage : Storage)
    (s : State α) (logical : LogicalInputs α) (view : OutputView.View Coord α)
    (before : Nat → α) : Prop where
  start : ∀ b, b < c.batches → (s.mem c.regions.start b).read .int = (c.start b : Int)
  length : ∀ b, b < c.batches → (s.mem c.regions.sequence b).read .int = (c.length b : Int)
  query : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    readFloat storage (s.mem c.regions.query ((c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim)) =
      logical.query (c.start b+row) h d
  key : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    readFloat storage (s.mem c.regions.key ((c.start b+row)*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim)) =
      logical.key (c.start b+row) (h/c.layout.kvGroup) d
  value : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    readFloat storage (s.mem c.regions.value ((c.start b+row)*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim)) =
      logical.value (c.start b+row) (h/c.layout.kvGroup) d
  initialized : ∀ i, active c i → values c storage s i = view.read before i
  separate : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] →
    rg ≠ c.regions.out

def Exec {α : Type} [Inhabited α] (o : FPBackend.Ops α) (c : Host.Call) (storage : Storage)
    (s : State α) (view : OutputView.View Coord α) (before : Nat → α)
    (result : Option Unit × (Nat → α)) : Prop :=
  result.1 = none ∧ ∃ work,
    Composition.schedule o.algebra (source c storage) c.grid (Composition.allPrograms c.grid) s = some work ∧
    result.2 = view.copySlices (slice c) (values c storage work) (spans c) before

/-- Readback is the successful complete typed reference grid on initially
resolved logical inputs. All original cells outside active copyback are framed.
The private reference also preserves every input/metadata region. -/
def Result {α : Type} [Inhabited α] (o : FPBackend.Ops α) (c : Host.Call) (storage : Storage)
    (s : State α) (view : OutputView.View Coord α) (before after : Nat → α) : Prop :=
  ∃ reference,
    Composition.schedule o.algebra (source c storage) c.grid (Composition.allPrograms c.grid) s = some reference ∧
    (∀ i, active c i → view.read after i = values c storage reference i) ∧
    (∀ a, (¬ ∃ i, active c i ∧ view.addr i = a) → after a = before a) ∧
    (∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] →
      ∀ off, reference.mem rg off = s.mem rg off)

/-- Successful source execution, every grid coordinate, snapshot-driven ordered
slice copies, original-view readback, original-storage frame and None return.
No whole-kernel success or value equality is assumed. -/
theorem whole_call {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Host.Call) (storage : Storage) (s : State α) (logical : LogicalInputs α)
    (view : OutputView.View Coord α) (before : Nat → α)
    (prepared : Prepared c storage s logical view before) (writable : view.Writable (active c)) :
    ∃ result, Exec o c storage s view before result ∧ Result o c storage s view before result.2 := by
  obtain ⟨work,hw,hframe⟩ := successful_grid o c storage s
  let after := view.copySlices (slice c) (values c storage work) (spans c) before
  refine ⟨(none,after),⟨rfl,work,hw,rfl⟩,work,hw,?_,?_,?_⟩
  · intro i hi
    exact OutputView.copySlices_at view (slice c) _ (spans c) before writable i hi
  · intro a ha
    exact OutputView.copySlices_frame view (slice c) _ (spans c) before a ha
  · intro rg hr off
    exact hframe rg off (prepared.separate rg hr)

end
end VeriTile.Bench.Repairs.ContextAttnMistral.FPHost

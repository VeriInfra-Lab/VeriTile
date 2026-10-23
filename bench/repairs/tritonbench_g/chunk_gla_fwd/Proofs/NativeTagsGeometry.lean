import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.HostMetadata

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
open VeriTile Triton FP.Structural
open FPStages (Stage)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- The attention cell's last-writer type, computed only from grid coordinates,
active store addresses and fixed store types. No source evaluator, State or
floating arithmetic participates in this recurrence. Split/output stages write
other regions, so their attention transition is the identity. -/
def stageTag (stage : Stage) (pid : Nat → Nat) (off : Nat) (before : TileDType) : TileDType :=
  match stage with
  | .inter c _ =>
    if pid 0*c.BT+pid 1/c.NC*16<c.T ∧ pid 1%c.NC<pid 1/c.NC then
      (TileShape.allIndices [16,16]).foldl (fun tag i =>
        if pid 0*c.BT+pid 1/c.NC*16+i.1.val<c.T ∧ pid 1%c.NC*16+i.2.1.val<c.BT ∧
          off=pid 2*c.T*c.BT+(pid 0*c.BT+pid 1/c.NC*16+i.1.val)*c.BT+(pid 1%c.NC*16+i.2.1.val)
        then .fp32 else tag) before
    else before
  | .intra c _ _ =>
    if pid 0*c.BT+pid 1*16<c.T then
      foldRange (fun j tag => (TileShape.allIndices [16]).foldl (fun tag i =>
        if pid 0*c.BT+pid 1*16+i.1.val<c.T ∧
          off=pid 2*c.T*c.BT+(pid 0*c.BT+pid 1*16+i.1.val)*c.BT+pid 1*16+j
        then .real else tag) tag)
        0 (min 16 (c.T-pid 0*c.BT-pid 1*16)) 1 before
    else before
  | .merge c =>
    if pid 0*c.BT+pid 1*16+0<c.T then
      (TileShape.allIndices [16,16]).foldl (fun tag i =>
        if pid 0*c.BT+pid 1*16+i.1.val<c.T ∧ pid 1*16+i.2.1.val<c.BT ∧
          off=pid 2*c.T*c.BT+(pid 0*c.BT+pid 1*16+i.1.val)*c.BT+(pid 1*16+i.2.1.val)
        then .fp32 else tag) before
    else before
  | .split _ _ | .output _ _ => before

def gridTag (stage : Stage) (off : Nat) (before : TileDType) : TileDType :=
  (FPGrid.jobs stage.grid).foldl (fun tag idx => stageTag stage idx.toPids off tag) before

/-- Complete producer order on attention tags. Attention initialization writes
real-tagged FP32 zero. The large-K scratch fill preserves attention; the full
split grid is retained and likewise has no attention writes. The scale is
absent because it cannot affect a store address or store type. -/
def producerTag (c : Host.Dimensions) (off : Nat) : TileDType :=
  if c.heads*c.T=0 then .real else
    let inter := gridTag (.inter c.producer 0) off .real
    if c.K≤256 then gridTag (.intra c.producer c.intraBK 0) off inter
    else gridTag (.merge c.split) off (gridTag (.split c.split 0) off inter)

def producerExplicit (c : Host.Dimensions) (off : Nat) : Bool :=
  decide (producerTag c off=.fp32)

def BinaryTag (tag : TileDType) : Prop := tag=.real ∨ tag=.fp32

theorem fold_binary {β : Type} (xs : List β) (f : TileDType → β → TileDType)
    (hf : ∀ b i,BinaryTag b → BinaryTag (f b i)) (b : TileDType) (hb : BinaryTag b) :
    BinaryTag (xs.foldl f b) := by
  induction xs generalizing b with
  | nil => exact hb
  | cons i rest ih => exact ih _ (hf b i hb)

theorem range_binary (f : Nat → TileDType → TileDType)
    (hf : ∀ i b,BinaryTag b → BinaryTag (f i b)) (start stop : Nat)
    (b : TileDType) (hb : BinaryTag b) : BinaryTag (foldRange f start stop 1 b) := by
  rw [foldRange]
  simp only [Nat.one_ne_zero,↓reduceDIte]
  split
  · exact range_binary f hf (start+1) stop _ (hf start b hb)
  · exact hb
termination_by stop-start
decreasing_by omega

theorem stage_binary (stage : Stage) (pid : Nat → Nat) (off : Nat)
    (b : TileDType) (hb : BinaryTag b) : BinaryTag (stageTag stage pid off b) := by
  cases stage <;> simp only [stageTag]
  all_goals try exact hb
  all_goals split
  all_goals try exact hb
  case inter.isTrue =>
    apply fold_binary _ _ _ _ hb
    intro b i hb
    split
    · exact Or.inr rfl
    · exact hb
  case intra.isTrue =>
    apply range_binary _ _ _ _ _ hb
    intro j b hb
    apply fold_binary _ _ _ _ hb
    intro b i hb
    split
    · exact Or.inl rfl
    · exact hb
  case merge.isTrue =>
    apply fold_binary _ _ _ _ hb
    intro b i hb
    split
    · exact Or.inr rfl
    · exact hb

theorem grid_binary (stage : Stage) (off : Nat) (b : TileDType) (hb : BinaryTag b) :
    BinaryTag (gridTag stage off b) :=
  fold_binary _ _ (fun tag idx h => stage_binary stage idx.toPids off tag h) b hb

theorem producer_binary (c : Host.Dimensions) (off : Nat) : BinaryTag (producerTag c off) := by
  unfold producerTag
  split
  · exact Or.inl rfl
  · split
    · exact grid_binary _ _ _ (grid_binary _ _ _ (Or.inl rfl))
    · exact grid_binary _ _ _ (grid_binary _ _ _ (grid_binary _ _ _ (Or.inl rfl)))

theorem producer_tag_eq (c : Host.Dimensions) (off : Nat) :
    producerTag c off=if producerExplicit c off then .fp32 else .real := by
  rcases producer_binary c off with h|h <;> simp [producerExplicit,h]

#specNonCircular producerExplicit avoiding [State,FPStages.program,FPStages.reference,FP.Structural.run,FP.Structural.exec]
#axiomsClean producer_tag_eq
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags

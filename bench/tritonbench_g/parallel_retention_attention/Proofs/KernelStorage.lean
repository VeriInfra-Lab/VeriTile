import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelProgramMemory
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelLaunch
import bench.tritonbench_g.parallel_retention_attention.Proofs.Runtime

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open Native
noncomputable section
set_option maxHeartbeats 4000000

/-- Physical realization of the source's masked stores. Values are read from
the completed source state; this adapter performs no retention arithmetic. -/
def writeMasked {ι : Type} (abi : LocalRuntime α σ) (address : ι → Nat)
    (value : ι → α) (active : ι → Prop) [DecidablePred active]
    (indices : List ι) (state : σ) : σ :=
  indices.foldl (fun t i=>if active i then abi.write t (address i) (value i) else t) state

theorem writeMasked_frame {ι : Type} (abi : LocalRuntime α σ) (address : ι → Nat)
    (value : ι → α) (active : ι → Prop) [DecidablePred active]
    (indices : List ι) (state : σ) (query : Nat)
    (outside : ∀ i∈indices,active i → address i≠query) :
    abi.observe (writeMasked abi address value active indices state) query=abi.observe state query := by
  induction indices generalizing state with
  | nil => rfl
  | cons i rest ih =>
    simp only [writeMasked,List.foldl_cons]
    rw [show rest.foldl _ _=writeMasked abi address value active rest _ from rfl,
      ih _ (fun j hj=>outside j (List.mem_cons_of_mem i hj))]
    split_ifs with h
    · rw [abi.write_at,if_neg (Ne.symm (outside i (by simp) h))]
    · rfl

theorem writeMasked_read {ι : Type} (abi : LocalRuntime α σ) (address : ι → Nat)
    (value : ι → α) (active : ι → Prop) [DecidablePred active]
    (indices : List ι) (state : σ) (i : ι) (member : i∈indices) (hi : active i)
    (unique : ∀ j,active j → address j=address i → j=i) :
    abi.observe (writeMasked abi address value active indices state) (address i)=value i := by
  induction indices generalizing state with
  | nil => simp at member
  | cons j rest ih =>
    change abi.observe (writeMasked abi address value active rest
      (if active j then abi.write state (address j) (value j) else state)) _=_
    by_cases later : i∈rest
    · exact ih _ later
    · have eq : i=j := by simpa [later] using member
      subst j
      rw [if_pos hi,writeMasked_frame abi address value active rest _ _ (by
        intro j hj activej collision
        exact later ((unique j activej collision) ▸ hj)),abi.write_at,if_pos rfl]

def activeTile (rows cols rowOffset colOffset : Nat) (i : TileIndex [h,w]) : Prop :=
  rowOffset+i.1.val<rows ∧ colOffset+i.2.1.val<cols

instance : DecidablePred (activeTile (h:=h) (w:=w) rows cols rowOffset colOffset) :=
  fun _=>inferInstanceAs (Decidable (_ ∧ _))

def tileFootprint (base rows cols rowOffset colOffset h w : Nat) : Set Nat :=
  {address | ∃ i : TileIndex [h,w],activeTile rows cols rowOffset colOffset i ∧
    address=tileOffset base cols rowOffset colOffset i}

def writeTile (abi : LocalRuntime α σ) (base rows cols rowOffset colOffset : Nat)
    (value : TileIndex [h,w] → α) (state : σ) : σ :=
  writeMasked abi (tileOffset base cols rowOffset colOffset) value
    (activeTile rows cols rowOffset colOffset) (TileShape.allIndices [h,w]) state

theorem writeTile_read (abi : LocalRuntime α σ) (base rows cols rowOffset colOffset : Nat)
    (value : TileIndex [h,w] → α) (state : σ) (i : TileIndex [h,w])
    (hi : activeTile rows cols rowOffset colOffset i) :
    abi.observe (writeTile abi base rows cols rowOffset colOffset value state)
      (tileOffset base cols rowOffset colOffset i)=value i := by
  apply writeMasked_read _ _ _ _ _ _ i (TileShape.mem_allIndices _ i) hi
  intro j hj eq
  have collision : (rowOffset+j.1.val)*cols+(colOffset+j.2.1.val)=
      (rowOffset+i.1.val)*cols+(colOffset+i.2.1.val) := by
    unfold tileOffset at eq
    omega
  obtain ⟨hr,hc⟩ := tile_injective cols _ _ _ _ hj.2 hi.2 collision
  apply Prod.ext
  · apply Fin.ext; omega
  · apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _

theorem writeTile_frame (abi : LocalRuntime α σ) (base rows cols rowOffset colOffset : Nat)
    (value : TileIndex [h,w] → α) (state : σ) (address : Nat)
    (outside : address∉tileFootprint base rows cols rowOffset colOffset h w) :
    abi.observe (writeTile abi base rows cols rowOffset colOffset value state) address=
      abi.observe state address := by
  apply writeMasked_frame
  intro i _ hi collision
  exact outside ⟨i,hi,collision.symm⟩

def forwardBase (s : Shape) (tile : Tile) (output : Nat) : Nat :=
  output+(tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)
def keyBase (s : Shape) (tile : Tile) (output : Nat) : Nat :=
  output+(tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)
def valueBase (s : Shape) (tile : Tile) (output : Nat) : Nat :=
  output+(tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)

def forwardFootprint (s : Shape) (tile : Tile) (output : Nat) : Set Nat :=
  tileFootprint (forwardBase s tile output) s.T s.DV (tile.chunk*128) (tile.value*s.BV) 128 s.BV
def keyFootprint (s : Shape) (tile : Tile) (output : Nat) : Set Nat :=
  tileFootprint (keyBase s tile output) s.T s.DK (tile.chunk*64) (tile.key*s.BK) 64 s.BK
def valueFootprint (s : Shape) (tile : Tile) (output : Nat) : Set Nat :=
  tileFootprint (valueBase s tile output) s.T s.DV (tile.chunk*64) (tile.value*s.BV) 64 s.BV

def forwardCommit (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (output : Nat) (final : State (Expr α)) (state : σ) : σ :=
  writeTile abi (forwardBase s tile output) s.T s.DV (tile.chunk*128) (tile.value*s.BV)
    (fun i=>evaluate a p ((final.mem "o" (forwardAddress s tile i)).read .real)) state

def keyCommit (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (region : RegionName) (output : Nat)
    (final : State (Expr α)) (state : σ) : σ :=
  writeTile abi (keyBase s tile output) s.T s.DK (tile.chunk*64) (tile.key*s.BK)
    (fun i=>evaluate a p ((final.mem region (keyAddress s tile i)).read .real)) state

def valueCommit (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (output : Nat) (final : State (Expr α)) (state : σ) : σ :=
  writeTile abi (valueBase s tile output) s.T s.DV (tile.chunk*64) (tile.value*s.BV)
    (fun i=>evaluate a p ((final.mem "dv" (valueAddress s tile i)).read .real)) state

/-- Only the ordinary separation of the three output allocations. It has no
input-value, numerical-result, successful-execution or compiler premise. -/
structure BackwardSeparate (s : Shape) (tile : Tile) (dq dk dv : Nat) : Prop where
  queryKey : Disjoint (keyFootprint s tile dq) (keyFootprint s tile dk)
  queryValue : Disjoint (keyFootprint s tile dq) (valueFootprint s tile dv)
  keyValue : Disjoint (keyFootprint s tile dk) (valueFootprint s tile dv)

def forwardRun (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (output : Nat) (state : σ) : Option σ := do
  let final ← FP.Structural.exec (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.withDot algebra dots)
    (FPStages.forwardKernel (config s 128)) (initial s b (abi.observe state) tile)
  pure (forwardCommit abi a p s tile output final state)

def backwardRun (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (dq dk dv : Nat) (state : σ) : Option σ := do
  let final ← FP.Structural.exec (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.withDot algebra dots)
    (FPBackward.source (decodedBackwardConfig s (initial s b (abi.observe state) tile).pids) "dq")
      (initial s b (abi.observe state) tile)
  pure (valueCommit abi a p s tile dv final
    (keyCommit abi a p s tile "dk" dk final (keyCommit abi a p s tile "dq" dq final state)))

#axiomsClean writeMasked_read
#axiomsClean writeTile_read
#axiomsClean writeTile_frame
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel

import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.FPBackwardWhole
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScalarProjection

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Exact source masked-store footprint, including its scratch-shard base. -/
structure Rectangle where
  region : RegionName
  base : Nat
  width : Nat
  height : Nat
  row : Nat
  col : Nat

def Rectangle.address (p : Rectangle) (i : TileIndex [m,n]) : Nat :=
  p.base+(p.row+i.1.val)*p.width+(p.col+i.2.1.val)
def Rectangle.active (p : Rectangle) (i : TileIndex [m,n]) : Prop :=
  p.row+i.1.val<p.height ∧ p.col+i.2.1.val<p.width
instance (p : Rectangle) (i : TileIndex [m,n]) : Decidable (p.active i) := by
  unfold Rectangle.active; infer_instance

def Rectangle.writes (p : Rectangle) (m n : Nat) (R : RegionName) (off : Nat) : Prop :=
  R=p.region ∧ ∃ i : TileIndex [m,n],p.active i ∧ off=p.address i

def rectangleStore (p : Rectangle) (s : State α) (values : TileIndex [m,n] → α) : State α :=
  (TileShape.allIndices [m,n]).foldl (fun t i =>
    if p.active i then t.write p.region (p.address i) (.mk .real (values i)) else t) s

/-- Active cells have unique contiguous addresses, even when the tile's
padded width is larger than the Tensor width. -/
theorem Rectangle.unique (p : Rectangle) (i j : TileIndex [m,n])
    (hi : p.active i) (hj : p.active j) (same : p.address i=p.address j) : i=j := by
  have hi' := hi.2
  have hj' := hj.2
  have positive : 0<p.width := by omega
  have addr : (p.row+i.1.val)*p.width+(p.col+i.2.1.val)=
      (p.row+j.1.val)*p.width+(p.col+j.2.1.val) := by
    simpa only [Rectangle.address,Nat.add_assoc,Nat.add_left_cancel_iff] using same
  have row : p.row+i.1.val=p.row+j.1.val := by
    have hdiv := congrArg (fun x => x/p.width) addr
    have divRule (a b : Nat) (hb : b<p.width) : (a*p.width+b)/p.width=a := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hb,Nat.zero_add]
    simpa only [divRule _ _ hi',divRule _ _ hj'] using hdiv
  have col : i.2.1.val=j.2.1.val := by rw [row] at addr; omega
  have r : i.1=j.1 := Fin.ext (by omega)
  have c : i.2.1=j.2.1 := Fin.ext col
  rcases i with ⟨a,b,⟨⟩⟩
  rcases j with ⟨d,e,⟨⟩⟩
  cases r; cases c; rfl

theorem rectangleStore_frame (p : Rectangle) (s : State α) (values : TileIndex [m,n] → α)
    (R : RegionName) (off : Nat) (outside : ¬p.writes m n R off) :
    (rectangleStore p s values).mem R off=s.mem R off := by
  unfold rectangleStore
  apply State.masked_scatter_frame
  by_cases h : R=p.region
  · right
    intro i _ hi heq
    exact outside ⟨h,i,hi,heq⟩
  · exact Or.inl h

theorem rectangleStore_read (p : Rectangle) (s : State α) (values : TileIndex [m,n] → α)
    (i : TileIndex [m,n]) (active : p.active i) :
    (rectangleStore p s values).mem p.region (p.address i)=.mk .real (values i) := by
  unfold rectangleStore
  rw [State.masked_scatter_eq_filter]
  let l := (TileShape.allIndices [m,n]).filter (fun j => decide (p.active j))
  have hi : i∈l := List.mem_filter.mpr ⟨TileShape.mem_allIndices _ i,by simpa using active⟩
  have hn : l.Nodup := (TileShape.allIndices_nodup _).filter _
  obtain ⟨before,after,hl⟩ := List.append_of_mem hi
  have ha : ∀ j∈after,p.active j := by
    intro j hj
    have hmem : j∈l := by rw [hl]; simp [hj]
    exact of_decide_eq_true (List.mem_filter.mp hmem).2
  rw [hl,List.nodup_append,List.nodup_cons] at hn
  change (l.foldl _ s).mem _ _ = _
  rw [hl,List.foldl_append,List.foldl_cons,State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    exact hn.2.1.1 ((p.unique i j active (ha j hj) heq) ▸ hj)

/-- Scalar source graph values at the actual output addresses. -/
def outputRectangle (shape : Shape) (tile : Tile) (L : Nat) : Rectangle :=
  ⟨"o",(tile.bh+shape.B*shape.H*tile.key)*(shape.T*shape.V),shape.V,shape.T,
    tile.chunk*L,tile.value*shape.BV⟩
def queryRectangle (shape : Shape) (tile : Tile) : Rectangle :=
  ⟨"dq",(tile.bh+shape.B*shape.H*tile.value)*(shape.T*shape.K),shape.K,shape.T,
    tile.chunk*64,tile.key*shape.BK⟩
def keyRectangle (shape : Shape) (tile : Tile) : Rectangle :=
  { queryRectangle shape tile with region := "dk" }
def valueRectangle (shape : Shape) (tile : Tile) : Rectangle :=
  { outputRectangle shape tile 64 with region := "dv" }

#axiomsClean Rectangle.unique
#axiomsClean rectangleStore_frame
#axiomsClean rectangleStore_read
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel

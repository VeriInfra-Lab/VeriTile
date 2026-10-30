import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Logical

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.ProducerGeometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Shape where
  heads : Nat
  T : Nat
  K : Nat
  BT : Nat
  NC : Nat
  BK : Nat
  nc_pos : 0 < NC
  bt_eq : BT=NC*16
  bk_pos : 0 < BK

theorem bt_pos (c : Shape) : 0<c.BT := by rw [c.bt_eq]; exact Nat.mul_pos c.nc_pos (by decide)

def layout (c : Shape) : RealInterMath.Layout :=
  ⟨c.T*c.K,c.K,c.T,c.K,c.BT,16,c.BK,c.NC⟩
def regions : RealInterProgram.Regions := ⟨"q","k","g","attention"⟩
def grid (c : Shape) : Grid := ⟨[Geometry.ceilDiv c.T c.BT,c.NC*c.NC,c.heads]⟩
def launchIndex (c : Shape) (q : GridIndex (grid c)) : RealInterMath.Launch :=
  ⟨(q ⟨0,by norm_num [grid,Grid.rank]⟩).val,
    (q ⟨1,by norm_num [grid,Grid.rank]⟩).val,(q ⟨2,by norm_num [grid,Grid.rank]⟩).val⟩
def index (c : Shape) (t : Fin (Geometry.ceilDiv c.T c.BT)) (pair : Fin (c.NC*c.NC))
    (h : Fin c.heads) : GridIndex (grid c) := by
  rintro ⟨n,hn⟩
  cases n with
  | zero => exact t
  | succ n =>
    cases n with
    | zero => exact pair
    | succ n =>
      cases n with
      | zero => exact h
      | succ n => simp [grid,Grid.rank] at hn; omega

theorem launch_grid (c : Shape) (s : BlockState) (q : GridIndex (grid c)) :
    RealInterProgram.launch (s.withGridIndex q)=launchIndex c q := by
  simp [RealInterProgram.launch,launchIndex,BlockState.withGridIndex,GridIndex.toPids,grid,Grid.rank]

theorem launchIndex_injective (c : Shape) : Function.Injective (launchIndex c) := by
  intro x y he
  have ht := congrArg RealInterMath.Launch.chunk he
  have hp := congrArg RealInterMath.Launch.pair he
  have hh := congrArg RealInterMath.Launch.head he
  funext a
  fin_cases a
  · exact Fin.ext ht
  · exact Fin.ext hp
  · exact Fin.ext hh

theorem quotient_bound {n N w : Nat} (hn : n<N*w) (hw : 0<w) : n/w<N := by
  have hd := Nat.div_mul_le_self n w
  by_contra h
  have hm := Nat.mul_le_mul_right w (Nat.le_of_not_gt h)
  omega

theorem rowBlock_bound (c : Shape) (q : GridIndex (grid c)) :
    RealInterMath.rowBlock (layout c) (launchIndex c q)<c.NC :=
  quotient_bound (q ⟨1,by norm_num [grid,Grid.rank]⟩).isLt c.nc_pos

theorem columnBlock_bound (c : Shape) (q : GridIndex (grid c)) :
    RealInterMath.columnBlock (layout c) (launchIndex c q)<c.NC := Nat.mod_lt _ c.nc_pos

theorem local_row_bound (c : Shape) (q : GridIndex (grid c)) (i : Fin 16) :
    RealInterMath.rowBlock (layout c) (launchIndex c q)*16+i.val<c.BT := by
  rw [c.bt_eq]
  exact Geometry.packed_lt (rowBlock_bound c q) i.isLt

theorem address_injective (c : Shape) (x y : GridIndex (grid c))
    (i j : TileIndex [16,16])
    (hi : RealInterMath.active (layout c) (launchIndex c x) i.1.val i.2.1.val)
    (hj : RealInterMath.active (layout c) (launchIndex c y) j.1.val j.2.1.val)
    (he : RealInterMath.outputAddress (layout c) (launchIndex c x) i.1.val i.2.1.val =
      RealInterMath.outputAddress (layout c) (launchIndex c y) j.1.val j.2.1.val) : x=y ∧ i=j := by
  let lx := launchIndex c x
  let ly := launchIndex c y
  have he' : Geometry.attentionAddress c.T c.BT lx.head
      (RealInterMath.rowStart (layout c) lx+i.1.val) (RealInterMath.columnBlock (layout c) lx*16+i.2.1.val) =
      Geometry.attentionAddress c.T c.BT ly.head
      (RealInterMath.rowStart (layout c) ly+j.1.val) (RealInterMath.columnBlock (layout c) ly*16+j.2.1.val) := by
    simpa only [Geometry.attentionAddress,RealInterMath.outputAddress,layout,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨hh,hr,hcol⟩ := Geometry.attention_injective hi.1 hj.1 hi.2 hj.2 he'
  have hr' : lx.chunk*c.BT+(RealInterMath.rowBlock (layout c) lx*16+i.1.val)=
      ly.chunk*c.BT+(RealInterMath.rowBlock (layout c) ly*16+j.1.val) := by
    simpa only [RealInterMath.rowStart,layout,Nat.add_assoc] using hr
  obtain ⟨hchunk,hlocal⟩ := Geometry.packed_injective (local_row_bound c x i.1) (local_row_bound c y j.1) hr'
  obtain ⟨hrow,hri⟩ := Geometry.packed_injective i.1.isLt j.1.isLt hlocal
  obtain ⟨hcolumn,hcj⟩ := Geometry.packed_injective i.2.1.isLt j.2.1.isLt hcol
  have hpair : lx.pair=ly.pair := by
    have hr : lx.pair/c.NC=ly.pair/c.NC := hrow
    have hc : lx.pair%c.NC=ly.pair%c.NC := hcolumn
    calc
      lx.pair = lx.pair/c.NC*c.NC+lx.pair%c.NC := (Nat.div_add_mod' lx.pair c.NC).symm
      _ = ly.pair/c.NC*c.NC+ly.pair%c.NC := by rw [hr,hc]
      _ = ly.pair := Nat.div_add_mod' ly.pair c.NC
  refine ⟨launchIndex_injective c ?_,Prod.ext (Fin.ext hri) (Prod.ext (Fin.ext hcj) (Subsingleton.elim _ _))⟩
  cases hx : launchIndex c x
  cases hy : launchIndex c y
  dsimp [lx,ly] at hh hchunk hpair
  simp_all

def owner (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) : GridIndex (grid c) :=
  index c ⟨t.val/c.BT,Geometry.owner_lt (bt_pos c) t.isLt⟩
    ⟨(t.val%c.BT/16)*c.NC+j.val/16,Geometry.packed_lt
      (quotient_bound (by simpa [c.bt_eq] using Nat.mod_lt t.val (bt_pos c)) (by decide))
      (quotient_bound (by simpa [c.bt_eq] using j.isLt) (by decide))⟩ h

def lane (c : Shape) (t : Fin c.T) (j : Fin c.BT) : TileIndex [16,16] :=
  (⟨t.val%c.BT%16,Nat.mod_lt _ (by decide)⟩,⟨j.val%16,Nat.mod_lt _ (by decide)⟩,⟨⟩)

@[simp] theorem owner_launch (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    launchIndex c (owner c h t j)=⟨t.val/c.BT,(t.val%c.BT/16)*c.NC+j.val/16,h.val⟩ := rfl

theorem owner_row (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    RealInterMath.rowStart (layout c) (launchIndex c (owner c h t j))+(lane c t j).1.val=t.val := by
  have hj : j.val/16<c.NC := quotient_bound (by simpa [c.bt_eq] using j.isLt) (by decide)
  simp only [owner_launch,RealInterMath.rowStart,RealInterMath.rowBlock,layout,lane]
  have hpair : ((t.val%c.BT/16)*c.NC+j.val/16)/c.NC=t.val%c.BT/16 := by
    rw [Nat.mul_comm _ c.NC,Nat.mul_add_div c.nc_pos,Nat.div_eq_of_lt hj,Nat.add_zero]
  rw [hpair]
  have h1 := Nat.div_add_mod' (t.val%c.BT) 16
  have h2 := Nat.div_add_mod' t.val c.BT
  omega

theorem owner_column (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    RealInterMath.columnBlock (layout c) (launchIndex c (owner c h t j))*16+(lane c t j).2.1.val=j.val := by
  have hj : j.val/16<c.NC := quotient_bound (by simpa [c.bt_eq] using j.isLt) (by decide)
  simp only [owner_launch,RealInterMath.columnBlock,layout,lane,Nat.add_mod,Nat.mul_mod_left,zero_add,
    Nat.mod_eq_of_lt hj,Nat.div_add_mod']

theorem owner_active (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    RealInterMath.active (layout c) (launchIndex c (owner c h t j)) (lane c t j).1.val (lane c t j).2.1.val := by
  change _<c.T ∧ _<c.BT
  rw [owner_row]
  simp only [show (layout c).BC=16 from rfl]
  rw [owner_column]
  exact ⟨t.isLt,j.isLt⟩

theorem owner_address (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    RealInterMath.outputAddress (layout c) (launchIndex c (owner c h t j))
      (lane c t j).1.val (lane c t j).2.1.val=h.val*c.T*c.BT+t.val*c.BT+j.val := by
  unfold RealInterMath.outputAddress
  rw [owner_row]
  change h.val*c.T*c.BT+t.val*c.BT+_= _
  simp only [show (layout c).BC=16 from rfl]
  rw [owner_column]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.ProducerGeometry

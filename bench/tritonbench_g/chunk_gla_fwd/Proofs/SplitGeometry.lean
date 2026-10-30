import bench.tritonbench_g.chunk_gla_fwd.Proofs.DiagonalGeometry
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SplitGeometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos quotient_bound)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def grid (c : Shape) : Grid := ⟨[Geometry.ceilDiv c.K c.BK,Geometry.ceilDiv c.T c.BT*c.NC,c.heads]⟩
def launchIndex (c : Shape) (q : GridIndex (grid c)) : RealSplitMath.Launch :=
  ⟨(q ⟨1,by norm_num [grid,Grid.rank]⟩).val/c.NC,
    (q ⟨1,by norm_num [grid,Grid.rank]⟩).val%c.NC,
    (q ⟨2,by norm_num [grid,Grid.rank]⟩).val,
    (q ⟨0,by norm_num [grid,Grid.rank]⟩).val,c.heads⟩
def index (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK))
    (tc : Fin (Geometry.ceilDiv c.T c.BT*c.NC)) (h : Fin c.heads) : GridIndex (grid c) := by
  rintro ⟨n,hn⟩
  cases n with
  | zero => exact f
  | succ n =>
    cases n with
    | zero => exact tc
    | succ n =>
      cases n with
      | zero => exact h
      | succ n => simp [grid,Grid.rank] at hn; omega

@[simp] theorem launch_index (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK))
    (tc : Fin (Geometry.ceilDiv c.T c.BT*c.NC)) (h : Fin c.heads) :
    launchIndex c (index c f tc h)=⟨tc.val/c.NC,tc.val%c.NC,h.val,f.val,c.heads⟩ := rfl

def row (c : Shape) (l : RealSplitMath.Launch) (i : Nat) := l.chunk*c.BT+l.block*16+i
def address (c : Shape) (l : RealSplitMath.Launch) (i : TileIndex [16,16]) :=
  ((l.feature*c.heads+l.head)*c.T+row c l i.1.val)*16+i.2.1.val
def active (c : Shape) (l : RealSplitMath.Launch) (i : TileIndex [16,16]) : Prop :=
  row c l i.1.val<c.T

theorem launchIndex_injective (c : Shape) : Function.Injective (launchIndex c) := by
  intro x y he
  have ht := congrArg RealSplitMath.Launch.chunk he
  have hb := congrArg RealSplitMath.Launch.block he
  have hh := congrArg RealSplitMath.Launch.head he
  have hf := congrArg RealSplitMath.Launch.feature he
  have hx := Nat.div_add_mod' (x ⟨1,by norm_num [grid,Grid.rank]⟩).val c.NC
  have hy := Nat.div_add_mod' (y ⟨1,by norm_num [grid,Grid.rank]⟩).val c.NC
  dsimp [launchIndex] at ht hb
  rw [ht,hb] at hx
  have htc : (x ⟨1,by norm_num [grid,Grid.rank]⟩).val=(y ⟨1,by norm_num [grid,Grid.rank]⟩).val := by omega
  funext a
  fin_cases a
  · exact Fin.ext hf
  · exact Fin.ext htc
  · exact Fin.ext hh

theorem local_bound (c : Shape) (q : GridIndex (grid c)) (i : Fin 16) :
    (launchIndex c q).block*16+i.val<c.BT := by
  rw [c.bt_eq]
  exact Geometry.packed_lt (Nat.mod_lt _ c.nc_pos) i.isLt

theorem address_injective (c : Shape) (x y : GridIndex (grid c))
    (i j : TileIndex [16,16]) (hi : active c (launchIndex c x) i) (hj : active c (launchIndex c y) j)
    (he : address c (launchIndex c x) i=address c (launchIndex c y) j) : x=y ∧ i=j := by
  obtain ⟨hr,hij⟩ := Geometry.packed_injective i.2.1.isLt j.2.1.isLt he
  obtain ⟨hhf,hrow⟩ := Geometry.packed_injective hi hj hr
  obtain ⟨hf,hh⟩ := Geometry.packed_injective
    (x ⟨2,by norm_num [grid,Grid.rank]⟩).isLt
    (y ⟨2,by norm_num [grid,Grid.rank]⟩).isLt hhf
  have hrow' : (launchIndex c x).chunk*c.BT+((launchIndex c x).block*16+i.1.val)=
      (launchIndex c y).chunk*c.BT+((launchIndex c y).block*16+j.1.val) := by
    simpa only [row,Nat.add_assoc] using hrow
  obtain ⟨ht,hl⟩ := Geometry.packed_injective (local_bound c x i.1) (local_bound c y j.1) hrow'
  obtain ⟨hb,hir⟩ := Geometry.packed_injective i.1.isLt j.1.isLt hl
  refine ⟨launchIndex_injective c ?_,Prod.ext (Fin.ext hir) (Prod.ext (Fin.ext hij) (Subsingleton.elim _ _))⟩
  have hn : (launchIndex c x).heads=(launchIndex c y).heads := rfl
  cases hx : launchIndex c x
  cases hy : launchIndex c y
  simp_all

def owner (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) :
    GridIndex (grid c) :=
  index c f ⟨t.val/c.BT*c.NC+t.val%c.BT/16,
    Geometry.packed_lt (Geometry.owner_lt (bt_pos c) t.isLt)
      (quotient_bound (by simpa [c.bt_eq] using Nat.mod_lt t.val (bt_pos c)) (by decide))⟩ h

def lane (t : Fin n) (j : Fin 16) : TileIndex [16,16] :=
  (⟨t.val%16,Nat.mod_lt _ (by decide)⟩,j,⟨⟩)

@[simp] theorem owner_launch (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) :
    launchIndex c (owner c f h t)=⟨t.val/c.BT,t.val%c.BT/16,h.val,f.val,c.heads⟩ := by
  have hb : t.val%c.BT/16<c.NC :=
    quotient_bound (by simpa [c.bt_eq] using Nat.mod_lt t.val (bt_pos c)) (by decide)
  unfold owner
  rw [launch_index]
  simp only [Nat.mul_comm (t.val/c.BT) c.NC,Nat.mul_add_div c.nc_pos,Nat.div_eq_of_lt hb,Nat.add_zero,
    Nat.add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt hb]

def ownerLane (c : Shape) (t : Fin c.T) (j : Fin 16) : TileIndex [16,16] :=
  (⟨t.val%c.BT%16,Nat.mod_lt _ (by decide)⟩,j,⟨⟩)

theorem owner_row (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16) :
    row c (launchIndex c (owner c f h t)) (ownerLane c t j).1.val=t.val := by
  simp only [owner_launch,row,ownerLane]
  have h1 := Nat.div_add_mod' (t.val%c.BT) 16
  have h2 := Nat.div_add_mod' t.val c.BT
  omega

theorem owner_address (c : Shape) (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16) :
    address c (launchIndex c (owner c f h t)) (ownerLane c t j)=
      ((f.val*c.heads+h.val)*c.T+t.val)*16+j.val := by
  unfold address
  rw [owner_row]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SplitGeometry

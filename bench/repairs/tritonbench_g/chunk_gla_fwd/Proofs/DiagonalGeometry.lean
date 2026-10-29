import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.ProducerGeometry
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.DiagonalGeometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos quotient_bound)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def grid (c : Shape) : Grid := ⟨[Geometry.ceilDiv c.T c.BT,c.NC,c.heads]⟩
def launchIndex (c : Shape) (q : GridIndex (grid c)) : RealIntraMath.Launch :=
  ⟨(q ⟨0,by norm_num [grid,Grid.rank]⟩).val,
    (q ⟨1,by norm_num [grid,Grid.rank]⟩).val,(q ⟨2,by norm_num [grid,Grid.rank]⟩).val⟩
def index (c : Shape) (t : Fin (Geometry.ceilDiv c.T c.BT)) (b : Fin c.NC)
    (h : Fin c.heads) : GridIndex (grid c) := by
  rintro ⟨n,hn⟩
  cases n with
  | zero => exact t
  | succ n =>
    cases n with
    | zero => exact b
    | succ n =>
      cases n with
      | zero => exact h
      | succ n => simp [grid,Grid.rank] at hn; omega

def row (c : Shape) (l : RealIntraMath.Launch) (i : Nat) := l.chunk*c.BT+l.block*16+i
def column (l : RealIntraMath.Launch) (j : Nat) := l.block*16+j
def address (c : Shape) (l : RealIntraMath.Launch) (i : TileIndex [16,16]) :=
  l.head*c.T*c.BT+row c l i.1.val*c.BT+column l i.2.1.val
def active (c : Shape) (l : RealIntraMath.Launch) (i : TileIndex [16,16]) : Prop :=
  row c l i.1.val<c.T ∧ column l i.2.1.val<c.BT

theorem launchIndex_injective (c : Shape) : Function.Injective (launchIndex c) := by
  intro x y he
  have ht := congrArg RealIntraMath.Launch.chunk he
  have hb := congrArg RealIntraMath.Launch.block he
  have hh := congrArg RealIntraMath.Launch.head he
  funext a
  fin_cases a
  · exact Fin.ext ht
  · exact Fin.ext hb
  · exact Fin.ext hh

theorem local_bound (c : Shape) (q : GridIndex (grid c)) (i : Fin 16) :
    (launchIndex c q).block*16+i.val<c.BT := by
  rw [c.bt_eq]
  exact Geometry.packed_lt (q ⟨1,by norm_num [grid,Grid.rank]⟩).isLt i.isLt

theorem address_injective (c : Shape) (x y : GridIndex (grid c))
    (i j : TileIndex [16,16]) (hi : active c (launchIndex c x) i) (hj : active c (launchIndex c y) j)
    (he : address c (launchIndex c x) i=address c (launchIndex c y) j) : x=y ∧ i=j := by
  have he' : Geometry.attentionAddress c.T c.BT (launchIndex c x).head (row c (launchIndex c x) i.1.val)
      (column (launchIndex c x) i.2.1.val) = Geometry.attentionAddress c.T c.BT (launchIndex c y).head
      (row c (launchIndex c y) j.1.val) (column (launchIndex c y) j.2.1.val) := by
    simpa only [address,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨hh,hr,hcol⟩ := Geometry.attention_injective hi.1 hj.1 hi.2 hj.2 he'
  have hr' : (launchIndex c x).chunk*c.BT+((launchIndex c x).block*16+i.1.val)=
      (launchIndex c y).chunk*c.BT+((launchIndex c y).block*16+j.1.val) := by
    simpa only [row,Nat.add_assoc] using hr
  obtain ⟨ht,hl⟩ := Geometry.packed_injective (local_bound c x i.1) (local_bound c y j.1) hr'
  obtain ⟨hb,hir⟩ := Geometry.packed_injective i.1.isLt j.1.isLt hl
  obtain ⟨_,hij⟩ := Geometry.packed_injective i.2.1.isLt j.2.1.isLt hcol
  refine ⟨launchIndex_injective c ?_,Prod.ext (Fin.ext hir) (Prod.ext (Fin.ext hij) (Subsingleton.elim _ _))⟩
  cases hx : launchIndex c x
  cases hy : launchIndex c y
  simp_all

def owner (c : Shape) (h : Fin c.heads) (t : Fin c.T) : GridIndex (grid c) :=
  index c ⟨t.val/c.BT,Geometry.owner_lt (bt_pos c) t.isLt⟩
    ⟨t.val%c.BT/16,quotient_bound (by simpa [c.bt_eq] using Nat.mod_lt t.val (bt_pos c)) (by decide)⟩ h

def lane (c : Shape) (t : Fin c.T) (j : Fin c.BT) : TileIndex [16,16] :=
  (⟨t.val%c.BT%16,Nat.mod_lt _ (by decide)⟩,⟨j.val%16,Nat.mod_lt _ (by decide)⟩,⟨⟩)

@[simp] theorem owner_launch (c : Shape) (h : Fin c.heads) (t : Fin c.T) :
    launchIndex c (owner c h t)=⟨t.val/c.BT,t.val%c.BT/16,h.val⟩ := rfl

theorem owner_row (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    row c (launchIndex c (owner c h t)) (lane c t j).1.val=t.val := by
  simp only [owner_launch,row,lane]
  have h1 := Nat.div_add_mod' (t.val%c.BT) 16
  have h2 := Nat.div_add_mod' t.val c.BT
  omega

theorem owner_column (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) :
    column (launchIndex c (owner c h t)) (lane c t j).2.1.val=j.val := by
  simp only [owner_launch,column,lane]
  rw [← hsame]
  exact Nat.div_add_mod' j.val 16

theorem owner_active (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) : active c (launchIndex c (owner c h t)) (lane c t j) := by
  unfold active
  rw [owner_row,owner_column c h t j hsame]
  exact ⟨t.isLt,j.isLt⟩

theorem owner_address (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) :
    address c (launchIndex c (owner c h t)) (lane c t j)=h.val*c.T*c.BT+t.val*c.BT+j.val := by
  unfold address
  rw [owner_row,owner_column c h t j hsame]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.DiagonalGeometry

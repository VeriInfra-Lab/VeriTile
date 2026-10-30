import bench.tritonbench_g.int4_matmul.Kernels
import bench.tritonbench_g.int4_matmul.Proofs.GroupedGrid
namespace VeriTile.Bench.Repairs.Int4Matmul
set_option maxHeartbeats 4000000

theorem split_positive (t : Tile) (h : t.Valid) : 0<t.splits := by rcases h.splits with h|h <;> omega

theorem ceil_cover (a b : Nat) (hb : 0<b) : a≤ceil a b*b := by
  have hd := Nat.div_add_mod (a+b-1) b
  have hr := Nat.mod_lt (a+b-1) hb
  unfold ceil
  rw [Nat.mul_comm b] at hd
  omega

theorem ceil_positive (a b : Nat) (ha : 0<a) (hb : 0<b) : 0<ceil a b := by
  have h := ceil_cover a b hb
  by_contra hz
  have he : ceil a b=0 := by omega
  rw [he,Nat.zero_mul] at h
  omega

theorem div_coordinate (a b c : Nat) (hb : 0<b) (hc : c<b) : (a*b+c)/b=a := by
  apply Nat.div_eq_of_lt_le <;> nlinarith

structure Lane where
  pid : Nat
  split : Nat
  i : Nat
  j : Nat
  deriving DecidableEq, Repr

def laneWork (s : Shape) (t : Tile) (l : Lane) : Work :=
  ⟨GroupedGrid.row (blocksM s t) (blocksN s t) t.gm l.pid*t.bm+l.i,
   GroupedGrid.col (blocksM s t) (blocksN s t) t.gm l.pid*t.bn+l.j,l.split⟩
def Lane.Valid (s : Shape) (t : Tile) (l : Lane) :=
  l.pid<blocksM s t*blocksN s t ∧ l.split<t.splits ∧ l.i<t.bm ∧ l.j<t.bn

theorem lane_coverage (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (w : Work) (hw : s.Active t w) :
    ∃ l,l.Valid s t ∧ laneWork s t l=w := by
  have hm := ceil_cover s.m t.bm ht.bmpos
  have hn := ceil_cover s.n t.bn ht.bnpos
  have hr : w.row/t.bm<blocksM s t := (Nat.div_lt_iff_lt_mul ht.bmpos).mpr (by simpa [blocksM] using lt_of_lt_of_le hw.1 hm)
  have hc : w.col/t.bn<blocksN s t := (Nat.div_lt_iff_lt_mul ht.bnpos).mpr (by simpa [blocksN] using lt_of_lt_of_le hw.2.1 hn)
  obtain ⟨pid,hpid,hrow,hcol⟩ := GroupedGrid.covers (blocksM s t) (blocksN s t) t.gm
    (ceil_positive _ _ hs.npos ht.bnpos) ht.gmpos _ _ hr hc
  refine ⟨⟨pid,w.split,w.row%t.bm,w.col%t.bn⟩,⟨hpid,hw.2.2,Nat.mod_lt _ ht.bmpos,Nat.mod_lt _ ht.bnpos⟩,?_⟩
  cases w <;> simp_all [laneWork,Nat.div_add_mod']

theorem lane_injective (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (a b : Lane) (ha : a.Valid s t) (hb : b.Valid s t)
    (he : laneWork s t a=laneWork s t b) : a=b := by
  have hr := congrArg Work.row he
  have hc := congrArg Work.col he
  have hsp := congrArg Work.split he
  dsimp [laneWork] at hr hc hsp
  have rows := congrArg (fun n => n/t.bm) hr
  have cols := congrArg (fun n => n/t.bn) hc
  simp only [div_coordinate _ _ _ ht.bmpos ha.2.2.1,div_coordinate _ _ _ ht.bmpos hb.2.2.1] at rows
  simp only [div_coordinate _ _ _ ht.bnpos ha.2.2.2,div_coordinate _ _ _ ht.bnpos hb.2.2.2] at cols
  have hp := GroupedGrid.injective (blocksM s t) (blocksN s t) t.gm
    (ceil_positive _ _ hs.npos ht.bnpos) ht.gmpos a.pid b.pid ha.1 hb.1 rows cols
  have hi : a.i=b.i := by rw [hp] at hr; omega
  have hj : a.j=b.j := by rw [hp] at hc; omega
  cases a;cases b;simp_all

theorem address_injective (s : Shape) (a b : Work)
    (ha : a.col<s.n) (hb : b.col<s.n) (h : cAddress s a.row a.col=cAddress s b.row b.col) :
    a.row=b.row ∧ a.col=b.col := by
  unfold cAddress at h
  have hn : 0<s.n := by omega
  have hd := congrArg (fun k => k/s.n) h
  have hm := congrArg (fun k => k%s.n) h
  simp [div_coordinate _ _ _ hn ha,div_coordinate _ _ _ hn hb] at hd
  simp [Nat.add_mod,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] at hm
  exact ⟨hd,hm⟩

theorem kIndex_eq (t : Tile) (sp b e : Nat) : kIndex t sp b e=(b*t.splits+sp)*t.bk+e := by
  unfold kIndex;ring

theorem k_coverage (s : Shape) (t : Tile) (ht : t.Valid) (k : Nat) (hk : k<s.k) :
    ∃ sp b e,sp<t.splits ∧ b<rounds s t ∧ e<t.bk ∧ kIndex t sp b e=k := by
  have hsp := split_positive t ht
  have hcover := ceil_cover s.k (t.bk*t.splits) (Nat.mul_pos ht.bkpos hsp)
  have hb : k/(t.bk*t.splits)<rounds s t :=
    (Nat.div_lt_iff_lt_mul (Nat.mul_pos ht.bkpos hsp)).mpr (by simpa [rounds] using lt_of_lt_of_le hk hcover)
  refine ⟨k/t.bk%t.splits,k/t.bk/t.splits,k%t.bk,Nat.mod_lt _ hsp,?_,Nat.mod_lt _ ht.bkpos,?_⟩
  · simpa [Nat.div_div_eq_div_mul] using hb
  · rw [kIndex_eq,Nat.div_add_mod',Nat.div_add_mod']

theorem k_unique (t : Tile) (ht : t.Valid) (sp b e sp' b' e' : Nat)
    (hs : sp<t.splits) (hs' : sp'<t.splits) (he : e<t.bk) (he' : e'<t.bk)
    (h : kIndex t sp b e=kIndex t sp' b' e') : sp=sp' ∧ b=b' ∧ e=e' := by
  rw [kIndex_eq,kIndex_eq] at h
  have hd := congrArg (fun k => k/t.bk) h
  have hm := congrArg (fun k => k%t.bk) h
  simp [div_coordinate _ _ _ ht.bkpos he,div_coordinate _ _ _ ht.bkpos he'] at hd
  simp [Nat.add_mod,Nat.mod_eq_of_lt he,Nat.mod_eq_of_lt he'] at hm
  have hd' := congrArg (fun k => k/t.splits) hd
  have hm' := congrArg (fun k => k%t.splits) hd
  simp [div_coordinate _ _ _ (split_positive t ht) hs,div_coordinate _ _ _ (split_positive t ht) hs'] at hd'
  simp [Nat.add_mod,Nat.mod_eq_of_lt hs,Nat.mod_eq_of_lt hs'] at hm'
  exact ⟨hm',hd',hm⟩
#axiomsClean lane_coverage
#axiomsClean lane_injective
#axiomsClean k_coverage
#axiomsClean k_unique
end VeriTile.Bench.Repairs.Int4Matmul

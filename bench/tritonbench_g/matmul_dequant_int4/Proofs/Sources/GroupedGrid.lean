import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Nat.ModEq
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.GroupedGrid
set_option maxHeartbeats 4000000

def first (G N pid : Nat) := pid/(G*N)*G
def size (M N G pid : Nat) := min (M-first G N pid) G
def row (M N G pid : Nat) := first G N pid+pid%size M N G pid
def col (M N G pid : Nat) := (pid%(G*N))/size M N G pid

theorem facts (M N G pid : Nat) (hN : 0<N) (hG : 0<G) (hp : pid<M*N) :
    first G N pid<M ∧ 0<size M N G pid ∧
    first G N pid+size M N G pid≤M ∧
    first G N pid+size M N G pid≤first G N pid+G ∧
    first G N pid*N+pid%(G*N)=pid ∧ pid%(G*N)<size M N G pid*N := by
  have hbase : first G N pid*N≤pid := by
    simpa [first,Nat.mul_assoc] using Nat.div_mul_le_self pid (G*N)
  have hf : first G N pid<M := Nat.lt_of_mul_lt_mul_right (lt_of_le_of_lt hbase hp)
  have hs : 0<size M N G pid := lt_min (Nat.sub_pos_of_lt hf) hG
  have hsM : first G N pid+size M N G pid≤M := by
    have h := min_le_left (M-first G N pid) G
    unfold size; omega
  have hsG : first G N pid+size M N G pid≤first G N pid+G := by
    exact Nat.add_le_add_left (min_le_right _ _) _
  have he : first G N pid*N+pid%(G*N)=pid := by
    simpa [first,Nat.mul_assoc] using Nat.div_add_mod' pid (G*N)
  refine ⟨hf,hs,hsM,hsG,he,?_⟩
  by_cases hfull : G≤M-first G N pid
  · have hz : size M N G pid=G := min_eq_right hfull
    rw [hz]
    exact Nat.mod_lt _ (Nat.mul_pos hG hN)
  · have hz : size M N G pid=M-first G N pid := min_eq_left (by omega)
    rw [hz]
    have hr : M-first G N pid+first G N pid=M := by omega
    nlinarith

theorem bounds (M N G pid : Nat) (hN : 0<N) (hG : 0<G) (hp : pid<M*N) :
    row M N G pid<M ∧ col M N G pid<N := by
  obtain ⟨hf,hs,hsM,hsG,he,hlocal⟩ := facts M N G pid hN hG hp
  have hr := Nat.mod_lt pid hs
  constructor
  · unfold row; omega
  · exact (Nat.div_lt_iff_lt_mul hs).mpr (by simpa [Nat.mul_comm] using hlocal)

theorem row_group (M N G pid : Nat) (hN : 0<N) (hG : 0<G) (hp : pid<M*N) :
    row M N G pid/G=pid/(G*N) := by
  obtain ⟨hf,hs,hsM,hsG,he,hlocal⟩ := facts M N G pid hN hG hp
  have hr := Nat.mod_lt pid hs
  apply Nat.div_eq_of_lt_le
  · change first G N pid≤row M N G pid
    unfold row; omega
  · change row M N G pid<(pid/(G*N)+1)*G
    rw [Nat.add_mul,Nat.one_mul]
    change row M N G pid<first G N pid+G
    unfold row; omega

theorem injective (M N G : Nat) (hN : 0<N) (hG : 0<G)
    (p q : Nat) (hp : p<M*N) (hq : q<M*N)
    (hr : row M N G p=row M N G q) (hc : col M N G p=col M N G q) : p=q := by
  have hg : p/(G*N)=q/(G*N) := by
    rw [← row_group M N G p hN hG hp,← row_group M N G q hN hG hq,hr]
  have hf : first G N p=first G N q := by simp only [first,hg]
  have hs : size M N G p=size M N G q := by simp only [size,hf]
  have hm : p%size M N G q=q%size M N G q := by
    unfold row at hr
    rw [hf,hs] at hr
    omega
  have ep := (facts M N G p hN hG hp).2.2.2.2.1
  have eq := (facts M N G q hN hG hq).2.2.2.2.1
  rw [hf] at ep
  have hm' : (p%(G*N))%size M N G q=(q%(G*N))%size M N G q := by
    apply Nat.ModEq.add_left_cancel' (first G N q*N)
    change (first G N q*N+p%(G*N))%size M N G q=
      (first G N q*N+q%(G*N))%size M N G q
    rw [ep,eq]
    exact hm
  have hd : (p%(G*N))/size M N G q=(q%(G*N))/size M N G q := by
    simpa only [col,hs] using hc
  have dp := Nat.div_add_mod' (p%(G*N)) (size M N G q)
  have dq := Nat.div_add_mod' (q%(G*N)) (size M N G q)
  rw [hd,hm'] at dp
  omega

def mapping (M N G : Nat) (hN : 0<N) (hG : 0<G) (p : Fin (M*N)) : Fin M × Fin N :=
  (⟨row M N G p, (bounds M N G p hN hG p.isLt).1⟩,
   ⟨col M N G p, (bounds M N G p hN hG p.isLt).2⟩)

theorem bijective (M N G : Nat) (hN : 0<N) (hG : 0<G) :
    Function.Bijective (mapping M N G hN hG) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  constructor
  · intro p q h
    apply Fin.ext
    apply injective M N G hN hG p q p.isLt q.isLt
    · exact congrArg (fun x : Fin M × Fin N => x.1.val) h
    · exact congrArg (fun x : Fin M × Fin N => x.2.val) h
  · simp

theorem covers (M N G : Nat) (hN : 0<N) (hG : 0<G)
    (r c : Nat) (hr : r<M) (hc : c<N) :
    ∃ pid, pid<M*N ∧ row M N G pid=r ∧ col M N G pid=c := by
  obtain ⟨p,h⟩ := (bijective M N G hN hG).2 (⟨r,hr⟩,⟨c,hc⟩)
  exact ⟨p,p.isLt,congrArg (fun x : Fin M × Fin N => x.1.val) h,
    congrArg (fun x : Fin M × Fin N => x.2.val) h⟩

#axiomsClean facts
#axiomsClean bounds
#axiomsClean row_group
#axiomsClean injective
#axiomsClean bijective
#axiomsClean covers
end VeriTile.Bench.Repairs.MatmulDequantInt4.GroupedGrid

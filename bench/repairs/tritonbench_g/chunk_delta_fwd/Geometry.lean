import bench.repairs.tritonbench_g.chunk_delta_fwd.Model
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Geometry
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Geometry
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv owner_lt packed_lt packed_injective)
set_option maxHeartbeats 6000000

structure ShapeValid (p : Parameters) : Prop where
  bt_pos : 0 < p.BT
  bc_pos : 0 < p.BC
  bv_pos : 0 < p.BV
  k_cover : p.K ≤ p.BK
  chunks : p.NT=ceilDiv p.T p.BT

def ValidPID (p : Parameters) (pid : PID) : Prop :=
  pid.1 < ceilDiv p.V p.BV ∧ pid.2 < p.B*p.H

theorem history_address (p : Parameters) (pid : PID) (c : Nat) (i : HIndex p) :
    hOffset p pid c i=((pid.2*p.NT+c)*p.K+i.1.val)*p.V+column p pid i.2.1.val := by
  unfold hOffset
  ring
theorem corrected_address (p : Parameters) (pid : PID) (c b : Nat) (i : CIndex p) :
    cOffset p pid c b i=(pid.2*p.T+time p c b i.1.val)*p.V+column p pid i.2.1.val := by
  unfold cOffset
  ring
theorem final_address (p : Parameters) (pid : PID) (i : HIndex p) :
    fOffset p pid i=(pid.2*p.K+i.1.val)*p.V+column p pid i.2.1.val := by
  unfold fOffset
  ring

theorem history_collision (p : Parameters) (a b : PID) (c d : Nat)
    (i j : HIndex p) (hi : hActive p a i) (hj : hActive p b j)
    (hc : c < p.NT) (hd : d < p.NT) (he : hOffset p a c i=hOffset p b d j) :
    a=b ∧ c=d ∧ i=j := by
  rw [history_address,history_address] at he
  obtain ⟨he,hcol⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨he,hkey⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hhead,hchunk⟩ := packed_injective hc hd he
  obtain ⟨htile,hlane⟩ := packed_injective i.2.1.isLt j.2.1.isLt hcol
  exact ⟨Prod.ext htile hhead,hchunk,Prod.ext (Fin.ext hkey)
    (Prod.ext (Fin.ext hlane) (Subsingleton.elim _ _))⟩

theorem corrected_collision (p : Parameters) (a b : PID) (c d k l : Nat)
    (i j : CIndex p) (hi : cActive p a c k i) (hj : cActive p b d l j)
    (he : cOffset p a c k i=cOffset p b d l j) :
    a=b ∧ time p c k i.1.val=time p d l j.1.val ∧ i.2.1=j.2.1 := by
  rw [corrected_address,corrected_address] at he
  obtain ⟨he,hcol⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨hhead,hrow⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨htile,hlane⟩ := packed_injective i.2.1.isLt j.2.1.isLt hcol
  exact ⟨Prod.ext htile hhead,hrow,Fin.ext hlane⟩

theorem final_collision (p : Parameters) (a b : PID) (i j : HIndex p)
    (hi : hActive p a i) (hj : hActive p b j) (he : fOffset p a i=fOffset p b j) :
    a=b ∧ i=j := by
  rw [final_address,final_address] at he
  obtain ⟨he,hcol⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨hhead,hkey⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨htile,hlane⟩ := packed_injective i.2.1.isLt j.2.1.isLt hcol
  exact ⟨Prod.ext htile hhead,Prod.ext (Fin.ext hkey)
    (Prod.ext (Fin.ext hlane) (Subsingleton.elim _ _))⟩

theorem inner_collision (p : Parameters) (pid : PID) (c b d : Nat)
    (i j : CIndex p) (hi : cActive p pid c b i) (hj : cActive p pid c d j)
    (he : cOffset p pid c b i=cOffset p pid c d j) : b=d ∧ i=j := by
  obtain ⟨_,hr,hcol⟩ := corrected_collision p pid pid c c b d i j hi hj he
  have hr' : b*p.BC+i.1.val=d*p.BC+j.1.val := by
    dsimp [time] at hr
    omega
  obtain ⟨hb,hl⟩ := packed_injective i.1.isLt j.1.isLt hr'
  exact ⟨hb,Prod.ext (Fin.ext hl) (Prod.ext hcol (Subsingleton.elim _ _))⟩

theorem history_bound (p : Parameters) (pid : PID) (c : Nat) (i : HIndex p)
    (hp : ValidPID p pid) (hc : c < p.NT) (hi : hActive p pid i) :
    hOffset p pid c i < p.B*p.H*p.NT*p.K*p.V := by
  rw [history_address]
  exact packed_lt (packed_lt (packed_lt hp.2 hc) hi.1) hi.2
theorem corrected_bound (p : Parameters) (pid : PID) (c b : Nat) (i : CIndex p)
    (hp : ValidPID p pid) (hi : cActive p pid c b i) :
    cOffset p pid c b i < p.B*p.H*p.T*p.V := by
  rw [corrected_address]
  exact packed_lt (packed_lt hp.2 hi.1) hi.2
theorem final_bound (p : Parameters) (pid : PID) (i : HIndex p)
    (hp : ValidPID p pid) (hi : hActive p pid i) :
    fOffset p pid i < p.B*p.H*p.K*p.V := by
  rw [final_address]
  exact packed_lt (packed_lt hp.2 hi.1) hi.2

theorem span_ge (p : Parameters) (hBC : 0 < p.BC) : p.BT ≤ innerCount p*p.BC := by
  have h := Nat.mod_add_div (p.BT+p.BC-1) p.BC
  have hm := Nat.mod_lt (p.BT+p.BC-1) hBC
  rw [Nat.mul_comm p.BC] at h
  dsimp [innerCount]
  omega

theorem last_chunk (p : Parameters) (h : ShapeValid p) (t : Nat) (ht : t < p.T) :
    t/p.BT < p.NT ∧ t/p.BT*p.BT ≤ t ∧ t < t/p.BT*p.BT+innerCount p*p.BC ∧
      ∀ c, t/p.BT < c → t < c*p.BT := by
  have hm := Nat.mod_lt t h.bt_pos
  have hd := Nat.div_add_mod t p.BT
  rw [Nat.mul_comm p.BT] at hd
  have hs := span_ge p h.bc_pos
  refine ⟨by rw [h.chunks]; exact owner_lt h.bt_pos ht,Nat.div_mul_le_self t p.BT,by omega,?_⟩
  intro c hc
  have hh := Nat.mul_le_mul_right p.BT hc
  simp only [Nat.succ_mul] at hh
  omega

#axiomsClean history_collision
#axiomsClean corrected_collision
#axiomsClean final_collision
#axiomsClean last_chunk
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Geometry

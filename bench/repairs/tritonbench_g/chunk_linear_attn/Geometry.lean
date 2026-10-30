import bench.repairs.tritonbench_g.chunk_linear_attn.GradientModel
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Geometry
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv owner_lt packed_lt packed_injective)
set_option maxHeartbeats 6000000
structure ShapeValid (p : Parameters) : Prop where
  bt_pos : 0 < p.BT
  bk_pos : 0 < p.BK
  bv_pos : 0 < p.BV
  chunks : p.NT=ceilDiv p.T p.BT

def StatePID (p : Parameters) (pid : PID) := pid.x<keyBlocks p ∧ pid.y<valueBlocks p ∧ pid.z<p.B*p.H
def OutputPID (p : Parameters) (pid : PID) := pid.x<valueBlocks p ∧ pid.y<p.NT ∧ pid.z<p.B*p.H
def GradientPID (p : Parameters) (pid : PID) := pid.x<keyBlocks p ∧ pid.y<p.NT ∧ pid.z<p.B*p.H

theorem pid_ext {a b : PID} (hx : a.x=b.x) (hy : a.y=b.y) (hz : a.z=b.z) : a=b := by
  cases a; cases b; simp_all

theorem history_address (p : Parameters) (pid : PID) (c : Nat) (i : TileIndex [p.BK,p.BV]) :
    hOffset p pid c i=((pid.z*p.NT+c)*p.K+(pid.x*p.BK+i.1.val))*p.V+(pid.y*p.BV+i.2.1.val) := by
  unfold hOffset; ring
theorem final_address (p : Parameters) (pid : PID) (i : TileIndex [p.BK,p.BV]) :
    fOffset p pid i=(pid.z*p.K+(pid.x*p.BK+i.1.val))*p.V+(pid.y*p.BV+i.2.1.val) := by
  unfold fOffset; ring
theorem output_address (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BV]) :
    outputOffset p pid i=(pid.z*p.T+(pid.y*p.BT+i.1.val))*p.V+(pid.x*p.BV+i.2.1.val) := by
  unfold outputOffset; ring
theorem gradient_address (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BK]) :
    gradKeyOffset p pid i=(pid.z*p.T+(pid.y*p.BT+i.1.val))*p.K+(pid.x*p.BK+i.2.1.val) := by
  unfold gradKeyOffset; ring
theorem partial_address (p : Parameters) (pid : PID) (heads vb : Nat) (i : TileIndex [p.BT,p.BV]) :
    gradValueOffset p pid heads vb i=((pid.x*heads+pid.z)*p.T+(pid.y*p.BT+i.1.val))*p.V+(vb*p.BV+i.2.1.val) := by
  unfold gradValueOffset; ring

theorem history_collision (p : Parameters) (a b : PID) (c d : Nat)
    (i j : TileIndex [p.BK,p.BV]) (hi : hActive p a i) (hj : hActive p b j)
    (hc : c<p.NT) (hd : d<p.NT) (he : hOffset p a c i=hOffset p b d j) :
    a=b ∧ c=d ∧ i=j := by
  rw [history_address,history_address] at he
  obtain ⟨he,hv⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨he,hk⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hz,hc⟩ := packed_injective hc hd he
  obtain ⟨hx,hki⟩ := packed_injective i.1.isLt j.1.isLt hk
  obtain ⟨hy,hvi⟩ := packed_injective i.2.1.isLt j.2.1.isLt hv
  exact ⟨pid_ext hx hy hz,hc,Prod.ext (Fin.ext hki) (Prod.ext (Fin.ext hvi) (Subsingleton.elim _ _))⟩
theorem final_collision (p : Parameters) (a b : PID) (i j : TileIndex [p.BK,p.BV])
    (hi : hActive p a i) (hj : hActive p b j) (he : fOffset p a i=fOffset p b j) : a=b ∧ i=j := by
  rw [final_address,final_address] at he
  obtain ⟨he,hv⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨hz,hk⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hx,hki⟩ := packed_injective i.1.isLt j.1.isLt hk
  obtain ⟨hy,hvi⟩ := packed_injective i.2.1.isLt j.2.1.isLt hv
  exact ⟨pid_ext hx hy hz,Prod.ext (Fin.ext hki) (Prod.ext (Fin.ext hvi) (Subsingleton.elim _ _))⟩
theorem output_collision (p : Parameters) (a b : PID) (i j : TileIndex [p.BT,p.BV])
    (hi : outputActive p a i) (hj : outputActive p b j) (he : outputOffset p a i=outputOffset p b j) : a=b ∧ i=j := by
  rw [output_address,output_address] at he
  obtain ⟨he,hv⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨hz,ht⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hy,hti⟩ := packed_injective i.1.isLt j.1.isLt ht
  obtain ⟨hx,hvi⟩ := packed_injective i.2.1.isLt j.2.1.isLt hv
  exact ⟨pid_ext hx hy hz,Prod.ext (Fin.ext hti) (Prod.ext (Fin.ext hvi) (Subsingleton.elim _ _))⟩
theorem gradient_collision (p : Parameters) (a b : PID) (i j : TileIndex [p.BT,p.BK])
    (hi : gradKeyActive p a i) (hj : gradKeyActive p b j) (he : gradKeyOffset p a i=gradKeyOffset p b j) : a=b ∧ i=j := by
  rw [gradient_address,gradient_address] at he
  obtain ⟨he,hk⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨hz,ht⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hy,hti⟩ := packed_injective i.1.isLt j.1.isLt ht
  obtain ⟨hx,hki⟩ := packed_injective i.2.1.isLt j.2.1.isLt hk
  exact ⟨pid_ext hx hy hz,Prod.ext (Fin.ext hti) (Prod.ext (Fin.ext hki) (Subsingleton.elim _ _))⟩
theorem partial_collision (p : Parameters) (heads : Nat) (a b : PID) (va vb : Nat) (i j : TileIndex [p.BT,p.BV])
    (ha : a.z<heads) (hb : b.z<heads) (hi : gradValueActive p a va i) (hj : gradValueActive p b vb j)
    (he : gradValueOffset p a heads va i=gradValueOffset p b heads vb j) : a=b ∧ va=vb ∧ i=j := by
  rw [partial_address,partial_address] at he
  obtain ⟨he,hv⟩ := packed_injective hi.2 hj.2 he
  obtain ⟨he,ht⟩ := packed_injective hi.1 hj.1 he
  obtain ⟨hx,hz⟩ := packed_injective ha hb he
  obtain ⟨hy,hti⟩ := packed_injective i.1.isLt j.1.isLt ht
  obtain ⟨hvb,hvi⟩ := packed_injective i.2.1.isLt j.2.1.isLt hv
  exact ⟨pid_ext hx hy hz,hvb,Prod.ext (Fin.ext hti) (Prod.ext (Fin.ext hvi) (Subsingleton.elim _ _))⟩

theorem history_bound (p : Parameters) (pid : PID) (c : Nat) (i : TileIndex [p.BK,p.BV])
    (hp : StatePID p pid) (hc : c<p.NT) (hi : hActive p pid i) : hOffset p pid c i < p.B*p.H*p.NT*p.K*p.V := by
  rw [history_address]; exact packed_lt (packed_lt (packed_lt hp.2.2 hc) hi.1) hi.2
theorem final_bound (p : Parameters) (pid : PID) (i : TileIndex [p.BK,p.BV])
    (hp : StatePID p pid) (hi : hActive p pid i) : fOffset p pid i < p.B*p.H*p.K*p.V := by
  rw [final_address]; exact packed_lt (packed_lt hp.2.2 hi.1) hi.2
theorem output_bound (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BV])
    (hp : OutputPID p pid) (hi : outputActive p pid i) : outputOffset p pid i < p.B*p.H*p.T*p.V := by
  rw [output_address]; exact packed_lt (packed_lt hp.2.2 hi.1) hi.2
theorem gradient_bound (p : Parameters) (pid : PID) (i : TileIndex [p.BT,p.BK])
    (hp : GradientPID p pid) (hi : gradKeyActive p pid i) : gradKeyOffset p pid i < p.B*p.H*p.T*p.K := by
  rw [gradient_address]; exact packed_lt (packed_lt hp.2.2 hi.1) hi.2
theorem partial_bound (p : Parameters) (pid : PID) (vb : Nat) (i : TileIndex [p.BT,p.BV])
    (hp : GradientPID p pid) (hi : gradValueActive p pid vb i) :
    gradValueOffset p pid (p.B*p.H) vb i < keyBlocks p*(p.B*p.H)*p.T*p.V := by
  rw [partial_address]; exact packed_lt (packed_lt (packed_lt hp.1 hp.2.2) hi.1) hi.2

theorem chunk_lt (p : Parameters) (rev : Bool) (n : Nat) (hn : n<p.NT) : chunk p rev n<p.NT := by
  cases rev <;> simp [chunk] <;> omega
theorem chunk_injective (p : Parameters) (rev : Bool) (a b : Nat) (ha : a<p.NT) (hb : b<p.NT)
    (h : chunk p rev a=chunk p rev b) : a=b := by cases rev <;> simp_all [chunk] <;> omega

#axiomsClean history_collision
#axiomsClean final_collision
#axiomsClean output_collision
#axiomsClean gradient_collision
#axiomsClean partial_collision
#axiomsClean history_bound
#axiomsClean partial_bound
end VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry

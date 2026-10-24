import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeGrid
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.RealSafety

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

def regions : List RegionName := ["q","k","v","h","g","do","dh","dq","dk","dg"]

def extent (c : Config) (r : RegionName) : Nat :=
  if r = "q" ∨ r = "k" ∨ r = "dq" ∨ r = "dk" then c.heads*c.T*c.K
  else if r = "v" ∨ r = "do" then c.heads*c.T*c.V
  else if r = "h" ∨ r = "dh" then c.heads*c.NT*c.K*c.V
  else if r = "g" then c.heads*c.T
  else if r = "dg" then c.NK*c.heads*c.T
  else 0

theorem extent_unlisted (c : Config) (r : RegionName) (h : r ∉ regions) : extent c r = 0 := by
  simp only [regions,List.mem_cons,List.mem_nil_iff,or_false,not_or] at h
  simp [extent,h]

theorem pack_bound (a A i w : Nat) (ha : a < A) (hi : i < w) : a*w+i < A*w := by
  calc
    a*w+i < a*w+w := Nat.add_lt_add_left hi _
    _ = (a+1)*w := by ring
    _ ≤ A*w := Nat.mul_le_mul_right w ha

theorem q_address_bound (c : Config) (a : CTA c) (r : Fin 64) (e : Fin c.BK)
    (ha : activeQK c a r e) : qAddress c a r e < c.heads*c.T*c.K :=
  pack_bound _ _ _ _ (pack_bound _ _ _ _ a.head.isLt ha.1) ha.2

theorem g_address_bound (c : Config) (a : CTA c) (r : Fin 64)
    (ha : activeG c a r) : gAddress c a r < c.NK*c.heads*c.T := by
  have h := pack_bound _ _ _ _ (pack_bound _ _ _ _ a.key.isLt a.head.isLt) ha
  simpa only [gAddress,Nat.add_assoc] using h

theorem dense_bounds (c : Config) (s : BlockState) (a : CTA c) :
    RealSafety.Bounds (prepare c s a) c.layout "q" "k" "v" "h" "g" "do" "dh"
      "dq" "dk" "dg" (extent c) := by
  have hq (i : TileIndex [64,c.BK])
      (hi : RealFrame.qkActive c.layout a.launch i) :
      RealFrame.qkAddress c.layout a.launch i < c.heads*c.T*c.K := by
    rcases i with ⟨r,e,⟨⟩⟩
    simpa only [RealFrame.qkAddress,Config.layout,CTA.launch,qAddress,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using q_address_bound c a r e hi
  have hg (i : TileIndex [64]) (hi : RealFrame.dgActive c.layout a.launch i) :
      RealFrame.dgAddress c.layout a.launch i < c.NK*c.heads*c.T := by
    rcases i with ⟨r,⟨⟩⟩
    exact g_address_bound c a r hi
  have hv (i : Fin 64) (j : Nat) (ht : a.time.val*64+i.val<c.T) (hj : j<c.V) :
      a.head.val*(c.T*c.V)+(a.time.val*64+i.val)*c.V+j<c.heads*c.T*c.V := by
    have h := pack_bound _ _ _ _ (pack_bound _ _ _ _ a.head.isLt ht) hj
    simpa only [Nat.add_mul,Nat.mul_assoc] using h
  have hh (j : Nat) (e : Fin c.BK) (hj : j<c.V) (he : a.key.val*c.BK+e.val<c.K) :
      a.head.val*(c.NT*c.K*c.V)+a.time.val*(c.K*c.V)+j+(a.key.val*c.BK+e.val)*c.V
        <c.heads*c.NT*c.K*c.V := by
    have h := pack_bound _ _ _ _ (pack_bound _ _ _ _
      (pack_bound _ _ _ _ a.head.isLt a.time.isLt) he) hj
    simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using h
  refine ⟨⟨?_,?_⟩,⟨?_,?_,?_,?_⟩,⟨?_,?_,?_,?_,?_⟩⟩
  · intro i hi
    have h := pack_bound _ _ _ _ a.head.isLt hi
    simpa only [extent,prepare,Config.layout,ite_true,ite_false,Nat.add_assoc] using h
  · have ht : min (a.time.val*64+64) c.T - 1 < c.T := by
      have := T_pos c a
      omega
    simpa [extent,prepare,Config.layout] using pack_bound _ _ _ _ a.head.isLt ht
  · simpa [extent,Config.layout,RealLoop.launch,prepare] using hv
  · simpa [extent,Config.layout,RealLoop.launch,prepare] using hv
  · simpa [extent,Config.layout,RealLoop.launch,prepare] using hh
  · simpa [extent,Config.layout,RealLoop.launch,prepare] using hh
  · simpa [extent,launch_prepare] using hq
  · simpa [extent,launch_prepare] using hq
  · simpa [extent,launch_prepare] using hq
  · simpa [extent,launch_prepare] using hq
  · simpa [extent,launch_prepare] using hg

theorem source_safe_dense (c : Config) (s : BlockState) (a : CTA c) :
    (kernel c).eraseDType.toAlgKernel.TraceSafe (extent c) (prepare c s a) :=
  RealSafety.source_safe (prepare c s a) c.layout .fp32 .fp32 .fp32 .fp32
    "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg" (extent c)
    (dense_bounds c s a) rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole

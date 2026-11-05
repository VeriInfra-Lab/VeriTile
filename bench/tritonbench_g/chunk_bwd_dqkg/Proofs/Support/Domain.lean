import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.DenseSafety

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
noncomputable section
set_option maxHeartbeats 2000000

def largest (c : Config) : Nat :=
  max (max (c.heads*c.T*c.K) (c.heads*c.T*c.V))
    (max (c.heads*c.NT*c.K*c.V) (c.NK*c.heads*c.T))
def unitBound (c : Config) : Nat :=
  max (max (max c.T c.K) (max c.V (c.NT*c.K))) (max (c.K*c.V) 1)
def envelope (c : Config) : Nat := largest c+(64+c.BK+c.BV+1)*unitBound c

structure Guard (c : Config) : Prop where
  keyPositive : 0<c.K
  valuePositive : 0<c.V
  integer32 : envelope c<2^31
  gridX : c.NK<2^31
  gridY : c.NT≤65535
  gridZ : c.heads≤65535

theorem unit_bounds (c : Config) :
    c.T≤unitBound c ∧ c.K≤unitBound c ∧ c.V≤unitBound c ∧
    c.NT*c.K≤unitBound c ∧ c.K*c.V≤unitBound c ∧ 1≤unitBound c := by
  unfold unitBound
  omega

theorem largest_bounds (c : Config) :
    c.heads*c.T*c.K≤largest c ∧ c.heads*c.T*c.V≤largest c ∧
    c.heads*c.NT*c.K*c.V≤largest c ∧ c.NK*c.heads*c.T≤largest c := by
  unfold largest
  omega

theorem value_start_lt (c : Config) (block : Nat) (h : block<(c.V+c.BV-1)/c.BV) :
    block*c.BV<c.V := by
  rcases width_cases c.V with hb|hb|hb
  all_goals
    change c.BV=_ at hb
    rw [hb] at h ⊢
    omega

/-- Every vector address, including checked-off tail lanes, lies in the host's
conservative signed-index envelope. This is stronger than active-load safety. -/
theorem formed_indices (c : Config) (a : CTA c) (block : Nat)
    (hb : block<(c.V+c.BV-1)/c.BV) (r : Fin 64) (e : Fin c.BK) (j : Fin c.BV) :
    qAddress c a r e < envelope c ∧
    a.head.val*(c.T*c.V)+(a.time.val*64+r.val)*c.V+(block*c.BV+j.val)<envelope c ∧
    a.head.val*(c.NT*c.K*c.V)+a.time.val*(c.K*c.V)+(block*c.BV+j.val)+
      (a.key.val*c.BK+e.val)*c.V<envelope c ∧
    gAddress c a r<envelope c := by
  obtain ⟨hT,hK,hV,hNK,hKV,h1⟩ := unit_bounds c
  obtain ⟨lq,lv,lh,lg⟩ := largest_bounds c
  have ht := time_start_lt c a
  have hk := key_start_lt c a
  have hv := value_start_lt c block hb
  have hp := pack_bound _ _ _ _ a.head.isLt ht
  have hh := pack_bound _ _ _ _ a.head.isLt a.time.isLt
  have hkh := pack_bound _ _ _ _ a.key.isLt a.head.isLt
  have rlt := r.isLt
  have elt := e.isLt
  have jlt := j.isLt
  have bkpos := width_pos c.K
  have bvpos := width_pos c.V
  change 0<c.BK at bkpos
  change 0<c.BV at bvpos
  have qbase := Nat.mul_le_mul_right c.K (Nat.le_of_lt hp)
  have vbase := Nat.mul_le_mul_right c.V (Nat.le_of_lt hp)
  have hbase := Nat.mul_le_mul_right (c.K*c.V) (Nat.le_of_lt hh)
  have gbase := Nat.mul_le_mul_right c.T (Nat.le_of_lt hkh)
  have rK := Nat.mul_le_mul_right c.K (Nat.le_of_lt rlt)
  have rV := Nat.mul_le_mul_right c.V (Nat.le_of_lt rlt)
  have eV := Nat.mul_le_mul_right c.V (Nat.le_of_lt (show a.key.val*c.BK+e.val<c.K+c.BK by omega))
  have mk := Nat.mul_le_mul_left 65 hK
  have mv := Nat.mul_le_mul_left 65 hV
  have mbk := Nat.mul_le_mul_left c.BK h1
  have mbv := Nat.mul_le_mul_left c.BV h1
  have mt := Nat.mul_le_mul_left 65 hT
  have mbkv := Nat.mul_le_mul_left c.BK hV
  unfold qAddress gAddress envelope
  constructor
  · nlinarith
  constructor
  · nlinarith
  constructor
  · nlinarith
  · nlinarith

theorem formed_signed32 (c : Config) (guard : Guard c) (a : CTA c) (block : Nat)
    (hb : block<(c.V+c.BV-1)/c.BV) (r : Fin 64) (e : Fin c.BK) (j : Fin c.BV) :
    qAddress c a r e<2^31 ∧
    a.head.val*(c.T*c.V)+(a.time.val*64+r.val)*c.V+(block*c.BV+j.val)<2^31 ∧
    a.head.val*(c.NT*c.K*c.V)+a.time.val*(c.K*c.V)+(block*c.BV+j.val)+
      (a.key.val*c.BK+e.val)*c.V<2^31 ∧ gAddress c a r<2^31 := by
  obtain ⟨hq,hv,hh,hg⟩ := formed_indices c a block hb r e j
  exact ⟨hq.trans guard.integer32,hv.trans guard.integer32,hh.trans guard.integer32,hg.trans guard.integer32⟩

/-- The same host pointer check applies to input and newly allocated output
buffers. FP32 cells use four bytes; masked address formation cannot wrap. -/
theorem formed_pointer64 (c : Config) (base off : Nat)
    (hb : base+4*envelope c<2^64) (ho : off<envelope c) : base+4*off<2^64 := by
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole

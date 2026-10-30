import bench.tritonbench_g.flash_attn.Proofs.Host
namespace VeriTile.Bench.Repairs.FlashAttn
set_option maxHeartbeats 4000000

theorem signed32_exact (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  rw [BitVec.toInt_eq_toNat_of_lt]
  · simp;omega
  · simp [Nat.mod_eq_of_lt (show n<2^32 by omega)];omega

theorem byte_offset_bound (n : Nat) (h : n<2^31) (i : Fin 4) : 4*n+i.val<2^63 := by omega

theorem masked_row_i32 (a : Host.Arguments) (valid : Host.Contract a) (l : Lane)
    (hl : l.Valid (Host.shape a)) : l.pidRow*128+l.rowLane<2^31 := by
  have hr:=row_envelope (Host.shape a) l hl
  have hlen : a.q.shape.length≤a.q.shape.batches*a.q.shape.heads*a.q.shape.length := by
    have hp : 1≤a.q.shape.batches*a.q.shape.heads := by
      have h₁:=valid.qguard.positive.1
      have h₂:=valid.qguard.positive.2.1
      nlinarith
    nlinarith
  have hs:=valid.scratch
  dsimp [Host.shape] at hr
  omega

theorem stop_endpoint_i32 (a : Host.Arguments) (valid : Host.Contract a) (pid : Nat)
    (hp : pid<ceil a.q.shape.length 128) : (pid+1)*128<2^31 := by
  have hs:=valid.scratch
  have hlen : a.q.shape.length≤a.q.shape.batches*a.q.shape.heads*a.q.shape.length := by
    have h₁:=valid.qguard.positive.1
    have h₂:=valid.qguard.positive.2.1
    have hp : 1≤a.q.shape.batches*a.q.shape.heads := by nlinarith
    nlinarith
  unfold ceil at hp
  omega

theorem scratch_formed_i32 (a : Host.Arguments) (valid : Host.Contract a) (l : Lane)
    (hl : l.Valid (Host.shape a)) :
    l.pidHead*a.q.shape.length+(l.pidRow*128+l.rowLane)<2^31 := by
  have hr:=row_envelope (Host.shape a) l hl
  have hh:=hl.2.1
  dsimp [Host.shape] at hr hh
  have hp:=Nat.mul_le_mul_right a.q.shape.length (show l.pidHead+1≤a.q.shape.batches*a.q.shape.heads by omega)
  have hs:=valid.scratch
  nlinarith

theorem key_final_envelope (s : Shape) (pid block lane : Nat) (hb : block≤rounds s pid)
    (hi : lane<64) : block*64+lane<s.length+127 := by
  have hh:=rounds_envelope s pid
  omega
#axiomsClean signed32_exact
#axiomsClean byte_offset_bound
#axiomsClean masked_row_i32
#axiomsClean stop_endpoint_i32
#axiomsClean scratch_formed_i32
#axiomsClean key_final_envelope
end VeriTile.Bench.Repairs.FlashAttn

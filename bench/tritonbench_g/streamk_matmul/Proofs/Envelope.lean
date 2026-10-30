import bench.tritonbench_g.streamk_matmul.Proofs.TensorBinding
import bench.tritonbench_g.streamk_matmul.Proofs.ABI

namespace VeriTile.Bench.Repairs.StreamK.Checked.Envelope
open Contract TensorBinding Whole001 VeriTile.Bench.TritonBenchG.StreamK
set_option maxHeartbeats 2000000

theorem valid (a b : Tensor) (c : Config) (h : InputContract a.metadata b.metadata c)
    (nonempty : ¬Empty a.metadata b.metadata) : ABI.Valid (parameters a b c h.profile) := by
  obtain ⟨control,stream,bytes,wa,wb⟩ := execution_nonempty a.metadata b.metadata c h nonempty
  let p := parameters a b c h.profile
  change Control p.h at control
  change StreamEnvelope p.h at stream
  change ByteEnvelope p.h at bytes
  rcases control with ⟨cm,cn,ck,cg,ct,cp⟩
  rcases bytes with ⟨ba,bb,bc,bs,bma,bkb,bmn⟩
  have bm := h.profile.1
  have bn := h.profile.2.1
  have bk := h.profile.2.2.1
  change Block p.h.bm at bm
  change Block p.h.bn at bn
  change Block p.h.bk at bk
  unfold Block at bm bn bk
  change ABI.Valid p
  have hm : p.h.m<2^31 := by omega
  have hn : p.h.n<2^31 := by omega
  have hk : p.h.k<2^31 := by omega
  have ia : p.strideAM=p.h.k := rfl
  have ib : p.strideBK=p.h.n := rfl
  have oneA : p.strideAK=1 := rfl
  have oneB : p.strideBN=1 := rfl
  have pa : pointerA p.h<2^63 := by omega
  have pb : pointerB p.h<2^63 := by omega
  have pc : pointerC p.h<2^63 := by omega
  have paddedM : ceilDiv p.h.m p.h.bm*p.h.bm<2^63 := by
    unfold pointerC at pc
    have mul := Nat.le_mul_of_pos_right (ceilDiv p.h.m p.h.bm*p.h.bm) (by omega : 0<max p.h.n 1)
    omega
  have paddedN : ceilDiv p.h.n p.h.bn*p.h.bn<2^63 := by unfold pointerB at pb; omega
  have iters : (Plan.select p.h).iterations<2^31 := by
    rw [Plan.selected_iterations]
    unfold ceilDiv
    exact lt_of_le_of_lt (Nat.div_le_self _ _) ck
  refine ⟨⟨hm,hn,hk⟩,⟨by omega,by omega,by omega⟩,cm,cn,ck,cp,iters,cg,
    paddedM,paddedN,?_,stream,?_,?_,?_,?_,?_,?_,?_⟩
  · unfold StreamEnvelope at stream; omega
  · rw [ia,ib,oneA,oneB]; omega
  · rw [oneA,Nat.mul_one]
    unfold pointerA advancedK at pa
    omega
  · rw [ib]
    unfold pointerB advancedK at pb
    omega
  · simpa only [pointerA,advancedK,ia,oneA,Nat.mul_one] using pa
  · simpa only [pointerB,advancedK,ib,oneB,Nat.mul_one] using pb
  · have mul := Nat.mul_le_mul_left (ceilDiv p.h.m p.h.bm*p.h.bm) (Nat.le_max_left p.h.n 1)
    unfold pointerC at pc
    omega
  · omega

/-- A guarded natural expression materializes as the same signed i64 value. -/
theorem signed64 (n : Nat) (h : n<2^63) : (BitVec.ofNat 64 n).toInt=n := by
  have hn : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega : n<2^64)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

#axiomsClean valid
#axiomsClean signed64
end VeriTile.Bench.Repairs.StreamK.Checked.Envelope

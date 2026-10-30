import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Admission
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.SourcePointer

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedNative
open VeriTile Triton HostFP Address SourceAddress
set_option maxHeartbeats 6000000

def capacity (c : HostFP.Inner) : InnerSite → Nat
  | .key | .gateKey | .gateKeyEnd=>c.heads*c.PT*c.PK
  | .value | .gateValue | .gateValueEnd=>c.heads*c.PT*c.PV
  | .history=>c.heads*c.NT*c.PK*c.PV
  | .seed | .final=>c.heads*c.PK*c.PV

theorem site_bound (r : Admission.Inner) (checked : r.check=Bool.true)
    (i : InnerCoord r.config) (site : InnerSite) : innerOffset r.config i site<capacity r.config site := by
  have blocks:=r.blocks checked
  have hb:=inner_bounds r.config i blocks.1 blocks.2.1 blocks.2.2
  have hk:=Address.flatten3_bounded i.head.isLt hb.2.2.1 hb.1
  have hv:=Address.flatten3_bounded i.head.isLt hb.2.2.1 hb.2.1
  have hke:=Address.flatten3_bounded i.head.isLt hb.2.2.2 hb.1
  have hve:=Address.flatten3_bounded i.head.isLt hb.2.2.2 hb.2.1
  have hs:=Address.flatten3_bounded i.head.isLt hb.1 hb.2.1
  have hh:=Address.history_bounded i.head.isLt i.chunk.isLt hb.1 hb.2.1
  cases site
  · exact hs
  · exact hk
  · exact hv
  · exact hh
  · exact hk
  · exact hke
  · exact hv
  · exact hve
  · exact hs

theorem site_fits (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (site : InnerSite) : Admission.Fits (capacity r.config site) := by
  have bounds:=r.padded_fits checked nonempty
  cases site <;> first | exact bounds.1 | exact bounds.2.1 | exact bounds.2.2.1 | exact bounds.2.2.2

/-- Every source pointer polynomial, signed byte scaling, actual pointer
addition and alignment follows from checked extents and the local pointer
guard. No allocator supplies a native-execution conclusion. -/
theorem site_native (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (i : InnerCoord r.config) (site : InnerSite) (base : Nat)
    (aligned : base%4=0) (span : base+4*capacity r.config site<2^64) :
    (innerWord r.config i site).toInt=(innerOffset r.config i site : Int) ∧
    (innerWord r.config i site*word 4).toInt=(4*innerOffset r.config i site : Nat) ∧
    (BitVec.ofNat 64 base+innerWord r.config i site*word 4).toNat=base+4*innerOffset r.config i site ∧
    (base+4*innerOffset r.config i site)%4=0 ∧ base+4*innerOffset r.config i site+4<2^64 := by
  have bound:=site_bound r checked i site
  have fits:=site_fits r checked nonempty site
  unfold Admission.Fits at fits
  have offset : innerOffset r.config i site<2^63 := by omega
  have bytes : 4*innerOffset r.config i site<2^63 := by omega
  rw [inner_word r.config i site (r.blocks checked).1]
  have scale : word (innerOffset r.config i site)*word 4=word (4*innerOffset r.config i site) := by
    simp [word,←BitVec.ofNat_mul,Nat.mul_comm]
  rw [scale]
  refine ⟨word_exact offset,word_exact bytes,?_,by omega,by omega⟩
  simp only [word,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  omega

theorem positive (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty) :
    0<r.config.heads ∧ 0<r.config.NT ∧ 1<r.config.PK ∧ 0<r.config.PV := by
  have hn : r.B*r.H*r.T*r.K*r.V≠0 := nonempty
  have hh : 0<r.B*r.H := by
    by_contra h
    have z : r.B*r.H=0 := by omega
    simp [z] at hn
  have ht : 0<r.T := by
    by_contra h
    have z : r.T=0 := by omega
    simp [z] at hn
  have hk : 0<r.K := by
    by_contra h
    have z : r.K=0 := by omega
    simp [z] at hn
  have hv : 0<r.V := by
    by_contra h
    have z : r.V=0 := by omega
    simp [z] at hn
  have blocks:=r.blocks checked
  have tcover:=HostShape.le_padded r.T r.BT blocks.1
  have kcover:=HostShape.le_padded r.K r.BK blocks.2.1
  have vcover:=HostShape.le_padded r.V r.BV blocks.2.2
  have nt : 0<HostFP.chunks r.T r.BT := by unfold HostShape.padded HostShape.chunks at tcover;unfold HostFP.chunks;nlinarith
  have nk : 0<HostFP.chunks r.K r.BK := by unfold HostShape.padded HostShape.chunks at kcover;unfold HostFP.chunks;nlinarith
  refine ⟨hh,nt,?_,lt_of_lt_of_le hv vcover⟩
  have bk:=((r.check_iff).mp checked).2.1.1
  change 1<HostFP.chunks r.K r.BK*r.BK
  nlinarith

theorem total_bound (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty) :
    r.config.total<2^63 := by
  have pos:=positive r checked nonempty
  have count:=r.padded_fits checked nonempty
  have h0 : r.config.heads≤r.config.heads*r.config.NT:=Nat.le_mul_of_pos_right _ pos.2.1
  have h1 : r.config.total≤r.config.heads*r.config.PK*r.config.PV:=
    Nat.mul_le_mul (Nat.mul_le_mul_left _ (Nat.div_le_self _ _)) (Nat.div_le_self _ _)
  have h2:=Nat.mul_le_mul_right r.config.PV (Nat.mul_le_mul_right r.config.PK h0)
  have fits:=count.2.2.1
  unfold Admission.Fits at fits
  omega

theorem linear (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (p : LaunchPlan.Item) (hp : p∈r.config.plans) (lane : Fin p.size) :
    (word lane.val+word p.base).toInt=(p.base+lane.val : Nat) := by
  have he : r.config.empty=Bool.false := by simpa [HostFP.Inner.empty,Admission.Inner.config,Admission.Inner.nonempty] using nonempty
  have hm : p∈LaunchPlan.plans r.config.total := by simpa [HostFP.Inner.plans,he] using hp
  have hb:=(LaunchPlan.plans_valid _ p hm).2.2
  exact NativeControl.linear_exact (by have:=lane.isLt;omega) (total_bound r checked nonempty).le

theorem range (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty) :
    (word 0).toInt=0 ∧ (word r.config.NT).toInt=(r.config.NT : Int) ∧ (word 1).toInt=1 ∧
    (∀ i,i≤r.config.NT → ((word i).toInt<(word r.config.NT).toInt ↔ i<r.config.NT)) ∧
    (∀ i,i<r.config.NT → (word i+word 1).toInt=(i+1 : Nat)) := by
  have pos:=positive r checked nonempty
  have fits:=(r.padded_fits checked nonempty).2.2.1
  unfold Admission.Fits at fits
  apply NativeControl.wide_range
  exact NativeControl.history_range _ _ _ _ pos.1 pos.2.2.1 pos.2.2.2 (by omega)

#axiomsClean site_native
#axiomsClean linear
#axiomsClean range
end VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedNative

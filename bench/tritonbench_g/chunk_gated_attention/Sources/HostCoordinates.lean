import bench.tritonbench_g.chunk_gated_attention.Sources.HostStorage
import bench.tritonbench_g.chunk_gated_attention.Sources.CheckedNative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostCoordinates
open HostStorage

theorem decode3 {H K V i : Nat} (hi : i<H*K*V) : i/(K*V)<H ∧ (i/V)%K<K ∧ i%V<V := by
  have hp : 0<H*K*V:=lt_of_le_of_lt (Nat.zero_le i) hi
  have hv : 0<V := by by_contra h;have z:=Nat.eq_zero_of_not_pos h;simp [z] at hp
  have hk : 0<K := by by_contra h;have z:=Nat.eq_zero_of_not_pos h;simp [z] at hp
  refine ⟨?_,Nat.mod_lt _ hk,Nat.mod_lt _ hv⟩
  apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos hk hv)).2
  simpa only [Nat.mul_assoc] using hi

theorem decode4 {H C K V i : Nat} (hi : i<H*C*K*V) :
    i/(C*K*V)<H ∧ (i/(K*V))%C<C ∧ (i/V)%K<K ∧ i%V<V := by
  have hp : 0<H*C*K*V:=lt_of_le_of_lt (Nat.zero_le i) hi
  have hv : 0<V := by by_contra h;have z:=Nat.eq_zero_of_not_pos h;simp [z] at hp
  have hk : 0<K := by by_contra h;have z:=Nat.eq_zero_of_not_pos h;simp [z] at hp
  have hc : 0<C := by by_contra h;have z:=Nat.eq_zero_of_not_pos h;simp [z] at hp
  refine ⟨?_,Nat.mod_lt _ hc,Nat.mod_lt _ hk,Nat.mod_lt _ hv⟩
  apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos (Nat.mul_pos hc hk) hv)).2
  simpa only [Nat.mul_assoc] using hi

theorem history_bound (r : Admission.Inner) (checked : r.check=Bool.true) (i : Nat) (hi : i < innerCount r) :
    HostData.historyOffset r i<historyCapacity r := by
  have hb:=decode4 hi
  have blocks:=r.blocks checked
  have hk:=HostShape.le_padded r.K r.BK blocks.2.1
  have hv:=HostShape.le_padded r.V r.BV blocks.2.2
  simpa [HostData.historyOffset,historyCapacity,Admission.Inner.config,HostFP.Inner.PK,HostFP.Inner.PV,Nat.mul_assoc] using
    Address.history_bounded hb.1 hb.2.1 (lt_of_lt_of_le hb.2.2.1 hk) (lt_of_lt_of_le hb.2.2.2 hv)

theorem final_bound (r : Admission.Inner) (checked : r.check=Bool.true) (i : Nat) (hi : i<finalCount r) :
    HostData.finalOffset r i<finalCapacity r := by
  have hb:=decode3 hi
  have blocks:=r.blocks checked
  have hk:=HostShape.le_padded r.K r.BK blocks.2.1
  have hv:=HostShape.le_padded r.V r.BV blocks.2.2
  simpa [HostData.finalOffset,finalCapacity,Admission.Inner.config,HostFP.Inner.PK,HostFP.Inner.PV,Nat.mul_assoc] using
    Address.flatten3_bounded hb.1 (lt_of_lt_of_le hb.2.1 hk) (lt_of_lt_of_le hb.2.2 hv)

theorem prefix_bound (r : Admission.Prefix) (checked : r.check=Bool.true) (i : Nat) (hi : i<prefixCount r) :
    HostData.prefixOffset r i<prefixCapacity r := by
  have hb:=decode3 hi
  have bt:=((r.check_iff).mp checked).1.1
  have ht:=HostShape.le_padded r.T r.BT (by omega)
  have hs:=HostShape.le_padded r.S 64 (by decide)
  simpa [HostData.prefixOffset,prefixCapacity,Nat.mul_assoc] using
    Address.flatten3_bounded hb.1 (lt_of_lt_of_le hb.2.1 ht) (lt_of_lt_of_le hb.2.2 hs)

theorem empty_count (r : Admission.Inner) (checked : r.check=Bool.true) (empty : ¬r.nonempty) : innerCount r=0 := by
  have he : r.B*r.H*r.T*r.K*r.V=0:=not_not.mp empty
  have bt:=(r.blocks checked).1
  have nt0 (ht : r.T=0) : r.config.NT=0 := by
    simp [Admission.Inner.config,HostFP.Inner.NT,HostFP.chunks,ht,Nat.div_eq_of_lt (show r.BT-1<r.BT by omega)]
  rcases mul_eq_zero.mp he with hab|hv
  · rcases mul_eq_zero.mp hab with hab|hk
    · rcases mul_eq_zero.mp hab with hbh|ht
      · simp [innerCount,hbh]
      · simp [innerCount,nt0 ht]
    · simp [innerCount,hk]
  · simp [innerCount,hv]

#axiomsClean history_bound
#axiomsClean final_bound
#axiomsClean prefix_bound
#axiomsClean empty_count
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostCoordinates

import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CheckedNative
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.SourceCumulative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedPrefixNative
open VeriTile Triton HostFP Address SourceAddress SourceCumulative
set_option maxHeartbeats 6000000

def Selected (BS : Nat) : Prop := BS=16 ∨ BS=32 ∨ BS=64
instance (BS : Nat) : Decidable (Selected BS) := inferInstanceAs (Decidable (_ ∨ _))

theorem counts (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : Selected BS) :
    (r.config BS).total≤(r.config BS).maximum ∧ (r.config BS).maximum<2^63 := by
  let c:=r.config BS
  have bs : 16≤BS := by rcases selected with h|h|h <;> omega
  have hs : c.PS/c.BS≤c.PS/16:=Nat.div_le_div_left bs (by decide)
  have hm : c.maximum≤c.heads*c.PT*c.PS:=
    Nat.mul_le_mul (Nat.mul_le_mul_left _ (Nat.div_le_self _ _)) (Nat.div_le_self _ _)
  have fits:=(((r.check_iff).mp checked).2.2.2 nonempty).1
  change 4*(c.heads*c.PT*c.PS)<2^63 at fits
  refine ⟨Nat.mul_le_mul_left _ hs,?_⟩
  change c.maximum<2^63
  omega

theorem offset_bound (r : Admission.Prefix) (checked : r.check=Bool.true)
    (BS : Nat) (selected : Selected BS) (i : Coord (r.config BS)) :
    offset (r.config BS) i<(r.config BS).heads*(r.config BS).PT*(r.config BS).PS := by
  have bt:=((r.check_iff).mp checked).1.1
  have ht:=SourceAddress.padded_tile (n:=r.T) (b:=r.BT) (by omega) i.chunk.isLt i.time.isLt
  have hf : i.tile.val*BS+i.feature.val<HostFP.padded r.S 64 := by
    have hi:=i.tile.isLt
    have hl:=i.feature.isLt
    change i.tile.val<HostFP.padded r.S 64/BS at hi
    change i.feature.val<BS at hl
    have he : HostFP.padded r.S 64/BS*BS=HostFP.padded r.S 64 := by
      rcases selected with h|h|h <;> rw [h] <;> unfold HostFP.padded <;> omega
    have hm:=Nat.mul_le_mul_right BS (Nat.succ_le_iff.mpr hi)
    simp only [Nat.succ_mul,he] at hm
    omega
  simpa only [offset,Nat.add_assoc,Admission.Prefix.config,HostFP.Cumulative.PT,HostFP.Cumulative.PS] using
    Address.flatten3_bounded i.head.isLt ht hf

theorem pointer (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : Selected BS) (i : Coord (r.config BS)) (base : Nat)
    (aligned : base%4=0) (span : base+4*((r.config BS).heads*(r.config BS).PT*(r.config BS).PS)<2^64) :
    (nativeWord (r.config BS) i).toInt=(offset (r.config BS) i : Int) ∧
    (nativeWord (r.config BS) i*word 4).toInt=(4*offset (r.config BS) i : Nat) ∧
    (BitVec.ofNat 64 base+nativeWord (r.config BS) i*word 4).toNat=base+4*offset (r.config BS) i ∧
    (base+4*offset (r.config BS) i)%4=0 ∧ base+4*offset (r.config BS) i+4<2^64 := by
  have bound:=offset_bound r checked BS selected i
  have fits:=(((r.check_iff).mp checked).2.2.2 nonempty).1
  change 4*((r.config BS).heads*(r.config BS).PT*(r.config BS).PS)<2^63 at fits
  have elem : offset (r.config BS) i<2^63:=by omega
  have bytes : 4*offset (r.config BS) i<2^63:=by omega
  rw [word_eq]
  have scale : word (offset (r.config BS) i)*word 4=word (4*offset (r.config BS) i) := by
    simp [word,←BitVec.ofNat_mul,Nat.mul_comm]
  rw [scale]
  refine ⟨word_exact elem,word_exact bytes,?_,by omega,by omega⟩
  simp only [word,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  omega

theorem linear (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : Selected BS) (job : LaunchPlan.Item) (hj : job∈(r.config BS).plans)
    (lane : Fin job.size) : (word lane.val+word job.base).toInt=(job.base+lane.val : Nat) := by
  let c:=r.config BS
  have hc:=counts r checked nonempty BS selected
  have hp : job∈AutotunePlan.plans c.maximum c.total := by
    have hh:=hj
    unfold HostFP.Cumulative.plans at hh
    split_ifs at hh with he
    · simp at hh
    · exact hh
  obtain ⟨q,hq,rfl⟩:=List.mem_map.mp hp
  have hs:=AutotunePlan.shrink c.maximum c.total hc.1 q hq
  have hv:=(LaunchPlan.plans_valid c.maximum q hq).2.2
  have hl:=lane.isLt
  exact NativeControl.linear_exact (by change q.base+lane.val<c.maximum;omega) hc.2.le

theorem guard (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : Selected BS) (job : LaunchPlan.Item) (hj : job∈(r.config BS).plans)
    (lane : Fin job.size) :
    ((word lane.val+word job.base).toInt<(word (r.config BS).total).toInt ↔ job.base+lane.val<(r.config BS).total) := by
  have hc:=counts r checked nonempty BS selected
  rw [linear r checked nonempty BS selected job hj lane,word_exact (lt_of_le_of_lt hc.1 hc.2)]
  omega

#axiomsClean pointer
#axiomsClean linear
#axiomsClean guard
end VeriTile.Bench.Repairs.ChunkGatedAttention.CheckedPrefixNative

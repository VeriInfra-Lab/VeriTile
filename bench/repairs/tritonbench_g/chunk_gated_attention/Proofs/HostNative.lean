import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostNative
open VeriTile Triton SourceAddress
open VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
open HostStorage
set_option maxHeartbeats 6000000

def siteTensor {r before} (p : InnerPrepared r before) : InnerSite → Admission.Tensor
  | .seed=>p.seed | .key=>p.key | .value=>p.value | .history=>p.history | .final=>p.final
  | .gateKey | .gateKeyEnd | .gateValue | .gateValueEnd=>p.gate

theorem site_count (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (site : InnerSite) (active : enabled r.config site) :
    (siteTensor p site).shape.prod=CheckedNative.capacity r.config site := by
  cases site <;> simp only [siteTensor]
  · rw [(p.seedValid active).2.2.2.2.2.2.2.2]
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,Nat.mul_assoc]
  · rw [p.keyValid.2.2.2.2.2.2.2.2]
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,Nat.mul_assoc]
  · rw [p.valueValid.2.2.2.2.2.2.2.2]
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,Nat.mul_assoc]
  · rw [p.historyValid.2.2.2.2.2.2.2.2]
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,Nat.mul_assoc]
  · rw [p.gateValid.2.2.2.2.2.2.2.2]
    have flag : r.gateK=Bool.true:=active
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,flag,Nat.mul_assoc]
  · rw [p.gateValid.2.2.2.2.2.2.2.2]
    have flag : r.gateK=Bool.true:=active
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,flag,Nat.mul_assoc]
  · rw [p.gateValid.2.2.2.2.2.2.2.2]
    have flag : r.gateK=Bool.false:=active
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,flag,Nat.mul_assoc]
  · rw [p.gateValid.2.2.2.2.2.2.2.2]
    have flag : r.gateK=Bool.false:=active
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,flag,Nat.mul_assoc]
  · rw [(p.finalValid active).2.2.2.2.2.2.2.2]
    simp [siteTensor,CheckedNative.capacity,Admission.Inner.config,Nat.mul_assoc]

theorem site_address (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (pointers : p.check=Bool.true) (site : InnerSite) (active : enabled r.config site) :
    Admission.Address (siteTensor p site) := by
  have all : ∀ t∈p.tensors,Admission.Address t := by simpa [InnerPrepared.check] using pointers
  apply all
  cases site <;> simp [siteTensor,InnerPrepared.tensors,enabled,Admission.Inner.config] at active ⊢
  all_goals simp_all

theorem inner (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (before : Memory) (p : InnerPrepared r before) (pointers : p.check=Bool.true)
    (i : InnerCoord r.config) (site : InnerSite) (active : enabled r.config site) :
    (innerWord r.config i site).toInt=(innerOffset r.config i site : Int) ∧
    (innerWord r.config i site*Address.word 4).toInt=(4*innerOffset r.config i site : Nat) ∧
    (BitVec.ofNat 64 (siteTensor p site).byteBase+innerWord r.config i site*Address.word 4).toNat=
      (siteTensor p site).byteBase+4*innerOffset r.config i site ∧
    ((siteTensor p site).byteBase+4*innerOffset r.config i site)%4=0 ∧
    (siteTensor p site).byteBase+4*innerOffset r.config i site+4<2^64 := by
  have address:=site_address r before p pointers site active
  apply CheckedNative.site_native r checked nonempty i site _ address.1
  rw [←site_count r before p site active]
  exact address.2

theorem prefix_input (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : CheckedPrefixNative.Selected BS) (before : Memory)
    (p : PrefixPrepared r before) (pointers : p.check=Bool.true) (i : SourceCumulative.Coord (r.config BS)) :
    (SourceCumulative.nativeWord (r.config BS) i).toInt=(SourceCumulative.offset (r.config BS) i : Int) ∧
    (SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toInt=(4*SourceCumulative.offset (r.config BS) i : Nat) ∧
    (BitVec.ofNat 64 p.input.byteBase+SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toNat=
      p.input.byteBase+4*SourceCumulative.offset (r.config BS) i ∧
    (p.input.byteBase+4*SourceCumulative.offset (r.config BS) i)%4=0 ∧
    p.input.byteBase+4*SourceCumulative.offset (r.config BS) i+4<2^64 := by
  have addresses : Admission.Address p.input ∧ Admission.Address p.output := by
    simpa [PrefixPrepared.check] using pointers
  have count : p.input.shape.prod=(r.config BS).heads*(r.config BS).PT*(r.config BS).PS := by
    rw [p.inputValid.2.2.2.2.2.2.2.2]
    simp [Admission.Prefix.config,HostFP.Cumulative.PT,HostFP.Cumulative.PS,Nat.mul_assoc]
  apply CheckedPrefixNative.pointer r checked nonempty BS selected i _ addresses.1.1
  rw [←count]
  exact addresses.1.2

theorem prefix_output (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : CheckedPrefixNative.Selected BS) (before : Memory)
    (p : PrefixPrepared r before) (pointers : p.check=Bool.true) (i : SourceCumulative.Coord (r.config BS)) :
    (SourceCumulative.nativeWord (r.config BS) i).toInt=(SourceCumulative.offset (r.config BS) i : Int) ∧
    (SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toInt=(4*SourceCumulative.offset (r.config BS) i : Nat) ∧
    (BitVec.ofNat 64 p.output.byteBase+SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toNat=
      p.output.byteBase+4*SourceCumulative.offset (r.config BS) i ∧
    (p.output.byteBase+4*SourceCumulative.offset (r.config BS) i)%4=0 ∧
    p.output.byteBase+4*SourceCumulative.offset (r.config BS) i+4<2^64 := by
  have addresses : Admission.Address p.input ∧ Admission.Address p.output := by
    simpa [PrefixPrepared.check] using pointers
  have count : p.output.shape.prod=(r.config BS).heads*(r.config BS).PT*(r.config BS).PS := by
    rw [p.outputValid.2.2.2.2.2.2.2.2]
    simp [Admission.Prefix.config,HostFP.Cumulative.PT,HostFP.Cumulative.PS,Nat.mul_assoc]
  apply CheckedPrefixNative.pointer r checked nonempty BS selected i _ addresses.2.1
  rw [←count]
  exact addresses.2.2

#axiomsClean inner
#axiomsClean prefix_input
#axiomsClean prefix_output
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostNative

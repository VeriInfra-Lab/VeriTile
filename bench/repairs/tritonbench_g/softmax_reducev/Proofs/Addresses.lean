import bench.repairs.tritonbench_g.softmax_reducev.Proofs.TensorHost

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.Checked
open VeriTile.Triton

def word (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

theorem signed32 (n : Nat) (bound : n < 2^31) : (word n).toInt=(n : Int) := by
  unfold word
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le <;> norm_num <;> omega

theorem checked_active_offsets (c : Call) (h : MetadataContract c) (x : Host.Input α)
    (data : Content c x) (j : Host.Job) (hj : Host.Legal (host c) j)
    (k d : Nat) (hk : k < x.counts j.batch) (hd : d < dimensions c) :
    (word (pageOffset (host c) x j k)).toInt=(pageOffset (host c) x j k : Int) ∧
    (word (logicAddress (host c) x j k)).toInt=(logicAddress (host c) x j k : Int) ∧
    (word (valueAddress (host c) x j k d)).toInt=(valueAddress (host c) x j k d : Int) ∧
    (word (Host.address (host c) j d)).toInt=(Host.address (host c) j d : Int) := by
  have bounds := active_int32 (host c) x (checked_admissible c h x data) j hj k d hk hd
  exact ⟨signed32 _ bounds.1,signed32 _ bounds.2.1,signed32 _ bounds.2.2.1,signed32 _ bounds.2.2.2⟩

theorem checked_padded_offsets (c : Call) (h : MetadataContract c) (x : Host.Input α)
    (data : Content c x) (j : Host.Job) (hj : Host.Legal (host c) j)
    (cur lane : Nat) (reached : cur < x.counts j.batch) (hlane : lane < 64) :
    cur+64 < 2^31 ∧ cur+lane < 2^31 ∧
    pageOffset (host c) x j (cur+lane) < 2^31 ∧ logicAddress (host c) x j (cur+lane) < 2^31 :=
  padded_int32 (host c) x (checked_admissible c h x data) j hj cur lane reached hlane

theorem mask_exact (i n : Nat) (hi : i < 2^31) (hn : n < 2^31) :
    (word i).slt (word n)=decide (i < n) := by
  simp [BitVec.slt,signed32 i hi,signed32 n hn]

theorem payload_address (x : Object) (contiguous : x.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i < x.metadata.numel) :
    x.tensor.metadata.dataPtr.toNat+4*i=x.tensor.storage.address (x.tensor.physicalIndex i) ∧
    x.tensor.metadata.dataPtr.toNat+4*i+4 ≤ x.tensor.storage.base+4*x.tensor.storage.elements ∧
    x.tensor.metadata.dataPtr.toNat+4*i+4 ≤ 2^64 := by
  change i < x.tensor.metadata.numel at hi
  have nonempty : 0 < x.tensor.metadata.numel := by omega
  let v := VeriTile.Torch.FP32.dataView x.tensor x.laws contiguous
  have pointer := VeriTile.Torch.FP32.actual_pointer x.tensor x.laws contiguous nonempty
  have span := v.spanFits
  have baseBound : v.base < 2^64 := by
    change v.base+4*x.tensor.metadata.numel ≤ 2^64 at span
    omega
  have ptr : x.tensor.metadata.dataPtr.toNat=v.base := by
    change x.tensor.dataPtr.toNat=v.base
    rw [← pointer]
    change v.base % 2^64 = v.base
    exact Nat.mod_eq_of_lt baseBound
  have bounds := x.laws.inBounds i hi
  have coords := x.laws.contiguous contiguous i hi
  rw [coords] at bounds
  have cap := x.tensor.storage.spanFits
  have vbase : v.base=x.tensor.storage.address x.tensor.storageOffset := by
    simp [v,VeriTile.Torch.FP32.dataView,VeriTile.Torch.FP32.DataView.base,Nat.ne_of_gt nonempty]
  rw [ptr,vbase,coords]
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

theorem metadata_address (x : MetadataObject) (contiguous : x.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i < x.metadata.numel) :
    x.metadata.dataPtr.toNat+4*i=x.storage.address (x.storageOffset+i) ∧
    x.metadata.dataPtr.toNat+4*i+4 ≤ x.storage.base+4*x.storage.elements ∧
    x.metadata.dataPtr.toNat+4*i+4 ≤ 2^64 := by
  have bounds := x.contiguous contiguous i hi
  have span := x.storage.spanFits
  have room : x.storage.address x.storageOffset < 2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  rw [x.pointer (by omega)]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt room,VeriTile.Torch.FP32.Storage.address]
  omega

theorem pointer64 (base index : Nat) (room : base+4*index+4 ≤ 2^64)
    (aligned : base%4=0) :
    (BitVec.ofNat 64 base+BitVec.ofNat 64 4*BitVec.ofNat 64 index).toNat=base+4*index ∧
    (base+4*index)%4=0 := by
  constructor
  · simp only [BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (show base+4*index < 2^64 by omega)]
  · omega

#axiomsClean checked_active_offsets
#axiomsClean checked_padded_offsets
#axiomsClean mask_exact
#axiomsClean payload_address
#axiomsClean metadata_address
#axiomsClean pointer64
end VeriTile.Bench.Repairs.SoftmaxReduceV.Checked

namespace VeriTile.Bench.Repairs.SoftmaxReduceV.Checked

theorem checked_accesses (c : Call) (h : MetadataContract c) (x : Host.Input α)
    (data : Content c x) (j : Host.Job) (hj : Host.Legal (host c) j)
    (k d : Nat) (hk : k < x.counts j.batch) (hd : d < dimensions c) :
    pageOffset (host c) x j k < c.locations.metadata.numel ∧
    logicAddress (host c) x j k < c.logics.metadata.numel ∧
    valueAddress (host c) x j k d < c.values.metadata.numel ∧
    Host.address (host c) j d < c.output.metadata.numel ∧
    j.batch < c.starts.metadata.numel ∧ j.batch < c.lengths.metadata.numel := by
  have bounds := active_addresses (host c) x (checked_admissible c h x data) j hj k d hk hd
  have extents := checked_extents c h
  rw [extents.locations,extents.logics,extents.values,extents.output,extents.starts,extents.lengths]
  exact ⟨bounds.1,bounds.2.1,bounds.2.2.1,bounds.2.2.2,hj.1,hj.1⟩

#axiomsClean checked_accesses
end VeriTile.Bench.Repairs.SoftmaxReduceV.Checked

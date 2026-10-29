import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBound
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelIndex
open VeriTile.Triton KernelProgram KernelTensor VeriTile.Torch.FP32
noncomputable section
set_option maxHeartbeats 12000000

def sourcePointer (b : Buffers kind s f) (r : RegionName) (off : Int) : BitVec 64 :=
  (b.tensor r).dataPtr+BitVec.ofInt 64 (4*off)
theorem source_pointer (b : Buffers kind s f) (a : Addr) (role : a.region∈KernelContract.roles kind f)
    (valid : Geometry.Valid s f a) :
    sourcePointer b (region a.region) (Geometry.offset s f a)=
      BitVec.ofNat 64 ((b.tensor (region a.region)).storage.address
        ((b.tensor (region a.region)).physicalIndex (Geometry.offset s f a))) := by
  have bound := KernelBound.coordinate_bound b a role valid
  rw [logical_address b (region a.region) _ bound]
  unfold sourcePointer
  rw [←b.pointer (region a.region) (by omega)]
  simp only [DataView.address,Nat.mul_zero,Nat.add_zero]
  rw [show (4*(Geometry.offset s f a : Int))=((4*Geometry.offset s f a : Nat):Int) by simp]
  rw [BitVec.ofInt_natCast,BitVec.ofNat_add]

/-- Signed i32 allocation-relative indices and the actual non-wrapping u64
pointer agree for every active coordinate used by a source load or store. -/
theorem active_exact (b : Buffers kind s f) (a : Addr) (role : a.region∈KernelContract.roles kind f)
    (valid : Geometry.Valid s f a) :
    (BitVec.ofInt 32 (Geometry.offset s f a)).toInt=Geometry.offset s f a ∧
      (sourcePointer b (region a.region) (Geometry.offset s f a)).toNat=
        (b.tensor (region a.region)).metadata.dataPtr.toNat+4*Geometry.offset s f a := by
  refine ⟨KernelBound.coordinate_i32 b a valid,?_⟩
  have bound := KernelBound.coordinate_bound b a role valid
  have nonempty : 0<(b.tensor (region a.region)).metadata.numel := by omega
  have span := (b.view (region a.region)).spanFits
  have count : (b.view (region a.region)).elements=(b.tensor (region a.region)).metadata.numel := rfl
  rw [count,b.base (region a.region) nonempty] at span
  have fits : (b.tensor (region a.region)).metadata.dataPtr.toNat+4*Geometry.offset s f a<2^64 := by omega
  rw [source_pointer b a role valid,logical_address b (region a.region) _ bound]
  rw [DataView.address,b.base (region a.region) nonempty]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]

theorem formed_i32 (b : Buffers kind s f) (a : Addr)
    (hp : a.plane<Geometry.planes s a.region)
    (hr : a.row≤Geometry.rows s a.region+Index.rowPadding s a.region)
    (hc : a.col<Geometry.cols s f a.region+s.BBV+s.BK) :
    (BitVec.ofInt 32 (Geometry.offset s f a)).toInt=Geometry.offset s f a :=
  Index.i32_exact _ (by omega) (Index.formed_source_i32 s f b.admitted.domain.toSafe a hp hr hc)

/-- The final unused reverse decrement is also an exact formed pointer.
The contract supplies the lower allocation address reserve. -/
theorem reverse_retreat (b : Buffers kind s f) (r : Region) (role : r∈KernelContract.roles kind f)
    (off stride : Nat) (inside : off<KernelContract.span s f r)
    (small : stride≤max (max s.K s.V) 1) :
    0≤((b.tensor (region r)).metadata.dataPtr.toNat:Int)+4*((off:Int)-stride) ∧
      ((b.tensor (region r)).metadata.dataPtr.toNat:Int)+4*((off:Int)-stride)<2^64 := by
  have low := (b.admitted.tensors r role).lower
  have high := (b.admitted.tensors r role).window
  dsimp only at low high
  have lowInt : (4*max (max s.K s.V) 1:Int)≤(b.tensor (region r)).metadata.dataPtr.toNat := by exact_mod_cast low
  have highInt : ((b.tensor (region r)).metadata.dataPtr.toNat:Int)+4*KernelContract.span s f r+4≤2^64 := by exact_mod_cast high
  constructor <;> omega

#axiomsClean source_pointer
#axiomsClean active_exact
#axiomsClean formed_i32
#axiomsClean reverse_retreat
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelIndex

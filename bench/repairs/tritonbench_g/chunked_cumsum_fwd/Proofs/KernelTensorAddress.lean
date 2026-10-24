import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorBindings

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
open VeriTile Triton
open Math

theorem access_span (t : Torch.FP32.Tensor) (laws : Torch.FP32.TensorLaws t)
    (index off : Nat) (hi : index<t.metadata.numel)
    (coord : t.physicalIndex index=t.storageOffset+off) :
    address t off+4≤t.storage.base+4*t.storage.elements ∧ address t off+4≤2^64 := by
  have he := tensor_address t laws index off hi coord
  have bound := laws.inBounds index hi
  have span := t.storage.spanFits
  unfold Torch.FP32.Storage.address at he
  omega

theorem native_address (t : Torch.FP32.Tensor) (off : Nat) (span : address t off+4≤2^64) :
    (Index.address t.dataPtr.toNat (Index.u off)).toNat=address t off := by
  unfold Index.address Index.u
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  change (t.dataPtr.toNat+off*4)%2^64=address t off
  rw [Nat.mod_eq_of_lt (by unfold address at span; omega)]
  simp [address,Nat.mul_comm]

theorem output_native_address (v : OutputView c) (off : Nat) (bound : off<c.outputSize) :
    (Index.address v.tensor.dataPtr.toNat (Index.u off)).toNat=address v.tensor off ∧
    address v.tensor off+4≤v.tensor.storage.base+4*v.tensor.storage.elements ∧
    address v.tensor off+4≤2^64 := by
  have h : off<v.tensor.metadata.numel := by simpa only [output_numel] using bound
  have span := access_span v.tensor v.laws off off h (v.laws.contiguous v.contiguous off h)
  exact ⟨native_address v.tensor off span.2,span⟩

theorem input_dt_native_address (a : Inputs c hasBias) (l : Launch) (hl : c.valid l)
    (i j : Nat) (active : activeInput c.layout l i j) :
    (Index.address a.dt.dataPtr.toNat (Index.dt c l i j)).toNat=address a.dt (dtAddress c.layout l i j) ∧
    address a.dt (dtAddress c.layout l i j)+4≤2^64 := by
  have start := chunk_start_lt c l hl
  have time : l.chunk*c.chunk+j<c.sequence := by
    change Math.head c.layout l i<c.heads ∧ j<min c.chunk (c.sequence-l.chunk*c.chunk) at active
    omega
  have he : dtAddress c.layout l i j=
      l.batch*c.dtBatch+(l.chunk*c.chunk+j)*c.dtTime+Math.head c.layout l i*c.dtHead := by
    simp [dtAddress,Geometry.layout]; ring
  have span := access_span a.dt a.dtLaws
    ((l.batch*c.sequence+(l.chunk*c.chunk+j))*c.heads+Math.head c.layout l i)
    (dtAddress c.layout l i j)
    (by rw [input_dt_numel]; exact dt_index_bound c _ _ _ hl.1 time active.1)
    (by rw [he]; exact a.dtCoordinates _ _ _ hl.1 time active.1)
  rw [dt_word]
  exact ⟨native_address a.dt _ span.2,span.2⟩

theorem packed_bound (c : Geometry) (b h k j : Nat)
    (hb : b<c.batches) (hh : h<c.heads) (hk : k<c.chunks) (hj : j<c.chunk) :
    packed c b h k j<c.outputSize := pair_bound (pair_bound (pair_bound hb hh) hk) hj

theorem input_rate_native_address (a : Inputs c hasBias) (l : Launch) (i : Nat)
    (active : Math.head c.layout l i<c.heads) :
    (Index.address a.rate.dataPtr.toNat (Index.rate c l i)).toNat=
      address a.rate (Math.head c.layout l i*c.rateHead) ∧
    address a.rate (Math.head c.layout l i*c.rateHead)+4≤2^64 := by
  have span := access_span a.rate a.rateLaws (Math.head c.layout l i)
    (Math.head c.layout l i*c.rateHead)
    (by simpa only [input_rate_numel] using active)
    (a.rateCoordinates _ active)
  have word : Index.rate c l i=Index.u (Math.head c.layout l i*c.rateHead) := by
    simp only [Index.rate,head_word,Index.u,← BitVec.ofNat_mul]
  rw [word]
  exact ⟨native_address a.rate _ span.2,span.2⟩

theorem input_bias_native_address (a : Inputs c hasBias) (t : Torch.FP32.Tensor)
    (present : t∈a.bias.toList) (l : Launch) (i : Nat)
    (active : Math.head c.layout l i<c.heads) :
    (Index.address t.dataPtr.toNat (Index.bias c l i)).toNat=
      address t (Math.head c.layout l i*c.biasHead) ∧
    address t (Math.head c.layout l i*c.biasHead)+4≤2^64 := by
  have span := access_span t (a.biasLaws t present) (Math.head c.layout l i)
    (Math.head c.layout l i*c.biasHead)
    (by simpa [Torch.TensorMetadata.numel,a.biasShape t present] using active)
    (a.biasCoordinates t present _ active)
  have word : Index.bias c l i=Index.u (Math.head c.layout l i*c.biasHead) := by
    simp only [Index.bias,head_word,Index.u,← BitVec.ofNat_mul]
  rw [word]
  exact ⟨native_address t _ span.2,span.2⟩

/-- Native addresses are required only for accesses enabled by the source
masks. Arithmetic for inactive pointer lanes may wrap without any access. -/
def Accesses (f : Buffers c hasBias) : Prop :=
  ∀ l∈c.grid,∀ (i : Fin c.BH) (j : Fin c.BC),
    ((Index.head c l i.val).ult (Index.u c.heads)=decide (Math.head c.layout l i.val<c.heads)) ∧
    ((Index.u j.val).ult (Index.u c.chunk)=decide (j.val<c.chunk)) ∧
    ((Index.u j.val).toNat < min (Index.u c.chunk).toNat (Index.remaining c l).toNat ↔
      j.val < min c.chunk (c.sequence-l.chunk*c.chunk)) ∧
    (activeInput c.layout l i.val j.val →
      (Index.address f.inputs.dt.dataPtr.toNat (Index.dt c l i.val j.val)).toNat=
        address f.inputs.dt (dtAddress c.layout l i.val j.val) ∧
      address f.inputs.dt (dtAddress c.layout l i.val j.val)+4≤2^64) ∧
    (Math.head c.layout l i.val<c.heads →
      ((Index.address f.inputs.rate.dataPtr.toNat (Index.rate c l i.val)).toNat=
        address f.inputs.rate (Math.head c.layout l i.val*c.rateHead) ∧
        address f.inputs.rate (Math.head c.layout l i.val*c.rateHead)+4≤2^64) ∧
      (∀ t∈f.inputs.bias.toList,
        (Index.address t.dataPtr.toNat (Index.bias c l i.val)).toNat=
          address t (Math.head c.layout l i.val*c.biasHead) ∧
        address t (Math.head c.layout l i.val*c.biasHead)+4≤2^64)) ∧
    (activeOutput c.layout l i.val j.val → ∀ r : Output,
      (Index.address (f.output r).tensor.dataPtr.toNat (Index.out c l i.val j.val)).toNat=
        address (f.output r).tensor (outAddress c.layout l i.val j.val) ∧
      address (f.output r).tensor (outAddress c.layout l i.val j.val)+4≤2^64)

theorem accesses (f : Buffers c hasBias) (sizes : Sizes c) : Accesses f := by
  intro l member i j
  have valid := (c.mem_grid l).1 member
  have hm := masks c sizes l valid i j
  refine ⟨hm.1,hm.2.1,hm.2.2,?_,?_,?_⟩
  · exact input_dt_native_address f.inputs l valid i.val j.val
  · intro active
    exact ⟨input_rate_native_address f.inputs l i.val active,
      fun t present=>input_bias_native_address f.inputs t present l i.val active⟩
  · intro active r
    have h := output_native_address (f.output r) (outAddress c.layout l i.val j.val)
      (output_bound c l valid i.val j.val active)
    rw [out_word]
    exact ⟨h.1,h.2.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor

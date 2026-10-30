import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Host
namespace VeriTile.Bench.Repairs.Ksoftmax.Pointers
open VeriTile.Triton
open VeriTile.Bench.Repairs

/-- Guarded i32 primitive widening, complete signed-i64 products/additions and
all read-mask comparisons. Formed tail pointers remain modular until active. -/
theorem indices (depth : Host.DepthLibrary) (x : TensorMetadata)
    (rank : x.shape.length=3) (extent : ExtentContract x)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x)))
    (nonempty : 0<columns x) (causal : Bool) :
    Native.widened32 b=(b.val : Int) ∧ Native.widened32 r=(r.val : Int) ∧
    Native.widened32 i=(i.val : Int) ∧
    Native.offset (rows x) (columns x) b r i=
      (Native.flat (rows x) (columns x) b r i : Int) ∧
    Native.readMask (columns x : Int) (r.val : Int) (i.val : Int) causal=
      (decide (i.val<columns x) && (!causal || decide (i.val≤r.val))) := by
  have d := depth.primitive (columns x) extent.2.2.1
  exact ⟨Native.widened32_exact _ (lt_trans b.isLt extent.1),
    Native.widened32_exact _ (lt_trans r.isLt extent.2.1),
    Native.widened32_exact _ (by omega),
    Native.offset_exact (batches x) (rows x) (columns x) _ b r i b.isLt r.isLt i.isLt
      (Native.depth_bounds _ _ nonempty (depth.contract _ nonempty)).2.2
      (by simpa only [count_three x rank] using extent.2.2.2) nonempty,
    Native.mask_exact _ _ _ causal⟩

/-- Parameter narrowing is exact, including every canonical row stride. -/
theorem strides (x : TensorMetadata) (rank : x.shape.length=3) (extent : ExtentContract x)
    (live : 0<x.numel) :
    VarLenCopy.signed64 (columns x : Int)=(columns x : Int) ∧
    VarLenCopy.signed64 ((rows x*columns x : Nat) : Int)=((rows x*columns x : Nat) : Int) := by
  have b : 0<batches x := by
    by_contra h
    have hz : batches x=0 := by omega
    rw [count_three x rank,hz] at live
    simp at live
  have le : rows x*columns x≤x.numel := by
    rw [count_three x rank]
    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right (rows x*columns x) (Nat.succ_le_of_lt b)
  exact ⟨VarLenCopy.signed64_exact _ (by omega) (by exact_mod_cast (show columns x<2^63 by have := extent.2.2.1;omega)),
    VarLenCopy.signed64_exact _ (by omega) (by exact_mod_cast (show rows x*columns x<2^63 by have := extent.2.2.2;omega))⟩

/-- Corrected contiguous qk/bk mask products include all formed tail lanes. -/
theorem mask_offset (R K D row lane : Nat) (hR : row<R) (hK : 0<K)
    (hD : D≤2*K) (hl : lane<D) (size : R*K<2^61) :
    Native.add (Native.mul row K) lane=(row*K+lane : Nat) := by
  simpa [Native.offset,Native.flat,Native.mul,Native.add,VarLenCopy.signed64] using
    Native.offset_exact R 1 K D row 0 lane hR (by decide) hl hD (by simpa using size) hK

theorem tensor_pointer (x : TensorMetadata) (j : Nat)
    (span : x.dataPtr.toNat+4*x.numel≤2^64) (active : j<x.numel) :
    (Native.pointer x.dataPtr.toNat j).toNat=x.dataPtr.toNat+4*j ∧
    x.dataPtr.toNat≤(Native.pointer x.dataPtr.toNat j).toNat ∧
    (Native.pointer x.dataPtr.toNat j).toNat+4≤x.dataPtr.toNat+4*x.numel :=
  Native.active_pointer x.dataPtr.toNat (4*x.numel) x.numel j (by omega) span active

theorem aligned (x : TensorMetadata) (h : x.dataPtr.toNat%4=0) (j : Nat) :
    (x.dataPtr.toNat+4*j)%4=0 := by omega

theorem separate_bytes (dst input : TensorMetadata) (h : PairContract dst input)
    (i j a b : Nat) (hi : i<dst.numel) (hj : j < input.numel) (ha : a<4) (hb : b<4) :
    dst.dataPtr.toNat+4*i+a≠input.dataPtr.toNat+4*j+b := by
  rcases h.2 with h|h <;> omega

/-- Admission provides the active qk/bk read bounds using the corrected row
stride, including when batch and query dimensions differ. -/
theorem forward_reads (y x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String)
    (admitted : ForwardContract y x mask mode)
    (b : Fin (batches x)) (r : Fin (rows x)) (j : Nat) (hj : j<columns x) :
    Native.flat (rows x) (columns x) b r j < x.numel ∧
    (∀ m : TensorMetadata, mode≠none → mask=.tensor m →
      VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.maskAddress
        (Host.decodeMode mode) b r (columns x) j < m.numel) := by
  rcases admitted with ⟨_,_,rank,_,_,known,hm,_⟩
  refine ⟨by rw [count_three x rank];exact Native.active_bound b.isLt r.isLt hj,?_⟩
  intro m present eq
  subst mask
  rcases known with h|h|h
  · exact False.elim (present h)
  · subst mode
    have hs : m.shape=[rows x,columns x] := by
      have hh : TensorContract m ∧ m.device=x.device ∧ m.shape=[rows x,columns x] := by
        simpa [MaskContract] using hm
      exact hh.2.2
    simpa [VeriTile.Torch.TensorMetadata.numel,hs,Host.decodeMode,
      VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.maskAddress] using
      Native.mask_address_bound (rows x) (columns x) r j r.isLt hj
  · subst mode
    have hs : m.shape=[batches x,columns x] := by
      have hh : TensorContract m ∧ m.device=x.device ∧ m.shape=[batches x,columns x] := by
        simpa [MaskContract] using hm
      exact hh.2.2
    simpa [VeriTile.Torch.TensorMetadata.numel,hs,Host.decodeMode,
      VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.maskAddress] using
      Native.mask_address_bound (batches x) (columns x) b j b.isLt hj

#axiomsClean forward_reads
#axiomsClean indices
#axiomsClean strides
#axiomsClean mask_offset
#axiomsClean tensor_pointer
#axiomsClean aligned
#axiomsClean separate_bytes
end VeriTile.Bench.Repairs.Ksoftmax.Pointers

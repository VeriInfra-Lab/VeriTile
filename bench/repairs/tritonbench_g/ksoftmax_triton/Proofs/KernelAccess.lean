import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelTensor
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.Ksoftmax.KernelAccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def matrixPointer (metadata : TensorMetadata) (Q K batch row lane : Nat) : BitVec 64 :=
  Native.pointer metadata.dataPtr.toNat
    (Native.add
      (Native.add
        (Native.mul (Native.widened32 batch) (VeriTile.Bench.Repairs.VarLenCopy.signed64 ((Q*K : Nat) : Int)))
        (Native.mul (Native.widened32 row) (VeriTile.Bench.Repairs.VarLenCopy.signed64 (K : Int))))
      (Native.widened32 lane))

/-- Every widened matrix access, including its original Tensor's arbitrary
harmless singleton strides and storage offset, has the same actual byte address.
This applies to source loads and stores, not merely a logical flat index. -/
theorem matrix_address (depth : Host.DepthLibrary) (x metadata : TensorMetadata)
    (rank : x.shape.length=3) (extent : ExtentContract x) (sameCount : metadata.numel=x.numel)
    (guard : TensorContract metadata) (view : KernelTensor.View metadata)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x) :
    (matrixPointer metadata (rows x) (columns x) b r i).toNat=
      KernelTensor.address metadata view (Native.flat (rows x) (columns x) b r i) := by
  have bound : Native.flat (rows x) (columns x) b r i<metadata.numel := by
    rw [sameCount,count_three x rank]
    exact Native.active_bound b.isLt r.isLt hi
  rw [KernelTensor.address_flat metadata view guard _ bound]
  have indices := Pointers.indices depth x rank extent b r i (by omega) Bool.false
  have live : 0<x.numel := by rw [←sameCount]; omega
  have strides := Pointers.strides x rank extent live
  unfold matrixPointer
  rw [indices.1,indices.2.1,indices.2.2.1,strides.1,strides.2]
  change (Native.pointer metadata.dataPtr.toNat (Native.offset (rows x) (columns x) b r i)).toNat=_
  rw [indices.2.2.2.1]
  exact (Pointers.tensor_pointer metadata _ (KernelTensor.span64 metadata view (by omega)) bound).1

theorem matrix_load (depth : Host.DepthLibrary) (x metadata : TensorMetadata)
    (rank : x.shape.length=3) (extent : ExtentContract x) (sameCount : metadata.numel=x.numel)
    (guard : TensorContract metadata) (view : KernelTensor.View metadata) (memory : Nat → Graph.Word)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x) :
    memory ((matrixPointer metadata (rows x) (columns x) b r i).toNat)=
      KernelTensor.logical memory metadata view (Native.flat (rows x) (columns x) b r i) := by
  rw [matrix_address depth x metadata rank extent sameCount guard view b r i hi]
  rfl

/-- qk and bk use exactly the same widened row*K+lane expression; only the
source-selected row differs. Active mask accesses retain all four bytes. -/
theorem mask_address (metadata : TensorMetadata) (view : KernelTensor.View metadata)
    (guard : TensorContract metadata) (R K D row lane : Nat) (count : metadata.numel=R*K)
    (hr : row<R) (hl : lane<K) (hd : lane<D) (depth : D≤2*K) (size : R*K<2^61) :
    (Native.pointer metadata.dataPtr.toNat (Native.add (Native.mul row K) lane)).toNat=
      KernelTensor.address metadata view (row*K+lane) := by
  have bound : row*K+lane<metadata.numel := by rw [count]; exact Native.mask_address_bound R K row lane hr hl
  rw [KernelTensor.address_flat metadata view guard _ bound,
    Pointers.mask_offset R K D row lane hr (by omega) depth hd size]
  exact (Pointers.tensor_pointer metadata _ (KernelTensor.span64 metadata view (by omega)) bound).1

theorem forward_matrices (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (mode : Option String) (admitted : ForwardContract y x mask mode)
    (yView : KernelTensor.View y) (xView : KernelTensor.View x)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x) :
    (matrixPointer y (rows x) (columns x) b r i).toNat=
      KernelTensor.address y yView (Native.flat (rows x) (columns x) b r i) ∧
    (matrixPointer x (rows x) (columns x) b r i).toNat=
      KernelTensor.address x xView (Native.flat (rows x) (columns x) b r i) := by
  have live : 0<x.numel := by
    rw [count_three x admitted.2.2.1]
    have bound := Native.active_bound b.isLt r.isLt hi
    omega
  have extent := (admitted.2.2.2.2.2.2.2 live).1
  have count : y.numel=x.numel := by change y.shape.prod=x.shape.prod; rw [admitted.2.2.2.1]
  exact ⟨matrix_address depth x y admitted.2.2.1 extent count admitted.1 yView b r i hi,
    matrix_address depth x x admitted.2.2.1 extent rfl admitted.2.1 xView b r i hi⟩

theorem backward_matrices (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (admitted : BackwardContract dst grad saved)
    (dstView : KernelTensor.View dst) (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))) (hi : i.val<columns saved) :
    (matrixPointer dst (rows saved) (columns saved) b r i).toNat=
      KernelTensor.address dst dstView (Native.flat (rows saved) (columns saved) b r i) ∧
    (matrixPointer grad (rows saved) (columns saved) b r i).toNat=
      KernelTensor.address grad gradView (Native.flat (rows saved) (columns saved) b r i) ∧
    (matrixPointer saved (rows saved) (columns saved) b r i).toNat=
      KernelTensor.address saved savedView (Native.flat (rows saved) (columns saved) b r i) := by
  have rank := admitted.2.2.2.1
  have live : 0<saved.numel := by
    rw [count_three saved rank]
    have bound := Native.active_bound b.isLt r.isLt hi
    omega
  have extent := (admitted.2.2.2.2.2.2.2.2 live).1
  have dstCount : dst.numel=saved.numel := by change dst.shape.prod=saved.shape.prod; rw [admitted.2.2.2.2.1]
  have gradCount : grad.numel=saved.numel := by change grad.shape.prod=saved.shape.prod; rw [admitted.2.2.2.2.2.1]
  exact ⟨matrix_address depth saved dst rank extent dstCount admitted.1 dstView b r i hi,
    matrix_address depth saved grad rank extent gradCount admitted.2.1 gradView b r i hi,
    matrix_address depth saved saved rank extent rfl admitted.2.2.1 savedView b r i hi⟩

def maskPointer (metadata : TensorMetadata) (mode : Forward.MaskMode) (batch row K lane : Nat) : BitVec 64 :=
  Native.pointer metadata.dataPtr.toNat
    (Native.add (Native.mul (if mode=.qk then Native.widened32 row else Native.widened32 batch)
      (VeriTile.Bench.Repairs.VarLenCopy.signed64 (K : Int))) (Native.widened32 lane))

theorem forward_mask (depth : Host.DepthLibrary) (y x : TensorMetadata)
    (mask : ArgumentMetadata) (mode : Option String) (admitted : ForwardContract y x mask mode)
    (view : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x)))
    (hi : i.val<columns x) (present : (Host.decodeMode mode).present=Bool.true) :
    (maskPointer (KernelInputs.maskTensor x mask mode) (Host.decodeMode mode) b r (columns x) i).toNat=
      KernelTensor.address (KernelInputs.maskTensor x mask mode) view
        (Forward.maskAddress (Host.decodeMode mode) b r (columns x) i) := by
  have rank := admitted.2.2.1
  have live : 0<x.numel := by
    rw [count_three x rank]
    have bound := Native.active_bound b.isLt r.isLt hi
    omega
  have extent := (admitted.2.2.2.2.2.2.2 live).1
  have elementBound := extent.2.2.2
  rw [count_three x rank] at elementBound
  have hb : 1≤batches x := by have := b.isLt; omega
  have hr : 1≤rows x := by have := r.isLt; omega
  have qsize : rows x*columns x<2^61 := by
    have h := Nat.mul_le_mul_right (rows x*columns x) hb
    nlinarith
  have bsize : batches x*columns x<2^61 := by
    have h := Nat.mul_le_mul_left (batches x*columns x) hr
    nlinarith
  have hd := (Native.depth_bounds (columns x) _ (by omega) (depth.contract _ (by omega))).2.2
  have widened := Pointers.indices depth x rank extent b r i (by omega) Bool.false
  have stride := (Pointers.strides x rank extent live).1
  simp only [maskPointer,widened.1,widened.2.1,widened.2.2.1,stride]
  have hm := admitted.2.2.2.2.2.2.1
  rcases admitted.2.2.2.2.2.1 with modeEq|modeEq|modeEq
  · subst mode
    simp [Host.decodeMode,Forward.MaskMode.present] at present
  · subst mode
    cases mask with
    | nonTensor => simp [MaskContract] at hm
    | tensor m =>
      have h : TensorContract m ∧ m.device=x.device ∧ m.shape=[rows x,columns x] := by
        simpa [MaskContract] using hm
      have count : m.numel=rows x*columns x := by simp [VeriTile.Torch.TensorMetadata.numel,h.2.2]
      simpa [KernelInputs.maskTensor,Host.decodeMode,Forward.maskAddress] using
        mask_address m view h.1 (rows x) (columns x) _ r i count r.isLt hi i.isLt hd qsize
  · subst mode
    cases mask with
    | nonTensor => simp [MaskContract] at hm
    | tensor m =>
      have h : TensorContract m ∧ m.device=x.device ∧ m.shape=[batches x,columns x] := by
        simpa [MaskContract] using hm
      have count : m.numel=batches x*columns x := by simp [VeriTile.Torch.TensorMetadata.numel,h.2.2]
      simpa [KernelInputs.maskTensor,Host.decodeMode,Forward.maskAddress] using
        mask_address m view h.1 (batches x) (columns x) _ b i count b.isLt hi i.isLt hd bsize

#axiomsClean matrix_address
#axiomsClean matrix_load
#axiomsClean mask_address
#axiomsClean forward_matrices
#axiomsClean backward_matrices
#axiomsClean forward_mask
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelAccess

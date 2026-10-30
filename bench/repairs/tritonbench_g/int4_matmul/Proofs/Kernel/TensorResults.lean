import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.GridProof
import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectMatmulNative

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource Direct Host Physical
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

@[simp] theorem request_shape (r : MatmulRequest) : r.arguments.shape=r.shape := rfl

/-- Output descriptor at the kernel boundary. The original wrapper either
checks the supplied output or allocates this ordinary independent tensor. -/
structure OutputBinding (r : MatmulRequest) (out : TensorInfo) : Prop where
  descriptor : OutputContract r.input r.weight out
  device : out.descriptor.device=r.input.descriptor.device
  independent : ∀ input∈r.views,input.descriptor.allocation≠out.descriptor.allocation

/-- Every supplied-output premise is already enforced by the existing guard. -/
theorem supplied_output (r : MatmulRequest) (out : TensorInfo) (ho : r.output=some out) :
    OutputBinding r out := by
  refine ⟨r.contract.output out ho,?_,?_⟩
  · exact r.contract.device out (by simp [matrixInputs,ho]) r.input (by simp [matrixInputs])
  · intro input hi
    have hn := r.contract.independent
    simp only [Independent,matrixInputs,ho,Option.toList_some,List.map_append,List.nodup_append] at hn
    exact hn.2.2 _ (List.mem_map.mpr ⟨input,hi,rfl⟩) _ (by simp)

/-- Fresh-output premises follow from the retained ordinary Tensor allocation
interface; no allocator implementation is added to the kernel proof. -/
theorem allocated_output (r : MatmulRequest) (initial : HeapState α) (live : Nat → Prop)
    (inputs : ∀ input∈r.views,live input.descriptor.allocation) (alloc : OutputAllocation r initial live) :
    OutputBinding r alloc.tensor :=
  ⟨⟨alloc.descriptor,alloc.shape⟩,alloc.device,alloc.separate r initial live inputs⟩

theorem output_logical (r : MatmulRequest) (out : TensorInfo) (h : OutputBinding r out) (row col : Nat) :
    logicalIndex out row col=cAddress r.shape row col := by
  simp only [logicalIndex,Tensor.elementOffset,h.descriptor.tensor.offset,
    h.descriptor.tensor.strides.1,h.descriptor.tensor.strides.2,
    h.descriptor.shape.2,Nat.zero_add,Nat.mul_one]
  rfl

def tensorState [Inhabited α] (r : MatmulRequest) (out : TensorInfo) (tile : Tile)
    (o : Ops α) (rest : Algebra α) (initial : HeapState α) (scratch : RegisterState α) (lanes : List Lane) :=
  launch r.arguments tile o rest (matrixState r out initial) scratch lanes

/-- Interpret the kernel's final typed output cells at the actual Tensor's
allocation. All other storage IDs retain their original contents. -/
def tensorHeap [Inhabited α] (out : TensorInfo) (initial : HeapState α) (final : RegisterState α) : HeapState α :=
  ⟨(fun storage i => if storage=out.descriptor.allocation then (final.mem "out" i).read .real
    else initial.values storage i),initial.packed⟩

def realKernel (r : MatmulRequest) (out : TensorInfo) (tile : Tile) (rest : Algebra ℝ)
    (initial : HeapState ℝ) (scratch : RegisterState ℝ) (lanes : List Lane) :=
  tensorHeap out initial (tensorState r out tile realOps rest initial scratch lanes)

theorem tensor_frame [Inhabited α] (r : MatmulRequest) (out : TensorInfo) (tile : Tile)
    (member : tile∈sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial : HeapState α) (scratch : RegisterState α) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes r.shape tile)) (storage i : Nat)
    (outside : storage ≠ out.descriptor.allocation ∨ r.shape.m * r.shape.n ≤ i) :
    (tensorHeap out initial (tensorState r out tile o rest initial scratch lanes)).values storage i=
      initial.values storage i := by
  by_cases hs : storage=out.descriptor.allocation
  · subst storage
    have hi := outside.resolve_left (by simp)
    have h := launch_frame r.arguments r.admitted tile member o rest (matrixState r out initial)
      scratch lanes complete "out" i (Or.inr hi)
    simp only [tensorHeap,ite_true,tensorState]
    rw [h]
    simp [attach,memory,matrixState]
  · simp [tensorHeap,hs]

theorem real_tensor_result (r : MatmulRequest) (out : TensorInfo) (output : OutputBinding r out)
    (tile : Tile) (member : tile∈sourceTiles) (rest : Algebra ℝ)
    (initial : HeapState ℝ) (scratch : RegisterState ℝ) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes r.shape tile)) :
    (∀ row col,row<r.shape.m → col<r.shape.n →
      (realKernel r out tile rest initial scratch lanes).values out.descriptor.allocation (logicalIndex out row col)=
        realTarget r out initial row col) ∧
    (∀ storage i,storage ≠ out.descriptor.allocation ∨ r.shape.m * r.shape.n ≤ i →
      (realKernel r out tile rest initial scratch lanes).values storage i=initial.values storage i) ∧
    (∀ input∈r.views,∀ i,(realKernel r out tile rest initial scratch lanes).values input.descriptor.allocation i=
      initial.values input.descriptor.allocation i) ∧
    (realKernel r out tile rest initial scratch lanes).packed=initial.packed := by
  refine ⟨?_,?_,?_,rfl⟩
  · intro row col hr hc
    simp only [realKernel,tensorHeap,ite_true,output_logical r out output]
    have h := launch_real r.arguments r.admitted tile member rest (matrixState r out initial)
      scratch lanes complete row col hr hc
    simp only [request_shape] at h
    change ((launch r.arguments tile realOps rest (matrixState r out initial) scratch lanes).mem "out"
      (cAddress r.shape row col)).read .real=_
    rw [h,Cell.read_same,matrixState_original]
    rfl
  · exact tensor_frame r out tile member realOps rest initial scratch lanes complete
  · intro input hi i
    exact tensor_frame r out tile member realOps rest initial scratch lanes complete _ i
      (Or.inl (output.independent input hi))

/-- Four-byte FP32 storage interpretation of the actual final output words.
This is ordinary typed-store serialization; it preserves all original bytes
outside the written output by the proved frame and encode/decode inverse. -/
def tensorBytes (out : TensorInfo) (initial : ByteHeap) (final : RegisterState Word) : ByteHeap :=
  fun storage b => if storage=out.descriptor.allocation then
    encode ((final.mem "out" (b/4)).read .real) ⟨b%4,Nat.mod_lt _ (by decide)⟩ else initial storage b

def nativeKernel (r : MatmulRequest) (out : TensorInfo) (tile : Tile)
    (instructions : Native.Instructions) (isa : Native.ISA) (rest : Algebra Word)
    (initial : ByteHeap) (scratch : RegisterState Word) (lanes : List Lane) : ByteHeap :=
  tensorBytes out initial (tensorState r out tile (Native.ops instructions isa) rest (heapWords initial) scratch lanes)

def nativeTarget (r : MatmulRequest) (out : TensorInfo) (tile : Tile)
    (instructions : Native.Instructions) (isa : Native.ISA) (initial : ByteHeap)
    (lanes : List Lane) (row col : Nat) :=
  referenceOutput r.shape tile (Native.ops instructions isa) (originalMatrixState r out (heapWords initial))
    row col (laneSchedule r.shape tile lanes) 0

theorem native_tensor_result (base : Nat → Nat) (r : MatmulRequest) (out : TensorInfo)
    (output : OutputBinding r out) (inputs : ∀ input∈r.views,TensorLaws base input) (outLaws : TensorLaws base out)
    (tile : Tile) (member : tile∈sourceTiles) (instructions : Native.Instructions)
    (isa : Native.ISA) (rest : Algebra Word) (initial : ByteHeap) (scratch : RegisterState Word)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes r.shape tile)) :
    (∀ input∈r.views,PointerBinding base input) ∧ PointerBinding base out ∧
    (∀ row col,row<r.shape.m → col<r.shape.n → ∀ byte : Fin 4,
      nativeKernel r out tile instructions isa rest initial scratch lanes out.descriptor.allocation
        (4*logicalIndex out row col+byte.val)=
      encode (nativeTarget r out tile instructions isa initial lanes row col) byte) ∧
    (∀ storage b,storage≠out.descriptor.allocation ∨ r.shape.m*r.shape.n≤b/4 →
      nativeKernel r out tile instructions isa rest initial scratch lanes storage b=initial storage b) := by
  refine ⟨fun input hi => pointer_binding base input (inputs input hi),pointer_binding base out outLaws,?_,?_⟩
  · intro row col hr hc byte
    have hd : (4*cAddress r.shape row col+byte.val)/4=cAddress r.shape row col := by omega
    have hm : (4*cAddress r.shape row col+byte.val)%4=byte.val := by omega
    simp only [nativeKernel,tensorBytes,ite_true,output_logical r out output,hd,hm]
    have h := launch_value r.arguments r.admitted tile member (Native.ops instructions isa) rest
      (matrixState r out (heapWords initial)) scratch lanes complete row col hr hc
    simp only [request_shape] at h
    dsimp only [tensorState]
    rw [h,Cell.read_same,matrixState_original]
    rfl
  · intro storage b outside
    by_cases hs : storage=out.descriptor.allocation
    · subst storage
      have hi := outside.resolve_left (by simp)
      have h := launch_frame r.arguments r.admitted tile member (Native.ops instructions isa) rest
        (matrixState r out (heapWords initial)) scratch lanes complete "out" (b/4) (Or.inr hi)
      simp only [nativeKernel,tensorBytes,ite_true,tensorState]
      rw [h]
      simp only [attach,memory,matrixState,heapWords,wordHeap,Cell.read_same,↓reduceIte]
      have hb := congrFun (encode_decode (fun j : Fin 4 => initial out.descriptor.allocation (4*(b/4)+j.val)))
        (⟨b%4,Nat.mod_lt _ (by decide)⟩ : Fin 4)
      simpa [show 4*(b/4)+b%4=b by omega] using hb
    · simp [nativeKernel,tensorBytes,hs]

#axiomsClean real_tensor_result
#axiomsClean native_tensor_result
#axiomsClean supplied_output
#axiomsClean allocated_output
#specNonCircular realTarget avoiding [launch,parallel,computed,realKernel]
#specNonCircular nativeTarget avoiding [launch,parallel,computed,nativeKernel,partialValue]
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution

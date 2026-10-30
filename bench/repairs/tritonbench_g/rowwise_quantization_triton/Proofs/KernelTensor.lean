import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelTensor
open VeriTile.Triton
open KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 2500000

/-- Ordinary Tensor allocation observations, including backing storage and
logical indexing. This is task-local output binding, not allocator execution. -/
structure OutputTensor where
  object : Nat
  metadata : TensorMetadata
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure OutputLaws (t : OutputTensor) (bytes : Nat) (dtype : VeriTile.Torch.DType)
    (shape : List Nat) (device : VeriTile.Torch.Device) : Prop where
  dtype_eq : t.metadata.dtype = dtype
  shape_eq : t.metadata.shape = shape
  device_eq : t.metadata.device = device
  layout : t.metadata.layout = .strided
  contiguous : t.metadata.isContiguous = Bool.true
  noNeg : t.metadata.isNeg = Bool.false
  indices : ∀ i, i < t.metadata.numel → t.physicalIndex i = t.storageOffset + i
  inBounds : ∀ i, i < t.metadata.numel → t.physicalIndex i < t.storageElements
  spanFits : t.storageBase + bytes*t.storageElements ≤ 2^64
  pointer : 0 < t.metadata.numel →
    t.metadata.dataPtr = BitVec.ofNat 64 (t.storageBase + bytes*t.storageOffset)
  alignment : 0 < t.metadata.numel → t.metadata.dataPtr.toNat % bytes = 0

def base (t : OutputTensor) (bytes : Nat) : Nat :=
  if t.metadata.numel = 0 then 0 else t.storageBase + bytes*t.storageOffset

def address (t : OutputTensor) (bytes i : Nat) : Nat :=
  t.storageBase + bytes*t.physicalIndex i

theorem direct_address (t : OutputTensor) (bytes : Nat) (dtype : VeriTile.Torch.DType)
    (shape : List Nat) (device : VeriTile.Torch.Device)
    (laws : OutputLaws t bytes dtype shape device) (i : Nat) (hi : i < t.metadata.numel) :
    address t bytes i = base t bytes + bytes*i := by
  simp [address, base, show t.metadata.numel ≠ 0 by omega, laws.indices i hi, Nat.mul_add, Nat.add_assoc]

theorem span (t : OutputTensor) (bytes : Nat) (dtype : VeriTile.Torch.DType)
    (shape : List Nat) (device : VeriTile.Torch.Device)
    (laws : OutputLaws t bytes dtype shape device) : base t bytes + bytes*t.metadata.numel ≤ 2^64 := by
  by_cases hn : t.metadata.numel = 0
  · simp [base, hn]
  have hi : t.metadata.numel - 1 < t.metadata.numel := by omega
  have hb := laws.inBounds _ hi
  rw [laws.indices _ hi] at hb
  have cap : t.storageOffset + t.metadata.numel ≤ t.storageElements := by omega
  have hmul := Nat.mul_le_mul_left bytes cap
  have hs := laws.spanFits
  simp only [base, hn, if_false]
  nlinarith

theorem actual_pointer (t : OutputTensor) (bytes : Nat) (dtype : VeriTile.Torch.DType)
    (shape : List Nat) (device : VeriTile.Torch.Device)
    (laws : OutputLaws t bytes dtype shape device) (hn : 0 < t.metadata.numel) :
    BitVec.ofNat 64 (base t bytes) = t.metadata.dataPtr := by
  simpa [base, show t.metadata.numel ≠ 0 by omega] using (laws.pointer hn).symm

def values (x : Direct.Tensor) (laws : Direct.TensorLaws x) (heap : Heap) (r c : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue (load32 heap) x laws (r*columns x.metadata+c)

def initial (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (heap : Heap) (scratch : FP.Structural.State Word) : FP.Structural.State Word :=
  { scratch with mem := fun region i =>
      if region = "x" then
        FP.Structural.Cell.mk .real (load32 heap ((Direct.view x laws h).base + 4*i))
      else scratch.mem region i }

def layout (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (q m : OutputTensor) : Layout := ⟨(Direct.view x laws h).base, base q 1, base m 2⟩

theorem input_read (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (heap : Heap) (r c : Nat) (hr : r < rows x.metadata) (hc : c < columns x.metadata) :
    load32 heap ((Direct.view x laws h).base + 4*(r*columns x.metadata+c)) = values x laws heap r c := by
  have hi : r*columns x.metadata+c < x.metadata.numel := by
    rw [numel_eq x.metadata h]
    exact linear_bound _ _ r c hr hc
  exact VeriTile.Torch.FP32.direct_read_eq_logical (load32 heap) x laws h.contiguous h.noNeg _ hi

theorem initial_reads (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (heap : Heap) (scratch : FP.Structural.State Word) :
    KernelWhole.Reads (Direct.launch x.metadata h tree cover) (values x laws heap)
      (initial x laws h heap scratch) := by
  intro r c hr hc
  simpa [initial, Direct.launch, FP.Structural.Cell.read] using input_read x laws h heap r c hr hc

/-- The existing byte-store model is retained as the storage lowering of the
new source execution proof. Its inputs are the original pointer-bound Tensor;
no copy, normalization or allocation execution occurs in this definition. -/
def byteExecution (a : Arithmetic) (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover)) (q m : OutputTensor)
    (heap : Heap) : Heap :=
  runGrid (operations a) (Direct.launch x.metadata h tree cover) (layout x laws h q m) schedule.order heap

/-- Actual logical output coordinates use each Tensor's physicalIndex and
storage offset. The maximum's two bytes encode its FP16 payload directly. -/
def ByteResult (a : Arithmetic) (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (q m : OutputTensor) (before after : Heap) : Prop :=
  (∀ r c, r < rows x.metadata → c < columns x.metadata →
    after (address q 1 (r*columns x.metadata+c)) =
      quantized a (Direct.launch x.metadata h tree cover) (values x laws before r) c) ∧
  (∀ r, r < rows x.metadata → ∀ byte : Fin 2,
    after (address m 2 r + byte.val) =
      halfByte (storedMaximum a (Direct.launch x.metadata h tree cover) (values x laws before r)) byte) ∧
  (∀ addr, ¬ OutputCell (Direct.launch x.metadata h tree cover) (layout x laws h q m) addr →
    after addr = before addr)

theorem byte_correct (a : Arithmetic) (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover)) (q m : OutputTensor)
    (ql : OutputLaws q 1 (.other "int8") [rows x.metadata, columns x.metadata] x.metadata.device)
    (ml : OutputLaws m 2 (.other "float16") [rows x.metadata] x.metadata.device)
    (sep : Separation (Direct.launch x.metadata h tree cover) (layout x laws h q m)) (heap : Heap) :
    ByteResult a x laws h tree cover q m heap (byteExecution a x laws h tree cover schedule q m heap) := by
  let p := Direct.launch x.metadata h tree cover
  let v := layout x laws h q m
  have nq : q.metadata.numel = p.rows*p.columns := by simp [VeriTile.Torch.TensorMetadata.numel, ql.shape_eq, p, Direct.launch]
  have nm : m.metadata.numel = p.rows := by simp [VeriTile.Torch.TensorMetadata.numel, ml.shape_eq, p, Direct.launch]
  obtain ⟨hq, hm⟩ := runGrid_readback (operations a) p v sep schedule heap
  have hmax : ∀ r, r < p.rows →
      rowMaximum (operations a) p (fun c => load32 heap (inputAddress p v r c)) =
        rowMaximum (operations a) p (values x laws heap r) := by
    intro r hr
    apply rowMaximum_congr
    intro c hc
    exact input_read x laws h heap r c hr hc
  refine ⟨?_, ?_, ?_⟩
  · intro r c hr hc
    have hi : r*columns x.metadata+c < q.metadata.numel := by
      rw [nq]
      exact linear_bound _ _ _ _ hr hc
    rw [direct_address q 1 _ _ _ ql _ hi]
    have result := hq r hr c hc
    have hx : load32 heap (inputAddress p v r c) = values x laws heap r c :=
      input_read x laws h heap r c hr hc
    unfold qValue at result
    rw [hmax r hr, hx] at result
    simpa [byteExecution, outputAddress, layout, p, v, Direct.launch, rowQuantized, rowScaled,
      operations, quantized, maximum, rowMaximum, Nat.add_assoc] using result
  · intro r hr byte
    have hi : r < m.metadata.numel := by rw [nm]; exact hr
    rw [direct_address m 2 _ _ _ ml _ hi]
    have result := hm r hr byte
    unfold mValue at result
    rw [hmax r hr] at result
    simpa [byteExecution, maximumAddress, layout, p, v, Direct.launch,
      operations, storedMaximum, maximum, rowMaximum, Nat.add_assoc] using result
  · intro addr outside
    exact runGrid_frame (operations a) p v schedule.order
      (fun r hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)) heap addr outside

/-- FP32 loads use the same checked observed pointer as Python, including
nonzero storage offsets, and all four bytes lie inside the finite address span. -/
theorem input_pointers (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (r c : Nat)
    (hr : r < rows x.metadata) (hc : c < columns x.metadata) :
    (x.dataPtr + Native.nativeOffset r (columns x.metadata) c * BitVec.ofNat 64 4).toNat =
      (Direct.view x laws h).base + 4*(r*columns x.metadata+c) ∧
    (Direct.view x laws h).base % 4 = 0 ∧
    (Direct.view x laws h).base + 4*(r*columns x.metadata+c) + 3 < 2^64 := by
  obtain ⟨aligned, bound⟩ := Native.input_bounds x laws h r c 3 hr hc (by decide)
  have hn : 0 < x.metadata.numel := by
    rw [numel_eq x.metadata h]
    have := linear_bound _ _ r c hr hc
    omega
  obtain ⟨rb, cb, _⟩ := native_indices x.metadata h r c hr (lt_of_lt_of_le hc (padded_covers x.metadata))
  refine ⟨?_, aligned, bound⟩
  rw [←Direct.actual_pointer x laws h hn]
  simpa [Native.pointer, Nat.mul_comm] using
    Native.pointer_exact (Direct.view x laws h).base 4 r (columns x.metadata) c rb cb (by omega)

#axiomsClean input_pointers
/-- Actual observed data pointers, finite integer indices, and both output
Tensor storage addresses agree on every active lane. Empty grids are vacuous. -/
theorem output_pointers (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (q m : OutputTensor)
    (ql : OutputLaws q 1 (.other "int8") [rows x.metadata, columns x.metadata] x.metadata.device)
    (ml : OutputLaws m 2 (.other "float16") [rows x.metadata] x.metadata.device)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < columns x.metadata) :
    (q.metadata.dataPtr + Native.nativeOffset r (columns x.metadata) c * BitVec.ofNat 64 1).toNat =
      address q 1 (r*columns x.metadata+c) ∧
    (m.metadata.dataPtr + (BitVec.ofNat 32 r).signExtend 64 * BitVec.ofNat 64 2).toNat =
      address m 2 r ∧
    address q 1 (r*columns x.metadata+c) < 2^64 ∧ address m 2 r + 1 < 2^64 ∧
    address m 2 r % 2 = 0 := by
  have nq : q.metadata.numel = rows x.metadata*columns x.metadata := by
    simp [VeriTile.Torch.TensorMetadata.numel, ql.shape_eq]
  have nm : m.metadata.numel = rows x.metadata := by
    simp [VeriTile.Torch.TensorMetadata.numel, ml.shape_eq]
  have qi : r*columns x.metadata+c < q.metadata.numel := by
    rw [nq]
    exact linear_bound _ _ _ _ hr hc
  have mi : r < m.metadata.numel := by omega
  have qb := span q 1 _ _ _ ql
  have mb := span m 2 _ _ _ ml
  have qa : address q 1 (r*columns x.metadata+c) = base q 1 + (r*columns x.metadata+c) := by
    simpa using direct_address q 1 _ _ _ ql _ qi
  have ma : address m 2 r = base m 2 + 2*r := direct_address m 2 _ _ _ ml _ mi
  have qbound : base q 1 + (r*columns x.metadata+c) < 2^64 := by omega
  have mbound : base m 2 + 2*r + 1 < 2^64 := by omega
  obtain ⟨rb, cb, _⟩ := native_indices x.metadata h r c hr (lt_of_lt_of_le hc (padded_covers x.metadata))
  have mAlignment : address m 2 r % 2 = 0 := by
    have aligned := ml.alignment (by omega : 0 < m.metadata.numel)
    rw [←actual_pointer m 2 _ _ _ ml (by omega), BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (by omega : base m 2 < 2^64)] at aligned
    simpa [ma, Nat.add_mod] using aligned
  refine ⟨?_, ?_, by omega, by omega, mAlignment⟩
  · rw [←actual_pointer q 1 _ _ _ ql (by omega), qa]
    simpa [Native.pointer] using Native.pointer_exact (base q 1) 1 r (columns x.metadata) c rb cb (by simpa using qbound)
  · rw [←actual_pointer m 2 _ _ _ ml (by omega), Native.widen32 r rb, ma,
      BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt (by omega : base m 2 + r*2 < 2^64)]
    omega

#axiomsClean output_pointers
/-- Complete source execution and its proved byte-storage realization. The
last two clauses connect actual output Tensor addresses directly to the two
source stores. No successful run or desired output is an input premise. -/
theorem actual_kernel (a : Arithmetic) (rest : FP.Structural.Algebra Word)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover)) (q m : OutputTensor)
    (ql : OutputLaws q 1 (.other "int8") [rows x.metadata, columns x.metadata] x.metadata.device)
    (ml : OutputLaws m 2 (.other "float16") [rows x.metadata] x.metadata.device)
    (sep : Separation (Direct.launch x.metadata h tree cover) (layout x laws h q m))
    (heap : Heap) (scratch : FP.Structural.State Word) :
    ∃ final,
      KernelGrid.run (algebra a (Direct.launch x.metadata h tree cover) rest) (llrint a) (equal a)
        (Direct.launch x.metadata h tree cover) schedule.order (initial x laws h heap scratch) = some final ∧
      KernelNativeResult.Result a (Direct.launch x.metadata h tree cover) (values x laws heap)
        (initial x laws h heap scratch) final ∧
      ByteResult a x laws h tree cover q m heap (byteExecution a x laws h tree cover schedule q m heap) ∧
      (∀ r c, r < rows x.metadata → c < columns x.metadata →
        byteExecution a x laws h tree cover schedule q m heap (address q 1 (r*columns x.metadata+c)) =
          narrow8 ((final.mem "q" (r*columns x.metadata+c)).read .int)) ∧
      (∀ r, r < rows x.metadata → ∀ b : Fin 2,
        byteExecution a x laws h tree cover schedule q m heap (address m 2 r + b.val) =
          halfByte (KernelEncoding.unpackHalf ((final.mem "m" r).read .fp16)) b) := by
  obtain ⟨final, run, result⟩ := KernelNativeResult.kernel_correct a rest
    (Direct.launch x.metadata h tree cover) schedule (values x laws heap) (initial x laws h heap scratch)
    (initial_reads x laws h tree cover heap scratch)
  have bytes := byte_correct a x laws h tree cover schedule q m ql ml sep heap
  refine ⟨final, run, result, bytes, ?_, ?_⟩
  · intro r c hr hc
    exact (bytes.1 r c hr hc).trans (result.1 r c hr hc).symm
  · intro r hr b
    exact (bytes.2.1 r hr b).trans (congrArg (fun v => halfByte v b) (result.2.1 r hr)).symm

#axiomsClean byte_correct
#axiomsClean actual_kernel
#axiomsClean direct_address
#axiomsClean span
#axiomsClean actual_pointer
#axiomsClean initial_reads
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelTensor

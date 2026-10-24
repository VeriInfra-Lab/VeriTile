import bench.tritonbench_g.relu_triton_kernel.Proofs.FP64Host
import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteIndices

/-! Source-specific applicability of trusted Torch metadata/allocation and
typed Triton memory primitives. A region is the tensor's data-pointer view,
not its storage origin. Strides are returned metadata; the source deliberately
does not use them. Thus the result describes physical consecutive elements,
including safe noncontiguous views, and never silently claims logical ReLU.
No allocator/compiler implementation or native error spelling is assumed. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteBoundary
open VeriTile.Triton HostWrapper GatedGrid FP64Host
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def numel (x : Tensor) : Nat := x.shape.prod

def bytes (dtype : String) : Nat :=
  if dtype = "float64" ∨ dtype = "int64" ∨ dtype = "uint64" then 8
  else if dtype = "float32" ∨ dtype = "int32" ∨ dtype = "uint32" then 4
  else if dtype = "float16" ∨ dtype = "bfloat16" ∨ dtype = "int16" ∨ dtype = "uint16" then 2
  else 1

structure StorageView where
  storageBase : Nat
  capacityBytes : Nat
  readable : Nat → Prop
  writable : Nat → Prop

/-- Trusted metadata interpretation of actual Torch storage and the default
preserve_format policy of empty_like. This is a metadata primitive only;
it has no access to kernel numerics, final state, or success/failure outcomes. -/
structure TensorSemantics where
  view : Tensor → StorageView
  emptyLikeStrides : Tensor → List Int

instance : CoeFun TensorSemantics (fun _ => Tensor → StorageView) := ⟨TensorSemantics.view⟩

def dataBase (v : StorageView) (x : Tensor) : Nat :=
  v.storageBase + x.storageOffset * bytes x.dtype

/-- Permissions refer to actual absolute byte addresses. The extent certifies
the flat N-element source load, which may exceed the span of an expanded view.
For N=0 the permission premises are vacuous; the pointer must still fit. -/
structure ValidView (v : StorageView) (x : Tensor) : Prop where
  positiveWidth : 0 < bytes x.dtype
  baseBound : dataBase v x < 2^64
  storageBound : v.storageBase + v.capacityBytes ≤ 2^64
  extent : x.storageOffset * bytes x.dtype + numel x * bytes x.dtype ≤ v.capacityBytes

def Readable (v : StorageView) (x : Tensor) : Prop :=
  ∀ i b, i < numel x → b < bytes x.dtype → v.readable (dataBase v x + i * bytes x.dtype + b)

def Writable (v : StorageView) (x : Tensor) : Prop :=
  ∀ i b, i < min (numel x) 1024 → b < bytes x.dtype → v.writable (dataBase v x + i * bytes x.dtype + b)

theorem view_byte_domain (v : StorageView) (x : Tensor) (h : ValidView v x) :
    FiniteAddress.ByteDomain (dataBase v x) (bytes x.dtype) (numel x) := by
  refine ⟨h.baseBound, ?_⟩
  have he := h.extent
  have hs := h.storageBound
  unfold dataBase
  omega

theorem actual_pointer (v : StorageView) (x : Tensor) (h : ValidView v x)
    (domain : FiniteIndices.Domain (numel x)) (pid lane byte : Nat)
    (launched : pid < gridCount (numel x) 1024) (inBlock : lane < 1024)
    (active : FiniteAddress.offset pid lane < numel x) (inElement : byte < bytes x.dtype) :
    (FiniteAddress.bytePointer64 (dataBase v x) (bytes x.dtype) pid lane byte).toNat =
      v.storageBase + (x.storageOffset + FiniteAddress.offset pid lane) * bytes x.dtype + byte := by
  rw [FiniteIndices.byte_pointer_exact _ _ _ _ _ _ domain (view_byte_domain v x h)
    launched inBlock active inElement]
  simp [dataBase, Nat.add_mul, Nat.add_assoc]

/-- Actual typed storage access is a trusted primitive view. Integer/bool
payloads are finite source words; floating payloads have their exact dtype tag.
The numerical backend relates typed float payloads to native values separately. -/
def Payload {α : Type} (p : TypedValue.Profile) (c : FP.Structural.Cell α) : Prop :=
  match p with
  | .i8 => ∃ w : BitVec 8, c = IntegerWords.decodeCell (.signed .w8) w
  | .i16 => ∃ w : BitVec 16, c = IntegerWords.decodeCell (.signed .w16) w
  | .i32 => ∃ w : BitVec 32, c = IntegerWords.decodeCell (.signed .w32) w
  | .i64 => ∃ w : BitVec 64, c = IntegerWords.decodeCell (.signed .w64) w
  | .u8 => ∃ w : BitVec 8, c = IntegerWords.decodeCell (.unsigned .w8) w
  | .u16 => ∃ w : BitVec 16, c = IntegerWords.decodeCell (.unsigned .w16) w
  | .u32 => ∃ w : BitVec 32, c = IntegerWords.decodeCell (.unsigned .w32) w
  | .u64 => ∃ w : BitVec 64, c = IntegerWords.decodeCell (.unsigned .w64) w
  | .boolean => ∃ w : BitVec 8, c = IntegerWords.decodeCell .boolean w
  | .fp16 => ∃ value : α, c = .mk .fp16 value
  | .bf16 => ∃ value : α, c = .mk .bf16 value
  | .fp32 => ∃ value : α, c = .mk .fp32 value

structure TypedInput {α : Type} (p : TypedValue.Profile) (x : Tensor)
    (v : StorageView) (storage : Nat → FP.Structural.Cell α) (s : Runtime α) : Prop where
  dtype : x.dtype = profileDType p
  valid : ValidView v x
  readable : Readable v x
  view : ∀ i, i < numel x → s.1.mem x.region i = storage (dataBase v x + i * bytes x.dtype)
  payload : ∀ i, i < numel x → Payload p (storage (dataBase v x + i * bytes x.dtype))

structure WideInput {α : Type} (x : Tensor) (v : StorageView)
    (storage : Nat → BitVec 64) (s : Runtime α) : Prop where
  dtype : x.dtype = "float64"
  valid : ValidView v x
  readable : Readable v x
  view : ∀ i, i < numel x → s.2.f64 x.region i = storage (dataBase v x + i * bytes x.dtype)

structure ReturnedMetadata (views : TensorSemantics) (x output : Tensor) : Prop where
  dtype : output.dtype = "float32"
  shape : output.shape = x.shape
  device : output.device = x.device
  offset : output.storageOffset = 0
  stridesRank : output.strides.length = output.shape.length
  strides : output.strides = views.emptyLikeStrides x

theorem returned_numel (views : TensorSemantics) (x output : Tensor) (h : ReturnedMetadata views x output) :
    numel output = numel x := by simp [numel, h.shape]

structure Allocation {α : Type} (views : TensorSemantics) (x output : Tensor) (before after : Runtime α) : Prop where
  metadata : ReturnedMetadata views x output
  fresh : x.region ≠ output.region
  typedFrame : ∀ r i, r ≠ output.region → after.1.mem r i = before.1.mem r i
  wideFrame : ∀ r i, r ≠ output.region → after.2.f64 r i = before.2.f64 r i
  outputFrame : ∀ r i, r ≠ output.region → after.2.f32 r i = before.2.f32 r i
  validOutput : ValidView (views output) output
  writableOutput : Writable (views output) output
  separated : dataBase (views x) x + numel x * bytes x.dtype ≤ dataBase (views output) output ∨
    dataBase (views output) output + numel output * bytes output.dtype ≤ dataBase (views x) x
  typedOutput : ∀ i, i < numel output → ∃ v : α, after.1.mem output.region i = .mk .fp32 v

/-- Successful read-only metadata queries preserve state. A successful
empty_like gives the actual metadata, fresh region, valid fp32 view and frame.
Errors remain fallible and retain the primitive's supplied state. No launch or
final output equation occurs in this trusted producer contract. -/
structure PrimitivesContract {α : Type} (views : TensorSemantics) (P : Primitives (Runtime α)) : Prop where
  device : ∀ x s d t, P.device x s = .ok d t → d = x.device ∧ t = s
  allocation : ∀ x d s output t, P.emptyLike x "float32" d s = .ok output t →
    d = x.device → Allocation views x output s t
  numel : ∀ x s n t, P.numel x s = .ok n t → n = numel x ∧ t = s

theorem metadata_chain {α : Type} (views : TensorSemantics) (P : Primitives (Runtime α)) (C : PrimitivesContract views P)
    (x output : Tensor) (s d a m : Runtime α) (device : String) (n : Nat)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    d = s ∧ m = a ∧ n = numel x ∧ Allocation views x output s a := by
  obtain ⟨dev, ds⟩ := C.device x s device d hd
  obtain ⟨nm, ma⟩ := C.numel output a n m hm
  have alloc := C.allocation x device d output a ha dev
  subst d
  exact ⟨rfl, ma, nm.trans (returned_numel views x output alloc.metadata), alloc⟩

theorem typed_input_preserved {α : Type} (views : TensorSemantics) (p : TypedValue.Profile) (x output : Tensor)
    (v : StorageView) (storage : Nat → FP.Structural.Cell α) (s a : Runtime α)
    (input : TypedInput p x v storage s) (alloc : Allocation views x output s a) :
    TypedInput p x v storage a := by
  refine ⟨input.dtype, input.valid, input.readable, ?_, input.payload⟩
  intro i hi
  rw [alloc.typedFrame _ i alloc.fresh]
  exact input.view i hi

theorem wide_input_preserved {α : Type} (views : TensorSemantics) (x output : Tensor)
    (v : StorageView) (storage : Nat → BitVec 64) (s a : Runtime α)
    (input : WideInput x v storage s) (alloc : Allocation views x output s a) :
    WideInput x v storage a := by
  refine ⟨input.dtype, input.valid, input.readable, ?_⟩
  intro i hi
  rw [alloc.wideFrame _ i alloc.fresh]
  exact input.view i hi

/-- The finite reviewed API: actual CUDA strided tensors, all seventeen
canonical scalar dtypes, and the source's safe source index domain including constexpr N=2^31.
Storage safety is separate, so an expanded/short view is explicitly unsafe,
not a frontend rejection or a silently omitted dtype. -/
def SourceDomain (x : Tensor) : Prop :=
  (x.device = "cuda" ∨ ∃ index : Nat, x.device = "cuda:" ++ toString index) ∧
  x.dtype ∈ declaredDTypes ∧
  x.strides.length = x.shape.length ∧ FiniteIndices.Domain (numel x)

theorem finite_classification (x : Tensor) (h : SourceDomain x) :
    (∃ p, admission x.dtype = .typed p) ∨ admission x.dtype = .fp64 ∨
    admission x.dtype = .comparisonRejected ∨ admission x.dtype = .signatureRejected := by
  have hd := h.2.1
  simp only [declaredDTypes, List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h
  all_goals simp [h, admission, selectProfile]

def AccessSummary (views : TensorSemantics) (x output : Tensor) : Prop :=
  (∀ pid lane, pid < gridCount (numel x) 1024 → lane < 1024 →
    FiniteIndices.sourceMask (numel x) pid lane = decide (FiniteAddress.offset pid lane < numel x)) ∧
  (∀ pid lane byte, pid < gridCount (numel x) 1024 → lane < 1024 →
    FiniteAddress.offset pid lane < numel x → byte < bytes x.dtype →
    (FiniteAddress.bytePointer64 (dataBase (views x) x) (bytes x.dtype) pid lane byte).toNat =
      (views x).storageBase + (x.storageOffset + FiniteAddress.offset pid lane) * bytes x.dtype + byte ∧
    (views x).readable (dataBase (views x) x + FiniteAddress.offset pid lane * bytes x.dtype + byte)) ∧
  (∀ i b, i < min (numel x) 1024 → b < bytes output.dtype →
    (FiniteAddress.bytePointer64 (dataBase (views output) output) (bytes output.dtype) 0 i b).toNat =
      (views output).storageBase + (output.storageOffset + i) * bytes output.dtype + b ∧
    (views output).writable (dataBase (views output) output + i * bytes output.dtype + b)) ∧
  (∀ i j u v, i < numel x → j < numel output → u < bytes x.dtype → v < bytes output.dtype →
    dataBase (views x) x + i * bytes x.dtype + u ≠
      dataBase (views output) output + j * bytes output.dtype + v)

theorem access_summary {α : Type} (views : TensorSemantics) (x output : Tensor)
    (s a : Runtime α) (domain : SourceDomain x) (valid : ValidView (views x) x)
    (readable : Readable (views x) x) (alloc : Allocation views x output s a) :
    AccessSummary views x output := by
  have nx := returned_numel views x output alloc.metadata
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro pid lane launched within
    exact FiniteIndices.mask_exact (numel x) pid lane domain.2.2.2 launched within
  · intro pid lane byte launched within active inElement
    exact ⟨actual_pointer (views x) x valid domain.2.2.2 pid lane byte launched within active inElement,
      readable _ byte active inElement⟩
  · intro i b active inElement
    have hi : i < numel output := by rw [nx]; exact (Nat.lt_min.mp active).1
    have hb : i < 1024 := (Nat.lt_min.mp active).2
    have launched := positive_grid_count (numel output) 1024 (by omega) (by decide)
    have ix : FiniteIndices.Domain (numel output) := nx ▸ domain.2.2.2
    have address := actual_pointer (views output) output alloc.validOutput ix 0 i b launched hb
      (by simpa [FiniteAddress.offset] using hi) inElement
    refine ⟨?_, alloc.writableOutput i b (by simpa [nx] using active) inElement⟩
    simpa [FiniteAddress.offset] using address
  · intro i j u v hi hj hu hv
    exact FiniteAddress.separate_byte_ranges _ _ _ _ _ _ _ _ _ _ alloc.separated hi hj hu hv

#specNonCircular SourceDomain avoiding [HostWrapper.relu, FP64Host.logicalLaunch]
#specNonCircular PrimitivesContract avoiding [HostWrapper.relu, FP64Host.logicalLaunch]
#axiomsClean view_byte_domain
#axiomsClean actual_pointer
#axiomsClean returned_numel
#axiomsClean metadata_chain
#axiomsClean typed_input_preserved
#axiomsClean wide_input_preserved
#axiomsClean finite_classification
#axiomsClean access_summary
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteBoundary

import bench.repairs.tritonbench_g.ksoftmax_triton.Kernels
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Observed
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Binding
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Pointers
namespace VeriTile.Bench.Repairs.Ksoftmax.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent FP32 instruction-tree reference; all exceptional words allowed.
Source-selected reduction trees, fusion, casts, sentinels and exp/log remain. -/
abbrev reference := Graph.reference

def OutputWords (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata)
    (final : State Graph.Expr) (left right : Nat → Graph.Word) : Prop :=
  ∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
    Graph.eval ops ((final.mem "dst" (Native.flat (rows x) (columns x) b r i.val)).read .real)=
      reference rest profile ops op (rows x) (columns x) b r 0 1 left right i

specification forward_words (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (y x : TensorMetadata) (mask : ArgumentMetadata)
    (log : Bool) (mode : Option String) (causal : Bool) (admitted : ForwardContract y x mask mode)
    (_compiler : Graph.Binding profile (columns x) (depth.depth (columns x)))
    (initial : State Graph.Expr) (left right : Nat → Graph.Word)
    (loads : InputWords depth (.forward log causal (Host.decodeMode mode)) x initial left right) :
    Spec.Real (∃ final,
      Host.softmax (Graph.algebra rest profile) depth y x mask log mode causal initial=.ok ((),final) ∧
      OutputWords rest profile ops depth (.forward log causal (Host.decodeMode mode)) x final left right ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=initial.mem region j)) := by
  refine ⟨Host.body (Graph.algebra rest profile) depth (.forward log causal (Host.decodeMode mode)) x initial,
    Host.forward_return _ _ _ _ _ _ _ _ admitted initial,?_,?_⟩
  · exact body_words rest profile ops depth _ x admitted.2.2.1 initial left right loads
  · exact Host.body_frame _ _ _ _ admitted.2.2.1 initial

specification backward_words (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (d g o : TensorMetadata) (log causal : Bool) (admitted : BackwardContract d g o)
    (_compiler : Graph.Binding profile (columns o) (depth.depth (columns o)))
    (initial : State Graph.Expr) (left right : Nat → Graph.Word)
    (loads : InputWords depth (.backward log causal) o initial left right) :
    Spec.Real (∃ final,
      Host.softmaxBackward (Graph.algebra rest profile) depth d g o log causal initial=.ok ((),final) ∧
      OutputWords rest profile ops depth (.backward log causal) o final left right ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ o.numel≤j → final.mem region j=initial.mem region j)) := by
  refine ⟨Host.body (Graph.algebra rest profile) depth (.backward log causal) o initial,
    Host.backward_return _ _ _ _ _ _ _ admitted initial,?_,?_⟩
  · exact body_words rest profile ops depth _ o admitted.2.2.2.1 initial left right loads
  · exact Host.body_frame _ _ _ _ admitted.2.2.2.1 initial

/-- Concrete caller-byte readback for every branch of either complete source
grid. The caller supplies the ordinary initial backing-byte binding only. -/
specification body_bytes (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (op : Operation) (dst x : TensorMetadata)
    (rank : x.shape.length=3) (size : dst.numel=x.numel)
    (initial : State Graph.Expr) (left right old : Nat → Graph.Word) (before : Binding.Bytes)
    (loads : InputWords depth op x initial left right)
    (oldLoads : ∀ j, j<dst.numel → (initial.mem "dst" j).read .real=.leaf (old j))
    (oldBytes : ∀ j, j<dst.numel → ∀ b, b<4 → before (dst.dataPtr.toNat+4*j+b)=Binding.byteOf (old j) b) :
    Spec.Real (Binding.callerBytes ops dst before initial=before ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (byte : Nat),
        i.val<columns x → byte<4 →
        Binding.callerBytes ops dst before (Host.body (Graph.algebra rest profile) depth op x initial)
          (dst.dataPtr.toNat+4*Native.flat (rows x) (columns x) b r i.val+byte)=
        Binding.byteOf (reference rest profile ops op (rows x) (columns x) b r 0 1 left right i) byte) ∧
      (∀ address, address<dst.dataPtr.toNat ∨ dst.dataPtr.toNat+4*dst.numel≤address →
        Binding.callerBytes ops dst before (Host.body (Graph.algebra rest profile) depth op x initial) address=before address)) := by
  refine ⟨Binding.initial_bytes ops dst before initial old oldLoads oldBytes,?_,?_⟩
  · intro b r i byte hi hb
    rw [Binding.coordinate_bytes ops dst before _ _ byte (by rw [size,count_three x rank];exact Native.active_bound b.isLt r.isLt hi) hb,
      body_words rest profile ops depth op x rank initial left right loads b r i hi]
  · exact Binding.exterior_bytes ops dst before _

/-- Caller-visible four-byte output and full exterior-byte frame. -/
def ByteOutput (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (op : Operation) (dst x : TensorMetadata) (before : Binding.Bytes)
    (final : State Graph.Expr) (left right : Nat → Graph.Word) : Prop :=
  (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (byte : Nat),
    i.val<columns x → byte<4 →
    Binding.callerBytes ops dst before final (dst.dataPtr.toNat+4*Native.flat (rows x) (columns x) b r i.val+byte)=
      Binding.byteOf (reference rest profile ops op (rows x) (columns x) b r 0 1 left right i) byte) ∧
  (∀ address, address<dst.dataPtr.toNat ∨ dst.dataPtr.toNat+4*dst.numel≤address →
    Binding.callerBytes ops dst before final address=before address)

specification forward_bytes (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (y x : TensorMetadata) (mask : ArgumentMetadata)
    (log : Bool) (mode : Option String) (causal : Bool) (admitted : ForwardContract y x mask mode)
    (_compiler : Graph.Binding profile (columns x) (depth.depth (columns x)))
    (initial : State Graph.Expr) (left right old : Nat → Graph.Word) (before : Binding.Bytes)
    (loads : InputWords depth (.forward log causal (Host.decodeMode mode)) x initial left right)
    (oldLoads : ∀ j, j<y.numel → (initial.mem "dst" j).read .real=.leaf (old j))
    (oldBytes : ∀ j, j<y.numel → ∀ b, b<4 → before (y.dataPtr.toNat+4*j+b)=Binding.byteOf (old j) b) :
    Spec.Real (∃ final,
      Host.softmax (Graph.algebra rest profile) depth y x mask log mode causal initial=.ok ((),final) ∧
      Binding.callerBytes ops y before initial=before ∧
      ByteOutput rest profile ops depth (.forward log causal (Host.decodeMode mode)) y x before final left right) := by
  have size : y.numel=x.numel := by
    change y.shape.prod=x.shape.prod
    rw [admitted.2.2.2.1]
  refine ⟨Host.body (Graph.algebra rest profile) depth (.forward log causal (Host.decodeMode mode)) x initial,
    Host.forward_return _ _ _ _ _ _ _ _ admitted initial,?_⟩
  exact body_bytes rest profile ops depth _ y x admitted.2.2.1 size initial left right old before loads oldLoads oldBytes

specification backward_bytes (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (depth : Host.DepthLibrary) (d g o : TensorMetadata) (log causal : Bool) (admitted : BackwardContract d g o)
    (_compiler : Graph.Binding profile (columns o) (depth.depth (columns o)))
    (initial : State Graph.Expr) (left right old : Nat → Graph.Word) (before : Binding.Bytes)
    (loads : InputWords depth (.backward log causal) o initial left right)
    (oldLoads : ∀ j, j<d.numel → (initial.mem "dst" j).read .real=.leaf (old j))
    (oldBytes : ∀ j, j<d.numel → ∀ b, b<4 → before (d.dataPtr.toNat+4*j+b)=Binding.byteOf (old j) b) :
    Spec.Real (∃ final,
      Host.softmaxBackward (Graph.algebra rest profile) depth d g o log causal initial=.ok ((),final) ∧
      Binding.callerBytes ops d before initial=before ∧
      ByteOutput rest profile ops depth (.backward log causal) d o before final left right) := by
  have size : d.numel=o.numel := by
    change d.shape.prod=o.shape.prod
    rw [admitted.2.2.2.2.1]
  refine ⟨Host.body (Graph.algebra rest profile) depth (.backward log causal) o initial,
    Host.backward_return _ _ _ _ _ _ _ admitted initial,?_⟩
  exact body_bytes rest profile ops depth _ d o admitted.2.2.2.1 size initial left right old before loads oldLoads oldBytes

/-- Only instruction-local numerical ports remain conditional. Domain checks
actual intermediate operands at every node of the selected expression. -/
theorem body_exact_if_local (rest : Algebra Graph.Expr) (profile : Graph.Profile) (ops : Graph.Operations)
    (ref : Words.PrimitiveReference) (ports : Words.LocalExact ops ref)
    (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata) (rank : x.shape.length=3)
    (initial : State Graph.Expr) (left right : Nat → Graph.Word) (loads : InputWords depth op x initial left right)
    (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (hi : i.val<columns x)
    (domain : Words.NumericDomain ops ((gridValue (Graph.algebra rest profile) op (rows x) (columns x) b r
      (fun j => .input 0 j (left j)) (fun j => .input 1 j (right j)) i).read .real)) :
    Graph.eval ops (((Host.body (Graph.algebra rest profile) depth op x initial).mem "dst"
      (Native.flat (rows x) (columns x) b r i.val)).read .real)=
    reference rest profile (Words.referenceOps ref) op (rows x) (columns x) b r 0 1 left right i :=
  (body_words rest profile ops depth op x rank initial left right loads b r i hi).trans
    (Words.reference_exact rest profile ops ref ports op _ _ _ _ _ _ left right i domain)

#specNonCircular reference avoiding [forward,backward,Host.body,Host.softmax,Host.softmaxBackward]
#axiomsClean forward_words
#axiomsClean backward_words
#axiomsClean forward_bytes
#axiomsClean backward_bytes
#axiomsClean body_bytes
#axiomsClean body_exact_if_local
end
end VeriTile.Bench.Repairs.Ksoftmax.FPEquiv

import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostFPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelInputs
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.Ksoftmax.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Complete observable source result, with source success as a conclusion. -/
def KernelResult (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (op : Operation) (dst left right x : TensorMetadata)
    (code : KernelCompiler.Selected platform op (rows x) (columns x) (depth.depth (columns x)) dst left right)
    (dstView : KernelTensor.View dst) (leftView : KernelTensor.View left) (rightView : KernelTensor.View right)
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes) : Prop :=
  Spec.Real (∃ final,
      KernelExecution.exec (code.model rest) depth op x (KernelTensor.nativeInitial dst left right memory s)=some final ∧
      OutputWords rest code.lowering.profile code.ops depth op x final
        (KernelTensor.values memory left leftView) (KernelTensor.values memory right rightView) ∧
      Binding.callerBytes code.ops dst before (KernelTensor.nativeInitial dst left right memory s)=before ∧
      ByteOutput rest code.lowering.profile code.ops depth op dst x before final
        (KernelTensor.values memory left leftView) (KernelTensor.values memory right rightView) ∧
      (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))) (byte : Nat),
        i.val<columns x → byte<4 →
        Binding.callerBytes code.ops dst before final
          (KernelTensor.address dst dstView (Native.flat (rows x) (columns x) b r i.val)+byte)=
        Binding.byteOf (reference rest code.lowering.profile code.ops op (rows x) (columns x) b r 0 1
          (KernelTensor.values memory left leftView) (KernelTensor.values memory right rightView) i) byte) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ x.numel≤j → final.mem region j=
        (KernelTensor.nativeInitial dst left right memory s).mem region j))

/-- All selected FP32 words, including NaNs, infinities and signed zero. Raw
source loads are constructed from actual Tensor pointers in one physical word
map. Source execution and complete output/frame are proved, not assumed. -/
theorem kernel_native (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (op : Operation) (dst left right x : TensorMetadata)
    (code : KernelCompiler.Selected platform op (rows x) (columns x) (depth.depth (columns x)) dst left right)
    (rank : x.shape.length=3) (size : dst.numel=x.numel)
    (dstGuard : TensorContract dst) (leftGuard : TensorContract left) (rightGuard : TensorContract right)
    (dstView : KernelTensor.View dst) (leftView : KernelTensor.View left) (rightView : KernelTensor.View right)
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes)
    (backing : KernelTensor.Backed memory before [dst,left,right])
    (loads : InputWords depth op x (KernelTensor.nativeInitial dst left right memory s)
      (KernelTensor.words memory left) (KernelTensor.words memory right)) :
    KernelResult rest platform depth op dst left right x code dstView leftView rightView memory s before := by
  unfold KernelResult
  let initial := KernelTensor.nativeInitial dst left right memory s
  let final := Host.body (code.model rest) depth op x initial
  have output := body_words rest code.lowering.profile code.ops depth op x rank initial
    (KernelTensor.words memory left) (KernelTensor.words memory right) loads
  have physical := body_bytes rest code.lowering.profile code.ops depth op dst x rank size initial
    (KernelTensor.words memory left) (KernelTensor.words memory right) (KernelTensor.words memory dst)
    before loads (fun j _ => KernelTensor.native_dst dst left right memory s j)
      (backing dst (by simp))
  rw [KernelTensor.values_words memory left leftView leftGuard,
    KernelTensor.values_words memory right rightView rightGuard]
  refine ⟨final,KernelExecution.success _ _ _ _ _,output,physical.1,⟨physical.2.1,physical.2.2⟩,?_,?_⟩
  · intro b r i byte hi hb
    have bound : Native.flat (rows x) (columns x) b r i.val<dst.numel := by
      rw [size,count_three x rank]
      exact Native.active_bound b.isLt r.isLt hi
    rw [KernelTensor.output_bytes code.ops dst dstView dstGuard before final _ byte bound hb,
      output b r i hi]
  · exact Host.body_frame _ _ _ _ rank initial

/-- Instantiation for every forward flag and admitted mask branch. No input
load-binding hypothesis remains: it follows from the constructed Tensor state. -/
specification forward_kernel_native (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (y x : TensorMetadata) (mask : ArgumentMetadata)
    (log : Bool) (mode : Option String) (causal : Bool) (admitted : ForwardContract y x mask mode)
    (code : KernelCompiler.Selected platform (.forward log causal (Host.decodeMode mode))
      (rows x) (columns x) (depth.depth (columns x)) y x (KernelInputs.maskTensor x mask mode))
    (yView : KernelTensor.View y) (xView : KernelTensor.View x)
    (maskView : KernelTensor.View (KernelInputs.maskTensor x mask mode))
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes)
    (backing : KernelTensor.Backed memory before [y,x,KernelInputs.maskTensor x mask mode]) :
    KernelResult rest platform depth (.forward log causal (Host.decodeMode mode)) y x
      (KernelInputs.maskTensor x mask mode) x code yView xView maskView memory s before ∧
    (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))), i.val<columns x →
      (KernelAccess.matrixPointer y (rows x) (columns x) b r i).toNat=
        KernelTensor.address y yView (Native.flat (rows x) (columns x) b r i) ∧
      (KernelAccess.matrixPointer x (rows x) (columns x) b r i).toNat=
        KernelTensor.address x xView (Native.flat (rows x) (columns x) b r i)) ∧
    (∀ (b : Fin (batches x)) (r : Fin (rows x)) (i : Fin (depth.depth (columns x))),
      i.val<columns x → (Host.decodeMode mode).present=Bool.true →
      (KernelAccess.maskPointer (KernelInputs.maskTensor x mask mode) (Host.decodeMode mode) b r (columns x) i).toNat=
        KernelTensor.address (KernelInputs.maskTensor x mask mode) maskView
          (Forward.maskAddress (Host.decodeMode mode) b r (columns x) i)) :=
  And.intro (kernel_native rest platform depth (.forward log causal (Host.decodeMode mode)) y x
    (KernelInputs.maskTensor x mask mode) x code admitted.2.2.1
    (by change y.shape.prod=x.shape.prod; rw [admitted.2.2.2.1])
    admitted.1 admitted.2.1 (KernelInputs.mask_admitted y x mask mode admitted)
    yView xView maskView memory s before backing (KernelInputs.forward_words depth y x mask mode log causal memory s))
    (And.intro (KernelAccess.forward_matrices depth y x mask mode admitted yView xView)
      (KernelAccess.forward_mask depth y x mask mode admitted maskView))

/-- Instantiation for both backward formulas and both causal branches. -/
specification backward_kernel_native (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata) (log causal : Bool)
    (admitted : BackwardContract dst grad saved)
    (code : KernelCompiler.Selected platform (.backward log causal)
      (rows saved) (columns saved) (depth.depth (columns saved)) dst grad saved)
    (dstView : KernelTensor.View dst) (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State Graph.Expr) (before : Binding.Bytes)
    (backing : KernelTensor.Backed memory before [dst,grad,saved]) :
    KernelResult rest platform depth (.backward log causal) dst grad saved saved code
      dstView gradView savedView memory s before ∧
    (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
      (KernelAccess.matrixPointer dst (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address dst dstView (Native.flat (rows saved) (columns saved) b r i) ∧
      (KernelAccess.matrixPointer grad (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address grad gradView (Native.flat (rows saved) (columns saved) b r i) ∧
      (KernelAccess.matrixPointer saved (rows saved) (columns saved) b r i).toNat=
        KernelTensor.address saved savedView (Native.flat (rows saved) (columns saved) b r i)) :=
  And.intro (kernel_native rest platform depth (.backward log causal) dst grad saved saved code admitted.2.2.2.1
    (by change dst.shape.prod=saved.shape.prod; rw [admitted.2.2.2.2.1])
    admitted.1 admitted.2.1 admitted.2.2.1 dstView gradView savedView memory s before backing
    (KernelInputs.backward_words depth dst grad saved log causal memory s))
    (KernelAccess.backward_matrices depth dst grad saved admitted dstView gradView savedView)

theorem read_tensor_bytes_preserved (ops : Graph.Operations) (dst input : TensorMetadata)
    (separate : PairContract dst input) (before : Binding.Bytes) (s : State Graph.Expr)
    (j byte : Nat) (hj : j < input.numel) (hb : byte<4) :
    Binding.callerBytes ops dst before s (input.dataPtr.toNat+4*j+byte)=
      before (input.dataPtr.toNat+4*j+byte) := by
  apply Binding.exterior_bytes
  rcases separate.2 with h|h
  · exact Or.inr (by omega)
  · exact Or.inl (by omega)

/-- Empty dispatch needs no compiled object, Tensor pointer query or numeric
domain. It preserves the supplied source state exactly for either entry. -/
specification empty_kernel {α : Type} [Inhabited α] (model : Algebra α)
    (depth : Host.DepthLibrary) (op : Operation) (x : TensorMetadata)
    (s : State α) (empty : x.numel=0) :
    KernelExecution.exec model depth op x s=some s :=
  KernelExecution.empty model depth op x s empty

#axiomsClean kernel_native
#axiomsClean forward_kernel_native
#axiomsClean backward_kernel_native
#axiomsClean read_tensor_bytes_preserved
#axiomsClean empty_kernel
end
end VeriTile.Bench.Repairs.Ksoftmax.FPEquiv

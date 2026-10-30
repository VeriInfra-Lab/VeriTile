import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_recurrent_delta.Correct
import bench.repairs.tritonbench_g.fused_recurrent_delta.FPEquiv
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealBound

/-! Complete forward/backward/native Tensor applications of the raw kernel results. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.TensorBridge
open VeriTile.Triton FP.Structural KernelProgram KernelTensor KernelBound KernelNativeCarrier KernelRealBound
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

theorem kernel_physical_logical_input {α : Type} [Inhabited α]
    {kind : Kind} {s : Shape} {f : Flags} (b : Buffers kind s f)
    (read : RegionName → Nat → Nat → α) (u : State α) :
    physical b read u = logical b read u := KernelTensor.physical_eq_logical b read u

theorem kernel_forward_real {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) (arg : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.ScaleAdmitted platform binding arg) (b : Buffers .forward s f)
    (p : Program .forward s) (u : State ℝ)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    (∃ q : ℚ,(KernelScale.launchWord platform binding arg).toRat?=some q ∧
      KernelScale.realScale platform binding arg=(q:ℝ)) ∧
    (∃ v,KernelForward.Facts KernelReal.algebra s f (KernelScale.realScale platform binding arg) p.head p.value
      (physical b (fun _ _ => api.read initial) u) v ∧
      Shared.TensorImage api (placement b p) (fun _ _ cell => cell.read .real) v initial ∧
      InputFrame api b p (fun _ _ cell => cell.read .real) v initial ∧
      (∀ r off value,KernelReal.ForwardExpected s f (KernelScale.realScale platform binding arg) p.head p.value
        (logical b (fun _ _ => api.read initial) u) r off value →
        api.read (Shared.commit api (placement b p) (fun _ _ cell => cell.read .real) v initial)
          ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=value)) :=
  And.intro (KernelScale.finite_projection platform binding arg scale)
    (KernelRealBound.forward_tensor api initial s f (KernelScale.realScale platform binding arg) b p u hv hk hh)

theorem kernel_backward_real {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) (arg : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.ScaleAdmitted platform binding arg) (b : Buffers .backward s f)
    (p : Program .backward s) (u : State ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : KernelReal.Saved s (KernelBackwardReference.input KernelReal.algebra s f p.head p.value
      (logical b (fun _ _ => api.read initial) u)) a g terminal)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    (∃ q : ℚ,(KernelScale.launchWord platform binding arg).toRat?=some q ∧
      KernelScale.realScale platform binding arg=(q:ℝ)) ∧
    (ColumnVJP s (KernelScale.realScale platform binding arg) a g terminal ∧ ∃ v,
      KernelBackwardFacts.Facts KernelReal.algebra s f (KernelScale.realScale platform binding arg) p.head p.value
        (physical b (fun _ _ => api.read initial) u) v ∧
      Shared.TensorImage api (placement b p) (fun _ _ cell => cell.read .real) v initial ∧
      InputFrame api b p (fun _ _ cell => cell.read .real) v initial ∧
      (∀ r off value,KernelReal.BackwardExpected s f (KernelScale.realScale platform binding arg) p.head p.value a g terminal r off value →
        api.read (Shared.commit api (placement b p) (fun _ _ cell => cell.read .real) v initial)
          ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=value)) :=
  And.intro (KernelScale.finite_projection platform binding arg scale)
    (KernelRealBound.backward_tensor api initial s f (KernelScale.realScale platform binding arg)
      b p u a g terminal saved hv hk hh)

theorem kernel_native_tensor {Store Byte : Type} (api : Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (kind : Kind) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform kind s f) (arg : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.ScaleAdmitted platform binding arg)
    (b : Buffers kind s f) (abi : PointerABI platform binding b) (p : Program kind s) (u : State Node)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    PointerABI platform binding b ∧ KernelCompiler.ScaleAdmitted platform binding arg ∧ ∃ v,
      KernelCompiler.Facts platform binding (KernelCompiler.hostScale platform s arg) p
        (physical b (nativeRead api initial) u) v ∧
      Shared.TensorImage api (placement b p)
        (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial ∧
      InputFrame api b p
        (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial ∧
      (∀ r off word,KernelCompiler.Expected platform binding (KernelCompiler.hostScale platform s arg) p
        (logical b (nativeRead api initial) u) r off word →
        (VeriTile.Torch.FP32.logicalValue (fun a => (⟨api.read (Shared.commit api (placement b p)
          (fun _ _ => observe (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)) v initial) a⟩ :
            VeriTile.Torch.FP32.Word)) (b.tensor r) (b.laws r) off).bits=word) :=
  KernelBound.native_tensor api initial platform kind s f binding arg scale b abi p u hv hk hh

#axiomsClean kernel_physical_logical_input
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.TensorBridge

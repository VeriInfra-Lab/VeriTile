import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.RawReal
import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.Compatibility
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.CheckedHost
import bench.repairs.tritonbench_g.chunk_gla_fwd.Correct
import bench.repairs.tritonbench_g.chunk_gla_fwd.FPEquiv

/-! Optional checked host/Tensor applications of the preserved proof modules. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.TensorBridge
open VeriTile Triton NativeInstructions
noncomputable section
open NativeInstructions

/-- Concrete FP32 words obtained from the fixed local scalar/instruction ISA.
No complete dot, reduction, kernel, or tensor result is an ISA field. -/
def words (I : ISA (BitVec 32)) (r : CheckedHost.Returned) : Nat → BitVec 32 :=
  fun p => (r.memory p).eval I

def Values (I : ISA (BitVec 32)) (r : Admission.Request)
    (a : {c : Host.Dimensions // Admission.Contract r c}) (rt : CheckedHost.Runtime r a.val)
    (before : Nat → BitVec 32) (result : CheckedHost.Returned) : Prop :=
  ∃ allocation w,rt.allocate=.ok allocation ∧ result.objectId=allocation.objectId ∧
    result.tensor=allocation.tensor ∧
    (∀ p,NativeRuntime.Outside r.mode allocation.abi p → words I result p=I.bits (before p)) ∧
    (match r.mode with
    | .producer => ∀ (h : Fin a.val.heads) (t : Fin a.val.T) (j : Fin a.val.BT),
        words I result ((allocation.abi.alloc "attention").base+(h.val*a.val.T*a.val.BT+t.val*a.val.BT+j.val))=
          (NativeRuntime.producerValue w r.scale.runtimeWord allocation.abi before h t j).eval I
    | .consumer => ∀ (h : Fin a.val.heads) (t : Fin a.val.T) (v : Fin a.val.V),
        words I result ((allocation.abi.alloc "o").base+h.val*(a.val.T*a.val.V)+t.val*a.val.V+v.val)=
          (NativeRuntime.consumerValue w r.scale.runtimeWord allocation.abi before h t v).eval I)

theorem successful_values (I : ISA (BitVec 32)) (r : Admission.Request)
    (a : {c : Host.Dimensions // Admission.Contract r c}) (rt : CheckedHost.Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) (result : CheckedHost.Returned)
    (post : CheckedHost.ResultSpec r a rt before caller (.ok result)) : Values I r a rt before result := by
  rcases post with ⟨allocation,w,allocated,identity,tensor,device,launch,source,memory,frame,inputs⟩
  refine ⟨allocation,w,allocated,identity,tensor,?_,?_⟩
  · intro p hp
    exact congrArg (Graph.eval I) (frame p hp)
  · unfold words
    rw [memory]
    cases mode : r.mode with
    | producer =>
      intro h t j
      exact congrArg (Graph.eval I) (NativeRuntime.producer_readback w r.scale.runtimeWord allocation.abi before h t j)
    | consumer =>
      intro h t v
      exact congrArg (Graph.eval I) (NativeRuntime.consumer_readback w r.scale.runtimeWord allocation.abi before h t v)

/-- Whole checked host behavior and all actual returned FP32 words for either
public call. ISA bits/literals/scalar ops/casts/FMA/fixed MMA are the only local
numerical boundary, with ordered reduction and dot structure already derived. -/
theorem wrapper_fp32 (I : ISA (BitVec 32)) (r : Admission.Request)
    (a : {c : Host.Dimensions // Admission.Contract r c}) (rt : CheckedHost.Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) :
    CheckedHost.ResultSpec r a rt before caller (CheckedHost.runAdmitted r a rt before caller) ∧
    (∀ result,CheckedHost.runAdmitted r a rt before caller=.ok result → Values I r a rt before result) := by
  have complete:=CheckedHost.complete r a rt before caller
  refine ⟨complete,?_⟩
  intro result success
  rw [success] at complete
  exact successful_values I r a rt before caller result complete

/-- Existing checked host behavior, separate from primary kernel specifications. -/
theorem wrapper_correct (r : Admission.Request)
    (rt : ∀ a : {c : Host.Dimensions // Admission.Contract r c},CheckedHost.Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) :
    (match Admission.admit r with
    | .error _ => CheckedHost.run r rt before caller=.error ⟨.admission,caller⟩
    | .ok a => CheckedHost.ResultSpec r a (rt a) before caller (CheckedHost.run r rt before caller)) :=
  CheckedHost.wrapper_complete r rt before caller

theorem rejected (r : Admission.Request)
    (rt : ∀ a : {c : Host.Dimensions // Admission.Contract r c},CheckedHost.Runtime r a.val)
    (before : Nat → BitVec 32) (caller : Nat) (why : String)
    (bad : Admission.admit r=.error why) :
    CheckedHost.run r rt before caller=.error ⟨.admission,caller⟩ :=
  CheckedHost.rejected r rt before caller why bad

/-- Kernel-only public result for exactly the unchanged admitted request.
All source launches are constructed. Every coordinate of the returned Tensor
has the independent mathematical value, and all other typed cells are framed. -/
theorem kernel_correct (r : Admission.Request) (shape : Admission.ShapeValid r.dims)
    (_admitted : Admission.Contract r (r.dims.checked shape))
    (view : KernelTensor.Output r.mode (r.dims.checked shape) r.query.device)
    (scale : ℝ) (initial : VeriTile.Triton.RegionName → Nat → ℝ) :
    VeriTile.Spec.Real (∃ final,
      KernelReal.sourceExecution r.mode (r.dims.checked shape) scale (KernelReal.entry initial) final ∧
      KernelReal.Post r.mode (r.dims.checked shape) view scale (KernelReal.entry initial) final) :=
  KernelReal.complete r.mode (r.dims.checked shape) view scale initial

/-- Complete shared-source native execution and actual returned Tensor words.
No host/allocator implementation, successful run, complete result, reduction
or variable-size dot is a premise of this kernel-only specification. -/
theorem kernel_equiv (r : Admission.Request) (shape : Admission.ShapeValid r.dims)
    (_admitted : Admission.Contract r (r.dims.checked shape))
    (I : ISA (BitVec 32)) (bits : ∀ b,I.bits b=b) (w : KernelProfile.Profile)
    (abi : Physical.ABI (r.dims.checked shape))
    (_inputs : KernelNative.InputBinding r abi) (fresh : KernelRuntime.Fresh r.mode abi)
    (view : KernelTensor.Output r.mode (r.dims.checked shape) r.query.device)
    (base : view.info.byteBase=4*(abi.alloc (KernelTensor.region r.mode)).base)
    (before : Nat → BitVec 32) :
    ∃ final,
      KernelNative.execute I r.mode w r.scale.runtimeWord abi before=some final ∧
      KernelNative.Post I r.mode w r.scale.runtimeWord abi view before final :=
  KernelNative.complete I bits r.mode w r.scale.runtimeWord abi fresh view base before

#axiomsClean kernel_correct
#axiomsClean kernel_equiv
#axiomsClean successful_values
#axiomsClean wrapper_fp32
#axiomsClean wrapper_correct
#axiomsClean rejected
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.TensorBridge

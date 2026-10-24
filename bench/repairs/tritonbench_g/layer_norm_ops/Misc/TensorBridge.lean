import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelBound

/-! Optional actual Tensor applications and Python-check correspondence for
the forward/backward kernels. The primary Real, source-graph and selected-object
kernel results remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.TensorBridge
open VeriTile.Triton FP.Structural KernelInputs KernelInputs.Shared KernelNative.Shared KernelBound
open Admission
noncomputable section

theorem forward_checks_iff (c : Admission.ForwardCall) :
    Admission.forwardCheck c = Bool.true ↔ Admission.ForwardContract c :=
  Admission.forwardCheck_iff c

theorem backward_checks_iff (c : Admission.BackwardCall) :
    Admission.backwardCheck c = Bool.true ↔ Admission.BackwardContract c :=
  Admission.backwardCheck_iff c

theorem layer_norm_ops_forward_tensor_real {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API ℝ Store Byte) (initial : Store)
    (rest : Algebra ℝ) (c : Admission.ForwardCall) (check : Admission.forwardCheck c=Bool.true)
    (inputs : ForwardBinding c) (u : State ℝ) (rowBound : u.pids 0<c.common.rows)
    (outputs : Storage.Placement (forwardFootprint (KernelDispatch.forward c) u))
    (fresh : Fresh (forwardViews (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check)) outputs) :
    let p := KernelDispatch.forward c
    let views := forwardViews (api.read initial) c inputs ((Admission.forwardCheck_iff c).mp check)
    let observe : RegionName → Nat → Cell ℝ → ℝ := fun _ _ cell => cell.read .fp32
    ForwardReads views c (u.pids 0) ∧ Pointers views ∧ ∃ t,
      (KernelExecution.ForwardFacts (Logical.algebra rest) p (physical views u) t ∧
        KernelReal.ForwardOutputs p (logical views u) t) ∧
      Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial := by
  exact KernelBound.real_forward_tensor api initial rest c check inputs u rowBound outputs fresh

theorem layer_norm_ops_backward_tensor_real {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API ℝ Store Byte) (initial : Store)
    (rest : Algebra ℝ) (c : Admission.BackwardCall) (check : Admission.backwardCheck c=Bool.true)
    (inputs : BackwardBinding c) (u : State ℝ) (programBound : u.pids 0<c.programs)
    (outputs : Storage.Placement (backwardFootprint (KernelDispatch.backward c) u))
    (fresh : Fresh (backwardViews (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check)) outputs) :
    let p := KernelDispatch.backward c
    let views := backwardViews (api.read initial) c inputs ((Admission.backwardCheck_iff c).mp check)
    let observe : RegionName → Nat → Cell ℝ → ℝ := fun _ _ cell => cell.read .fp32
    ParameterReads views c.common ∧ u.pids 0<c.programs ∧
      (∀ row,row<c.common.rows → BackwardReads views c row) ∧
      Pointers views ∧ ∃ t,
      (KernelExecution.BackwardFacts (Logical.algebra rest) p (physical views u) t ∧
        KernelReal.BackwardOutputs p (logical views u) t) ∧
      Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial := by
  exact KernelBound.real_backward_tensor api initial rest c check inputs u programBound outputs fresh

theorem layer_norm_ops_forward_tensor_equiv {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (c : Admission.ForwardCall) (check : Admission.forwardCheck c=Bool.true)
    (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inl (KernelDispatch.forward c)) host R S)
    (conversion : KernelCompiler.Conversion platform (.inl (KernelDispatch.forward c)) host binding)
    (inputs : ForwardBinding c) (u : State Term) (rowBound : u.pids 0<c.common.rows)
    (outputs : Storage.Placement (forwardFootprint (KernelDispatch.forward c) u))
    (fresh : Fresh (forwardViews (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check)) outputs) :
    let p := KernelDispatch.forward c
    let views := forwardViews (words (api.read initial)) c inputs ((Admission.forwardCheck_iff c).mp check)
    let observe : RegionName → Nat → Cell Term → Word :=
      fun r i => KernelNative.observe (platform.primitives binding.object.primitives) (S r i) c.common.columns
    ForwardReads views c (u.pids 0) ∧
      (Pointers views ∧ KernelNative.CountFacts (platform.primitives binding.object.primitives) p.columns ∧
        ∀ schedule,eval (platform.primitives binding.object.primitives) schedule
          (.literal (some .fp32) .real p.epsilon)=platform.epsilon binding.object host) ∧
      ∃ t,
      (KernelExecution.ForwardFacts (algebra R) p (physical views u) t ∧
        KernelNative.ForwardOutputs R (platform.primitives binding.object.primitives)
          (withEpsilon (platform.primitives binding.object.primitives) p.epsilon (platform.epsilon binding.object host))
          S p (logical views u) t) ∧
      Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial ∧
      (∀ r i,forwardFootprint p u r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (Storage.commit api outputs observe t initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe r i (t.mem r i)) := by
  exact KernelBound.native_forward_tensor api initial platform c check host R S binding conversion
    inputs u rowBound outputs fresh

theorem layer_norm_ops_backward_tensor_equiv {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (c : Admission.BackwardCall) (check : Admission.backwardCheck c=Bool.true)
    (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inr (KernelDispatch.backward c)) host R S)
    (conversion : KernelCompiler.CountConversion platform (.inr (KernelDispatch.backward c)) host binding)
    (inputs : BackwardBinding c) (u : State Term) (programBound : u.pids 0<c.programs)
    (outputs : Storage.Placement (backwardFootprint (KernelDispatch.backward c) u))
    (fresh : Fresh (backwardViews (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check)) outputs) :
    let p := KernelDispatch.backward c
    let views := backwardViews (words (api.read initial)) c inputs ((Admission.backwardCheck_iff c).mp check)
    let observe : RegionName → Nat → Cell Term → Word :=
      fun r i => KernelNative.observe (platform.primitives binding.object.primitives) (S r i) c.common.columns
    ParameterReads views c.common ∧ u.pids 0<c.programs ∧
      (∀ row,row<c.common.rows → BackwardReads views c row) ∧
      (Pointers views ∧ KernelNative.CountFacts (platform.primitives binding.object.primitives) p.columns) ∧
      ∃ t,
      (KernelExecution.BackwardFacts (algebra R) p (physical views u) t ∧
        KernelNative.BackwardOutputs R (platform.primitives binding.object.primitives) S p (logical views u) t) ∧
      Storage.TensorImage api outputs observe t initial ∧
      InputFrame api views outputs observe t initial ∧
      (∀ r i,backwardFootprint p u r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (Storage.commit api outputs observe t initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe r i (t.mem r i)) := by
  exact KernelBound.native_backward_tensor api initial platform c check host R S binding conversion
    inputs u programBound outputs fresh


/-- Empty-range and finite-index consequences of the actual checked call. -/
theorem admitted_empty_backward_partials {α : Type} (c : BackwardCall) (empty : c.common.rows=0)
    (A : Algebra α) [Inhabited α] (s : State α) :
    KernelReference.partials A (KernelDispatch.backward c) s=
      (fun _ => Forward.literal A 0,fun _ => Forward.literal A 0) :=
  KernelDispatch.backward_empty_range c empty A s

theorem admitted_forward_indices (c : ForwardCall) (checked : forwardCheck c=Bool.true)
    (row lane : Nat) (hr : row<c.common.rows) (hl : lane<blockSize c.common.columns) :
    row*c.common.columns+lane<2^31 := KernelIndex.forward c checked row lane hr hl

theorem admitted_backward_indices (c : BackwardCall) (checked : backwardCheck c=Bool.true)
    (pid lane : Nat) (hp : pid<c.programs) (hl : lane<blockSize c.common.columns) :
    pid*c.rowWidth<2^31 ∧ (pid+1)*c.rowWidth<2^31 ∧
    pid*c.rowWidth*c.common.columns+lane<2^31 ∧ pid*c.common.columns+lane<2^31 :=
  KernelIndex.backward c checked pid lane hp hl

#axiomsClean admitted_empty_backward_partials
#axiomsClean admitted_forward_indices
#axiomsClean admitted_backward_indices

#axiomsClean forward_checks_iff
#axiomsClean backward_checks_iff
#axiomsClean layer_norm_ops_forward_tensor_real
#axiomsClean layer_norm_ops_backward_tensor_real
#axiomsClean layer_norm_ops_forward_tensor_equiv
#axiomsClean layer_norm_ops_backward_tensor_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormOps.TensorBridge

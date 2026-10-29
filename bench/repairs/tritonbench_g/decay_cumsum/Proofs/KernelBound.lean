import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.DecayCumsum.KernelBound
open VeriTile.Triton FP.Structural Geometry KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

def Pointers (b : Buffers h kind) (p : Phase) : Prop :=
  ∀ r∈argumentRoles kind,0<(b.tensor r).metadata.numel →
    sourcePointer b p r=(b.tensor r).dataPtr+BitVec.ofNat 64 (4*shift h p)
theorem pointers (b : Buffers h kind) (p : Phase) : Pointers b p :=
  fun r _ hn => tail_pointer b p r hn

/-- Exact selected-object pointer specialization metadata. Divisors are
object data, never an extra fixed alignment condition on admitted callers. -/
def PointerABI (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind h p)
    (b : Buffers h kind) : Prop :=
  List.Forall₂ (fun divisor address => 0<divisor ∧ address%divisor=0)
    binding.object.signature.pointerDivisibility
    ((argumentRoles kind).map fun r => (sourcePointer b p r).toNat)

def Reads (b : Buffers h kind) (p : Phase) (prog : Program h p) : Prop :=
  ∀ r∈argumentRoles kind,∀ (t : Fin p.span) (j : Fin h.BK),
    shift h p+FPPrefix.off h p prog t.val j<(b.tensor r).metadata.numel

theorem real_tensor {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
    (rest : Algebra ℝ) (h : Call) (kind : Kind) (observed : RegionName → VeriTile.Torch.TensorMetadata)
    (scaleIsFloat : Bool) (check : checkInput h kind observed scaleIsFloat=.ok ())
    (p : Phase) (hp : p∈phases h) (prog : Program h p) (scale : ℝ)
    (b : Buffers h kind) (out : Outputs b p prog) (isolated : Isolated b p prog out) (u : State ℝ)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    InputContract h kind observed scaleIsFloat ∧ Pointers b p ∧ Reads b p prog ∧ ∃ v,
      KernelExecution.Facts (KernelReal.algebra rest) kind h p prog scale (physical b p (api.read initial) u) v ∧
      KernelReal.Outputs kind h p prog scale (logical b p (api.read initial) u) v ∧
      Shared.TensorImage api out.toPlacement (fun _ _ cell => cell.read .real) (relocated h p v) initial ∧
      InputFrame api b p prog out (fun _ _ cell => cell.read .real) v initial ∧
      (∀ r∈FPTile.outputs kind,∀ (t : Fin p.span) (j : Fin h.BK),
        api.read (Shared.commit api out.toPlacement (fun _ _ cell => cell.read .real) (relocated h p v) initial)
          ((b.tensor r).storage.address ((b.tensor r).physicalIndex (Geometry.offset h p prog t.val j.val)))=
          KernelReal.evaluate kind p.span scale
            (KernelExecution.read h p prog (logical b p (api.read initial) u) j) r t.val) := by
  obtain ⟨v,facts,values⟩ := KernelReal.complete rest kind h p hp prog scale
    (physical b p (api.read initial) u) hf hc hr
  rw [physical_eq_logical] at values
  refine ⟨(KernelContract.checkInput_iff h kind observed scaleIsFloat).mp check,
    pointers b p,coverage b p hp prog,v,facts,values,Shared.complete api out.toPlacement _ _ initial,
    input_frame api b p prog out isolated _ v initial,?_⟩
  intro r member t j
  rw [real_output api b p prog out v initial r member t j,values r member t j]
  rfl

theorem native_tensor {Store Byte : Type} (api : Memory.API Native.Word Store Byte) (initial : Store)
    (platform : KernelCompiler.Platform) (h : Call) (kind : Kind)
    (observed : RegionName → VeriTile.Torch.TensorMetadata) (scaleIsFloat : Bool)
    (check : checkInput h kind observed scaleIsFloat=.ok ())
    (p : Phase) (hp : p∈phases h) (prog : Program h p)
    (binding : KernelCompiler.Binding platform kind h p) (host : UInt64)
    (b : Buffers h kind) (abi : PointerABI platform binding b)
    (out : Outputs b p prog) (isolated : Isolated b p prog out) (u : State Native.Node)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    InputContract h kind observed scaleIsFloat ∧ Pointers b p ∧ Reads b p prog ∧ PointerABI platform binding b ∧
      ((KernelCompiler.algebra platform binding host).literal (some .fp32) .real Native.scaleSlot).value=
        platform.scale binding.object host ∧ ∃ v,
      KernelCompiler.Facts platform binding host prog
        (physical b p (fun a=>Native.atom (api.read initial a)) u) v ∧
      KernelNative.Outputs (platform.instructions binding.object.primitives) kind h p prog
        (platform.scale binding.object host) (logical b p (fun a=>Native.atom (api.read initial a)) u) v ∧
      Shared.TensorImage api out.toPlacement (fun _ _ => KernelNative.observe) (relocated h p v) initial ∧
      InputFrame api b p prog out (fun _ _ => KernelNative.observe) v initial ∧
      (∀ r∈FPTile.outputs kind,∀ (t : Fin p.span) (j : Fin h.BK),
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (Shared.commit api out.toPlacement (fun _ _ => KernelNative.observe)
            (relocated h p v) initial) a⟩ : VeriTile.Torch.FP32.Word)) (b.tensor r) (b.laws r)
          (Geometry.offset h p prog t.val j.val)).bits=
          Native.evaluateWords (platform.instructions binding.object.primitives) kind p.span
            (platform.scale binding.object host)
            (fun s n => (KernelExecution.read h p prog
              (logical b p (fun a=>Native.atom (api.read initial a)) u) j s n).value) r t.val) := by
  obtain ⟨v,facts,values⟩ := KernelCompiler.complete platform kind h p binding host hp prog
    (physical b p (fun a=>Native.atom (api.read initial a)) u) hf hc hr
  rw [physical_eq_logical] at values
  refine ⟨(KernelContract.checkInput_iff h kind observed scaleIsFloat).mp check,
    pointers b p,coverage b p hp prog,abi,KernelCompiler.runtime_scale_value platform binding host,
    v,facts,values,Shared.complete api out.toPlacement _ _ initial,
    input_frame api b p prog out isolated _ v initial,?_⟩
  intro r member t j
  simp only [VeriTile.Torch.FP32.logicalValue,(b.checked r).noNeg,Bool.false_eq_true,if_false]
  rw [native_output api b p prog out v initial r member t j,values r member t j]
  rfl

#axiomsClean real_tensor
#axiomsClean native_tensor
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelBound

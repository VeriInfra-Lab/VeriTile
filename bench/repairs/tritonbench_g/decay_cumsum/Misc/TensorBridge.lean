import bench.repairs.tritonbench_g.decay_cumsum.Misc.KernelRealResults
import bench.repairs.tritonbench_g.decay_cumsum.Misc.KernelNativeResults
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelBound

/-! Optional actual-Tensor applications, pointer binding and check correspondence. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.TensorBridge
open VeriTile.Triton FP.Structural Geometry KernelTensor KernelBound
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

theorem decay_cumsum_tensor_real {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
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
            (KernelExecution.read h p prog (logical b p (api.read initial) u) j) r t.val) :=
  KernelBound.real_tensor api initial rest h kind observed scaleIsFloat check p hp prog scale b out isolated u hf hc hr

theorem decay_cumsum_tensor_equiv {Store Byte : Type} (api : Memory.API Native.Word Store Byte) (initial : Store)
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
              (logical b p (fun a=>Native.atom (api.read initial a)) u) j s n).value) r t.val) :=
  KernelBound.native_tensor api initial platform h kind observed scaleIsFloat check p hp prog binding host b abi out isolated u hf hc hr

theorem decay_cumsum_native_addresses (b : Buffers h kind) (p : Phase) (hp : p∈phases h)
    (contract : InputContract h kind observed scaleIsFloat) (prog : Program h p)
    (r : RegionName) (hr : r∈argumentRoles kind) (t : Fin p.span) (j : Fin h.BK) :
    let pid := KernelContract.pid contract (KernelIndex.nonempty h p hp prog) p hp prog
    (NativeIndex.offsetWord h p pid t.val j.val).toInt=
      (NativeIndex.relative h p pid t.val j.val:Int) ∧
    (sourcePointer b p r+NativeIndex.offsetWord h p pid t.val j.val*BitVec.ofNat 64 4).toNat=
      (b.view r).address (Geometry.offset h p prog t.val j.val) ∧
    (b.view r).address (Geometry.offset h p prog t.val j.val)+4≤2^64 :=
  KernelIndex.addresses b p hp contract prog r hr t j

theorem decay_cumsum_check_iff (h : Call) (kind : Kind) (observed : RegionName → VeriTile.Torch.TensorMetadata)
    (scaleIsFloat : Bool) : checkInput h kind observed scaleIsFloat=.ok () ↔ InputContract h kind observed scaleIsFloat :=
  KernelContract.checkInput_iff h kind observed scaleIsFloat

theorem decay_cumsum_loop_bounds (b : Buffers h kind) (p : Phase) (hp : p∈phases h)
    (prog : Program h p) (r : RegionName) (hr : r∈argumentRoles kind) :
    0<p.span ∧ p.span<2^63 ∧ (BitVec.ofNat 64 p.span).toInt=(p.span:Int) ∧
      (BitVec.ofNat 64 (p.span-1)).toInt=(p.span-1:Nat) :=
  KernelIndex.loop_bounds b p hp prog r hr

#axiomsClean decay_cumsum_loop_bounds
#axiomsClean decay_cumsum_tensor_real
#axiomsClean decay_cumsum_tensor_equiv
#axiomsClean decay_cumsum_native_addresses
#axiomsClean decay_cumsum_check_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.TensorBridge

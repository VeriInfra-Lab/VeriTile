import bench.repairs.tritonbench_g.rotary_emb.Correct
import bench.repairs.tritonbench_g.rotary_emb.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rotary_emb.FPEquiv
import bench.repairs.tritonbench_g.rotary_emb.Proofs.RealBinding
import bench.repairs.tritonbench_g.rotary_emb.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.rotary_emb.Proofs.KernelNativeResult

/-! Optional applications of the shared typed kernel to actual Tensor objects. -/
namespace VeriTile.Bench.Repairs.RotaryEmb.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Real correctness bound to actual Tensor objects and current caller words.
The explicit word interpretation is total and unconstrained; this is the
mathematical kernel statement, with no FP-to-Real arithmetic premise. -/
theorem rotary_emb_checked_real_correct {M : Type}
    (api : Binding.API M Native.Word Physical.Byte) (rest : Algebra ℝ)
    (c : Checked.Call) (accepted : Checked.check c=Bool.true)
    (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ)
    (m : M) (surrounding : State ℝ) :
    Spec.Real (let s := RealBinding.state api c decode m surrounding
      ∃ t,exec (Typed.algebra rest realOps) (Kernels.kernel (Typed.layout c.shape)) s=some t ∧
        t.mem=(Typed.target realOps c.shape s).mem ∧
        (∀ i o,o<Checked.extent c.shape i →
          decode (api.read m (Binding.tensorKey (c.tensor i) o))=
            RealBinding.logical api c objects decode m i o) ∧
        (∀ key odd i,Typed.active (Typed.layout c.shape) surrounding key i →
          t.mem (Typed.region (Typed.layout c.shape) key) (Typed.offset (Typed.layout c.shape) s key odd i)=
            .mk .real (RealBinding.logicalPair api c objects decode m
              (Typed.work c.shape surrounding key i) odd)) ∧
        (∀ r o,Typed.Outside c.shape s r o → t.mem r o=s.mem r o)) := by
  dsimp only
  obtain ⟨t,source,memory,values,frame⟩ := Typed.successful rest realOps c.shape
    (Checked.shape_valid c accepted) (RealBinding.state api c decode m surrounding)
  refine ⟨t,source,memory,RealBinding.direct api c objects decode m accepted,?_,frame⟩
  intro key odd i live
  exact (values key odd i live).trans
    (RealBinding.value_logical api c objects decode m accepted surrounding key odd i live)

/-- The public binding uses actual Torch objects and current caller memory.
It connects every typed output to the physical complete-grid result and
preserves all bytes outside the admitted stores. -/
theorem rotary_emb_checked_fp32_equiv {M : Type}
    (api : Binding.API M Native.Word Physical.Byte) (rest : Algebra Native.Node)
    (p : Native.Instructions) (c : Checked.Call) (accepted : Checked.check c=Bool.true)
    (objects : TypedBinding.Objects c) (m : M) (surrounding : State Native.Node) :
    let s := TypedBinding.state api c m surrounding
    ∃ t,exec (Typed.algebra rest (Native.ops p)) (Kernels.kernel (Typed.layout c.shape)) s=some t ∧
      t.mem=(Typed.graphTarget p c.shape s).mem ∧
      (∀ r o,Typed.Outside c.shape s r o → t.mem r o=s.mem r o) ∧
      (∀ i o,o<Checked.extent c.shape i →
        api.read m (Binding.tensorKey (c.tensor i) o)=TypedBinding.logical api c objects m i o) ∧
      (∀ key odd i,Typed.active (Typed.layout c.shape) surrounding key i →
        api.read (Binding.execute api p c m) (Binding.key c (Typed.work c.shape surrounding key i,odd))=
          ((t.mem (Typed.region (Typed.layout c.shape) key)
            (Typed.offset (Typed.layout c.shape) surrounding key odd i)).read .real).value) ∧
      (∀ owner address,Binding.Outside c owner address →
        api.observe (Binding.execute api p c m) owner address=api.observe m owner address) := by
  dsimp only
  obtain ⟨t,source,memory,_,frame⟩ := Typed.native_successful rest p c.shape
    (Checked.shape_valid c accepted) (TypedBinding.state api c m surrounding)
  have same : t=(Typed.execution rest (Native.ops p) (Typed.layout c.shape)
      (TypedBinding.state api c m surrounding)).val :=
    Option.some.inj (source.symm.trans (Typed.execution rest (Native.ops p) (Typed.layout c.shape)
      (TypedBinding.state api c m surrounding)).property)
  refine ⟨t,source,memory,frame,TypedBinding.direct api c accepted objects m,?_,Binding.frame api p c m⟩
  subst t
  exact TypedBinding.physical_agreement api p c accepted m surrounding rest

theorem rotary_emb_correct (rest : Algebra ℝ) (c : Checked.Call)
    (accepted : Checked.check c=Bool.true) (s : State ℝ) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.kernel (Typed.layout c.shape)) s=some t ∧
      t.mem=(Typed.target realOps c.shape s).mem ∧
      (∀ key odd i,Typed.active (Typed.layout c.shape) s key i →
        t.mem (Typed.region (Typed.layout c.shape) key) (Typed.offset (Typed.layout c.shape) s key odd i)=
          .mk .real (formula c.shape (Typed.inputs s) key
            (Typed.token (Typed.layout c.shape) s i) (Typed.head (Typed.layout c.shape) s i)
            (Typed.dimension (Typed.layout c.shape) odd i))) ∧
      (∀ r o,Typed.Outside c.shape s r o → t.mem r o=s.mem r o)) :=
  KernelRealResult.rotary_emb_real rest c.shape (Checked.shape_valid c accepted) s

theorem rotary_emb_fp32_equiv (rest : Algebra Native.Node) (p : Native.Instructions)
    (c : Checked.Call) (accepted : Checked.check c=Bool.true) (s : State Native.Node) :
    ∃ t,exec (Typed.algebra rest (Native.ops p)) (Kernels.kernel (Typed.layout c.shape)) s=some t ∧
      exec (Typed.algebra rest (Native.ops p)) (Candidate.optimized (Typed.layout c.shape)) s=some t ∧
      t.mem=(Typed.graphTarget p c.shape s).mem ∧
      (∀ key odd i,Typed.active (Typed.layout c.shape) s key i →
        t.mem (Typed.region (Typed.layout c.shape) key) (Typed.offset (Typed.layout c.shape) s key odd i)=
          Typed.graphValue p c.shape s key odd i) ∧
      (∀ r o,Typed.Outside c.shape s r o → t.mem r o=s.mem r o) :=
  KernelNativeResult.rotary_emb_native_equiv rest p c.shape (Checked.shape_valid c accepted) s

#axiomsClean rotary_emb_correct
#axiomsClean rotary_emb_fp32_equiv
#axiomsClean rotary_emb_checked_real_correct
#axiomsClean rotary_emb_checked_fp32_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.TensorBridge

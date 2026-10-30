import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Domain
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Real
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section

/-- Whole forward/helper contract on the supported FP32 affine input domain.
Every output, optional final state, saved context and outside cell is covered.
No backward or jvp exists in this source. -/
specification correct (c : Shape) (hc : Domain c) (plan : Plans c) (pythonScale scale : ℝ) (memory : Memory ℝ) :
    (∀ z, z<c.Z → ∀ t, t<c.T → ∀ f, f<c.V →
      (wrapper realOps c plan pythonScale scale memory).output z t f=
      realOutput c (inputs c memory) scale z (visit c t) f) ∧
    (c.final=Bool.true → ∀ z, z<c.Z → ∀ e, e<c.K → ∀ f, f<c.V →
      forward realOps c plan scale memory .ht (stateAddress c z e f)=
      realState c (inputs c memory) z e f c.T) ∧
    (∀ r b, ¬WholeOutput c r b → forward realOps c plan scale memory r b=memory r b) ∧
    (wrapper realOps c plan pythonScale scale memory).saved=inputs c memory ∧
    (wrapper realOps c plan pythonScale scale memory).savedScale=pythonScale ∧
    (wrapper realOps c plan pythonScale scale memory).savedReverse=c.reverse ∧
    (wrapper realOps c plan pythonScale scale memory).finalState.isSome=c.final := by
  refine ⟨?_,?_,?_,wrapper_context realOps c plan pythonScale scale memory⟩
  · intro z hz t ht f hf
    exact real_returned c hc.sizes plan scale memory z t f hz ht hf
  · intro hs z hz e he f hf
    exact real_terminal c hc.sizes plan scale memory hs z e f hz he hf
  · exact complete_frame realOps c hc.sizes plan scale memory

/-- The public helper selects forward traversal and delegates the same complete
execution; Python's K**-0.5 result and FP32 launch binding are ordinary scalar
runtime boundaries. -/
theorem helper_correct (c : Shape) (hc : Domain {c with reverse:=Bool.false})
    (plan : Plans {c with reverse:=Bool.false}) (request defaultScale : ℝ) (bind : ℝ → ℝ)
    (memory : Memory ℝ) (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    (helper realOps c plan request defaultScale bind memory).output z t f=
      realOutput {c with reverse:=Bool.false} (inputs {c with reverse:=Bool.false} memory)
        (bind (helperScale request defaultScale)) z t f := by
  exact real_returned {c with reverse:=Bool.false} hc.sizes plan _ memory z t f hz ht hf

/-- The helper inherits the complete pair result, frame, and original Python
context value after selecting its sentinel. -/
specification helper_whole_correct (c : Shape) (hc : Domain {c with reverse:=Bool.false})
    (plan : Plans {c with reverse:=Bool.false}) (request defaultScale : ℝ) (bind : ℝ → ℝ)
    (memory : Memory ℝ) :
    (∀ z, z<c.Z → ∀ t, t<c.T → ∀ f, f<c.V →
      (helper realOps c plan request defaultScale bind memory).output z t f=
      realOutput {c with reverse:=Bool.false} (inputs {c with reverse:=Bool.false} memory)
        (bind (helperScale request defaultScale)) z t f) ∧
    (c.final=Bool.true → ∀ z, z<c.Z → ∀ e, e<c.K → ∀ f, f<c.V →
      forward realOps {c with reverse:=Bool.false} plan (bind (helperScale request defaultScale))
        memory .ht (stateAddress c z e f)=
      realState {c with reverse:=Bool.false} (inputs {c with reverse:=Bool.false} memory) z e f c.T) ∧
    (∀ r b, ¬WholeOutput c r b →
      forward realOps {c with reverse:=Bool.false} plan (bind (helperScale request defaultScale))
        memory r b=memory r b) ∧
    (helper realOps c plan request defaultScale bind memory).saved=inputs c memory ∧
    (helper realOps c plan request defaultScale bind memory).savedScale=helperScale request defaultScale ∧
    (helper realOps c plan request defaultScale bind memory).savedReverse=Bool.false ∧
    (helper realOps c plan request defaultScale bind memory).finalState.isSome=c.final := by
  exact correct {c with reverse:=Bool.false} hc plan (helperScale request defaultScale)
    (bind (helperScale request defaultScale)) memory

#axiomsClean correct
#axiomsClean helper_whole_correct
#axiomsClean helper_correct
#axiomsClean unique_owner
#axiomsClean cover
#specNonCircular realOutput avoiding [runLoop,runProgram,runGrid,forward,wrapper]
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel

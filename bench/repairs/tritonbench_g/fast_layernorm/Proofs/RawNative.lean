import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RawValues
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelNative

/-! Canonical typed source execution versus independent ordered native scalar
graphs. Reduction plans list every lane exactly once and explicit neutral leaves.
Compiler operand order and FMA/subtraction choices remain data. Primitive FP32
rounding, FTZ and NaN policies are target-model parameters, not experimental
transformation assumptions. No finite-only premise or output oracle is used.
Theorems take raw kernel states and include all outputs, entire source memory, and
the exterior frame. Historical host results remain in Proofs/HostFPEquiv. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Equivalence
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelNative
noncomputable section

def nativeForwardReference (R : Reductions) (I : Instructions) (S : Schedule)
    (N : Nat) (eps : ℝ) (bias : Bool) (x w b : Fin B → Term) (j : Fin B) :=
  observe I S (KernelReference.output (algebra R) N eps bias x w b j)
def nativeGradientReference (R : Reductions) (I : Instructions) (S : Schedule)
    (N : Nat) (x dy w : Fin B → Term) (avg inv : Term) (j : Fin B) :=
  observe I S (KernelReference.gradient (algebra R) N x dy w avg inv j)

theorem forward_graph_equiv_raw (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (eps : ℝ) (bias : Bool)
    (u : State Term) :
    ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias u ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R I (S "y" (u.pids 0*N+j.val)) N eps bias (fun k => RawValues.value u "x" (u.pids 0*N+k.val))
          (fun k => RawValues.value u "w" k.val) (fun k => RawValues.value u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=observe I (S "r" (u.pids 0))
        (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hy,hr,hmu,hf⟩ := KernelOutputs.forward_complete (algebra R) N B eps bias u
  simp only [RawValues.raw,Nat.mul_zero,Nat.zero_add] at hy hr hmu
  exact ⟨v,he,hm,fun j hj => congrArg (observe I (S "y" (u.pids 0*N+j.val))) (hy j hj),
    congrArg (observe I (S "r" (u.pids 0))) hr,congrArg (observe I (S "mu" (u.pids 0))) hmu,hf⟩

/-- The independent reference receives the actual converted epsilon word.
The only additional obligation identifies that one scalar leaf; the proof
uses it to rewrite the native output and saved-inverse reference, including
coinciding literals, without changing the source graph or compiler schedule. -/
theorem forward_scalar_graph_equiv_raw (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (eps : ℝ) (epsilon : Word) (bias : Bool)
    (conversion : literal32 I eps=epsilon)
    (u : State Term) :
    (∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias u ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => RawValues.value u "x" (u.pids 0*N+k.val))
          (fun k => RawValues.value u "w" k.val) (fun k => RawValues.value u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  rw [withEpsilon_eq I eps epsilon conversion]
  obtain ⟨v,hv⟩ := forward_graph_equiv_raw R I S N B eps bias u
  exact ⟨fun schedule => (eval_literal32 I schedule eps).trans conversion,v,hv⟩

theorem backward_graph_equiv_raw (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (u : State Term) :
    ∃ v,
      FP.Structural.exec (engine (algebra R))
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u=some v ∧
      v.mem=KernelRows.backwardHeap (algebra R) N B u ∧
      (∀ j : Fin B,j.val<N → observe I (S "dx" (u.pids 0*N+j.val)) (v.mem "dx" (u.pids 0*N+j.val))=
        nativeGradientReference R I (S "dx" (u.pids 0*N+j.val)) N (fun k => RawValues.value u "x" (u.pids 0*N+k.val))
          (fun k => RawValues.value u "dy" (u.pids 0*N+k.val)) (fun k => RawValues.value u "w" k.val)
          (RawValues.value u "mu" (u.pids 0)) (RawValues.value u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hy,hf⟩ := KernelOutputs.backward_complete (algebra R) N B u
  simp only [RawValues.raw,Nat.mul_zero,Nat.zero_add] at hy
  exact ⟨v,he,hm,fun j hj => congrArg (observe I (S "dx" (u.pids 0*N+j.val))) (hy j hj),hf⟩

theorem affine_graph_equiv_raw (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (rows N wc bc : Nat) (bias : Bool)
    (u : State Term) :
    ∃ v,
      FP.Structural.exec (engine (algebra R)) (KernelAffine.kernel rows N wc bc bias) u=some v ∧
      v.mem=KernelAffine.heap (algebra R) rows N wc bc (u.pids 0) bias u ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → observe I (S "dw" (u.pids 0*128+j.val)) (v.mem "dw" (u.pids 0*128+j.val))=
        observe I (S "dw" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (RawValues.affineValues (algebra R) rows N u (u.pids 0*128+j.val)).1))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → observe I (S "db" (u.pids 0*128+j.val)) (v.mem "db" (u.pids 0*128+j.val))=
        observe I (S "db" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (RawValues.affineValues (algebra R) rows N u (u.pids 0*128+j.val)).2))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hw,hb,hf⟩ := KernelOutputs.affine_complete (algebra R) rows N wc bc bias u
  exact ⟨v,he,hm,fun j hj => congrArg (observe I (S "dw" (u.pids 0*128+j.val))) (by simpa only [RawValues.carry] using hw j hj),
    fun h j hj => congrArg (observe I (S "db" (u.pids 0*128+j.val))) (by simpa only [RawValues.carry] using hb h j hj),hf⟩

/-- Public native specialization: the graph, primitive meanings and scalar
conversion are bound to the same selected object and original host epsilon. -/
theorem forward_kernel_equiv_raw (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (N B : Nat)
    (binding : Binding platform decoder N B .forward R S) (eps : ℝ) (bias : Bool)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .forward [N,N,N] hostEpsilon bias)
    (conversion : EpsilonConversion platform binding.selected.object P eps hostEpsilon)
    (u : State Term) :
    let I := P binding.selected.object.primitives
    let epsilon := epsilonWord platform binding.selected.object hostEpsilon
    (∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias u ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => RawValues.value u "x" (u.pids 0*N+k.val))
          (fun k => RawValues.value u "w" k.val) (fun k => RawValues.value u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => RawValues.value u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) :=
  forward_scalar_graph_equiv_raw R (P binding.selected.object.primitives) S N B eps
    (epsilonWord platform binding.selected.object hostEpsilon) bias conversion u

theorem backward_kernel_equiv_raw (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (N B : Nat)
    (binding : Binding platform decoder N B .backward R S)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .backward [N,N,N,N] hostEpsilon binding.selected.object.signature.hasBias)
    (u : State Term) :
    let I := P binding.selected.object.primitives
    ∃ v,
      FP.Structural.exec (engine (algebra R))
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u=some v ∧
      v.mem=KernelRows.backwardHeap (algebra R) N B u ∧
      (∀ j : Fin B,j.val<N → observe I (S "dx" (u.pids 0*N+j.val)) (v.mem "dx" (u.pids 0*N+j.val))=
        nativeGradientReference R I (S "dx" (u.pids 0*N+j.val)) N (fun k => RawValues.value u "x" (u.pids 0*N+k.val))
          (fun k => RawValues.value u "dy" (u.pids 0*N+k.val)) (fun k => RawValues.value u "w" k.val)
          (RawValues.value u "mu" (u.pids 0)) (RawValues.value u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) :=
  backward_graph_equiv_raw R (P binding.selected.object.primitives) S N B u

theorem affine_kernel_equiv_raw (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (rows N B wc bc : Nat)
    (binding : Binding platform decoder N B .affineBackward R S) (bias : Bool)
    (_abi : Compiler.Bound platform binding.selected .affineBackward [rows,N,wc,bc] (.boolean Bool.false) bias)
    (u : State Term) :
    let I := P binding.selected.object.primitives
    ∃ v,
      FP.Structural.exec (engine (algebra R)) (KernelAffine.kernel rows N wc bc bias) u=some v ∧
      v.mem=KernelAffine.heap (algebra R) rows N wc bc (u.pids 0) bias u ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → observe I (S "dw" (u.pids 0*128+j.val)) (v.mem "dw" (u.pids 0*128+j.val))=
        observe I (S "dw" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (RawValues.affineValues (algebra R) rows N u (u.pids 0*128+j.val)).1))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → observe I (S "db" (u.pids 0*128+j.val)) (v.mem "db" (u.pids 0*128+j.val))=
        observe I (S "db" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (RawValues.affineValues (algebra R) rows N u (u.pids 0*128+j.val)).2))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=u.mem q o) :=
  affine_graph_equiv_raw R (P binding.selected.object.primitives) S rows N wc bc bias u

#axiomsClean forward_kernel_equiv_raw
#axiomsClean backward_kernel_equiv_raw
#axiomsClean affine_kernel_equiv_raw
#axiomsClean forward_graph_equiv_raw
#axiomsClean forward_scalar_graph_equiv_raw
#axiomsClean backward_graph_equiv_raw
#axiomsClean affine_graph_equiv_raw
#specNonCircular nativeForwardReference avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular nativeGradientReference avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Equivalence

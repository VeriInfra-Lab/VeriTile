import bench.tritonbench_g.fast_ce_loss.Proofs.KernelOutputs
import bench.tritonbench_g.fast_ce_loss.Proofs.KernelMath
import bench.tritonbench_g.fast_ce_loss.Proofs.KernelCompiler

/-! Complete raw kernel interfaces. Tensor applications use the same
KernelExecution output certificates and independent mathematical recurrences. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.KernelInterface
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
namespace Real

theorem forward_kernel (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ)
    (positive : 0<p.vocab) (covered : p.vocab≤p.block)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col))
    (valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int)))
    (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.forward p) initial=some final ∧
      (final.mem p.lse (initial.pids 0)).read .real=some (KernelMath.fullLSE p x) ∧
      (final.mem p.loss (initial.pids 0)).read .real=some (KernelMath.rowLoss p x (label p initial)) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,lse,loss,frame⟩ := KernelExecution.forward_outputs
    KernelReal.algebra p initial (lt_of_lt_of_le positive covered) (valid.imp_right And.left) distinct
  have cast_id (a : WithBot ℝ) : KernelModel.cast KernelReal.algebra a=a := rfl
  rw [cast_id,KernelMath.lse_full p initial flags x positive covered inputs] at lse
  rw [cast_id,KernelMath.loss_full p initial flags x positive covered inputs valid] at loss
  exact ⟨final,run,lse,loss,frame⟩

theorem chunked_kernel (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ)
    (positive : 0<p.block) (nonempty : initial.pids 1*p.block<p.vocab)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col))
    (valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int)))
    (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.chunked p) initial=some final ∧
      (final.mem p.lse (initial.pids 0*p.chunks+initial.pids 1)).read .real=
        some (KernelMath.blockLSE p (initial.pids 1) x) ∧
      (initial.pids 1=0 → (final.mem p.loss (initial.pids 0)).read .real=
        some (KernelMath.partialLoss p x (label p initial))) ∧
      (∀ region offset, (initial.pids 1=0 → region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0*p.chunks+initial.pids 1) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,lse,loss,frame⟩ := KernelExecution.chunked_outputs
    KernelReal.algebra p initial positive (valid.imp_right And.left) distinct
  have cast_id (a : WithBot ℝ) : KernelModel.cast KernelReal.algebra a=a := rfl
  rw [cast_id,KernelMath.lse_block p initial flags x (initial.pids 1) positive nonempty inputs] at lse
  simp only [cast_id,KernelMath.loss_partial p initial flags x inputs valid] at loss
  exact ⟨final,run,lse,loss,frame⟩

theorem backward_kernel (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ) (lse upstream : ℝ)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col))
    (saved : (initial.mem p.lse (initial.pids 0)).read .real=some lse)
    (gradient : (initial.mem p.upstream (initial.pids 0*p.upstreamStride)).read .real=some upstream) :
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block,initial.pids 1*p.block+lane.val<p.vocab →
        (final.mem p.logits (initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val))).read .real=
          some (KernelReal.gradient p (label p initial) (initial.pids 1*p.block+lane.val)
            (x (initial.pids 1*p.block+lane.val)) lse upstream)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block,initial.pids 1*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,values,frame⟩ := KernelExecution.backward_outputs KernelReal.algebra p initial
  refine ⟨final,run,?_,frame⟩
  intro lane active
  exact (values lane active).trans (KernelReal.gradient_finite p initial flags lane
    (x (initial.pids 1*p.block+lane.val)) lse upstream active
    (inputs _ active) saved gradient)

#axiomsClean forward_kernel
#axiomsClean chunked_kernel
#axiomsClean backward_kernel
end Real
namespace Native
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native

theorem forward_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (initial : State Term) (positive : 0<p.block)
    (valid : label p initial = -100 ∨ 0≤label p initial) (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine (algebra R)) (KernelSource.forward p) initial=some final ∧
      KernelCompiler.observe I S ((final.mem p.lse (initial.pids 0)).read .real)=
        KernelCompiler.observe I S (KernelModel.cast (algebra R) (lseValue (algebra R) p initial 0)) ∧
      KernelCompiler.observe I S ((final.mem p.loss (initial.pids 0)).read .real)=
        KernelCompiler.observe I S (KernelModel.cast (algebra R) (forwardLoss (algebra R) p initial)) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,lse,loss,frame⟩ := KernelExecution.forward_outputs (algebra R) p initial positive valid distinct
  exact ⟨final,run,congrArg (KernelCompiler.observe I S) lse,congrArg (KernelCompiler.observe I S) loss,frame⟩

theorem chunked_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (initial : State Term) (positive : 0<p.block)
    (valid : label p initial = -100 ∨ 0≤label p initial) (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine (algebra R)) (KernelSource.chunked p) initial=some final ∧
      KernelCompiler.observe I S ((final.mem p.lse (initial.pids 0*p.chunks+initial.pids 1)).read .real)=
        KernelCompiler.observe I S
          (KernelModel.cast (algebra R) (lseValue (algebra R) p initial (initial.pids 1))) ∧
      (initial.pids 1=0 → KernelCompiler.observe I S ((final.mem p.loss (initial.pids 0)).read .real)=
        KernelCompiler.observe I S (KernelModel.cast (algebra R) (chunkLoss (algebra R) p initial))) ∧
      (∀ region offset, (initial.pids 1=0 → region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0*p.chunks+initial.pids 1) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,lse,loss,frame⟩ := KernelExecution.chunked_outputs (algebra R) p initial positive valid distinct
  exact ⟨final,run,congrArg (KernelCompiler.observe I S) lse,
    fun first => congrArg (KernelCompiler.observe I S) (loss first),frame⟩

theorem backward_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (initial : State Term) :
    ∃ final, exec (engine (algebra R)) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block,initial.pids 1*p.block+lane.val<p.vocab →
        KernelCompiler.observe I S
          ((final.mem p.logits (initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val))).read .real)=
        KernelCompiler.observe I S (gradientValue (algebra R) p initial lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block,initial.pids 1*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,values,frame⟩ := KernelExecution.backward_outputs (algebra R) p initial
  exact ⟨final,run,fun lane active => congrArg (KernelCompiler.observe I S) (values lane active),frame⟩

#axiomsClean forward_kernel
#axiomsClean chunked_kernel
#axiomsClean backward_kernel
end Native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelInterface

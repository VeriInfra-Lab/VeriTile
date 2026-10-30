import bench.tritonbench_g.cross_entropy2.Proofs.KernelReal
import bench.tritonbench_g.cross_entropy2.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.CrossEntropy2.RawCorrect
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

/-! Complete forward/backward mathematical kernel results over raw memory,
scalar ABI registers, logical input arrays, labels and kernel geometry. -/
theorem forward (p : Parameters) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block) (x : Nat → ℝ)
    (input : CrossEntropy1.RealSource.Input p.toParameters (scalarMemory initial) row x)
    (label : Int) (target : CrossEntropy1.Source.label p.toParameters (scalarMemory initial) row=label)
    (positive : 0<p.block) (start : block*p.block<p.cols) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=↑(KernelMath.lse p x block) ∧
      (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
        ↑(KernelMath.loss p x label block) ∧
      (final.mem p.zLoss (p.zLossBase+block*p.rows+row)).read .real=
        ↑(KernelMath.zLoss p x label block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row) →
        (region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+block*p.rows+row) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelNative.forward_outputs KernelReal.algebra p initial
    positive separate outputs (KernelReal.arguments p) abi
  simp only [rowPID,blockPID] at lseRead lossRead zRead frame
  have pair := KernelReal.loss_z_scalar p initial abi
  rw [KernelReal.target_scalar,rowPID,blockPID,target] at pair
  have mathPair := RealSource.scalar_outputs p (scalarMemory initial) row block x input
    label positive start
  have paired := pair.trans mathPair
  have lossEq := congrArg Prod.fst paired
  have zEq := congrArg Prod.snd paired
  rw [KernelReal.lse_scalar p initial abi,rowPID,blockPID] at lseRead
  refine ⟨final,execution,lseRead.trans (RealSource.block_lse p (scalarMemory initial) row block x input positive start),?_,?_,frame⟩
  · exact lossRead.trans (by simpa only [KernelMath.loss,KernelMath.lse,KernelMath.sum,Source.blockOwns] using lossEq)
  · exact zRead.trans (by simpa only [KernelMath.zLoss,KernelMath.lse] using zEq)

theorem backward (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block) (x : Nat → ℝ) (L upstream : ℝ)
    (input : CrossEntropy1.RealSource.Input p.toParameters (scalarMemory initial) row x)
    (label : Int) (target : CrossEntropy1.Source.label p.toParameters (scalarMemory initial) row=label)
    (lse : ((scalarMemory initial) p.lse (p.lseBase+row)).read .real=↑L)
    (up : ((scalarMemory initial) p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        (final.mem p.gradient (p.gradientBase+row*p.gradientStride+(block*p.block+lane.val))).read .real=
          ↑(KernelMath.gradient p x label L upstream (block*p.block+lane.val))) ∧
      (∀ region offset, (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        region≠p.gradient ∨ offset≠p.gradientBase+row*p.gradientStride+(block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs KernelReal.algebra p initial (KernelReal.arguments p) abi
  simp only [KernelNative.gradientOffset,rowPID,blockPID] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  have computed := reads lane active
  rw [KernelReal.gradient_scalar p initial abi lane (by simpa only [rowPID,blockPID] using active),rowPID,blockPID] at computed
  have mathematical := RealSource.backward_lane p (scalarMemory initial) row block x input
    label L upstream target lse up lane active
  exact computed.trans (by simpa only [KernelMath.gradient,Math.gradientWithLSE,CrossEntropy1.Source.column] using mathematical)

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RawCorrect

import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelReal
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.CrossEntropyOps.RawKernel
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

/-- Complete typed forward execution over raw memory, independent mathematical outputs,
and every untouched memory cell for an unsplit launch. -/
theorem forward_unsplit (p : Parameters)
    (lseLogits : p.lse≠p.logits) (distinctOutputs : p.lse≠p.loss)
    (zLse : p.zloss≠p.lse) (zLoss : p.zloss≠p.loss)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=0)
    (x : Nat → ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (label : Int) (labelRead : CrossEntropy1.Source.label p.toParameters (scalarMemory initial) row=label)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : label=p.ignored ∨
      (0≤label ∧ label<(p.cols : Int))) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+row)).read .real=↑(Spec.lse p.cols p.logitScale x) ∧
      (final.mem p.loss (p.lossBase+row)).read .real=
        ↑(Spec.loss p.cols p.logitScale p.smoothing p.squareScale x (label) p.ignored) ∧
      (final.mem p.zloss (p.zlossBase+row)).read .real=
        ↑(Spec.zLoss p.cols p.logitScale p.squareScale x (label) p.ignored) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+row) →
        (region≠p.lse ∨ offset≠p.lseBase+row) →
        (region≠p.zloss ∨ offset≠p.zlossBase+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial (by omega) lseLogits distinctOutputs zLse zLoss (KernelReal.arguments p) abi
  simp only [rowPID,blockPID,Nat.zero_mul,Nat.add_zero] at lseRead lossRead zRead frame
  have lse := RealSource.forward_lse p (scalarMemory initial) row 0 x input (by omega) (by simpa using positive)
  rw [CrossEntropy1.Math.block_zero_lse p.cols p.block covered] at lse
  have loss := RealSource.forward_unsplit_loss p (scalarMemory initial) row x input
    (label) positive covered splitFlag noSmooth startZero total valid
  rw [KernelReal.lse_scalar p initial abi,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID,labelRead] at lossRead
  refine ⟨final,execution,lseRead.trans lse,lossRead.trans loss,?_,?_⟩
  · have logging := zRead splitFlag
    rw [KernelReal.z_scalar p initial abi,rowPID,blockPID,labelRead] at logging
    rw [lse] at logging
    by_cases ignored : label=p.ignored
    all_goals simpa [ignored,Spec.zLoss,Spec.lse,CrossEntropy1.Math.lse,CrossEntropy1.Math.expSum,Spec.scaled,
      RealSource.algebra,CrossEntropy1.RealSource.algebra] using logging
  · intro region offset hl he hz
    exact frame region offset hl he (Or.inr hz)

/-- A split launch stores its own block LSE and local loss contribution.
No host aggregation is part of this local kernel claim. -/
theorem forward_split (p : Parameters)
    (lseLogits : p.lse≠p.logits) (distinctOutputs : p.lse≠p.loss)
    (zLse : p.zloss≠p.lse) (zLoss : p.zloss≠p.loss)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (label : Int) (labelRead : CrossEntropy1.Source.label p.toParameters (scalarMemory initial) row=label)
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=↑(CrossEntropy1.Math.blockLSE p.cols p.block block (Spec.scaled p.logitScale x)) ∧
      (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
        ↑(CrossEntropy1.RealSource.partialLoss p.toParameters (Spec.scaled p.logitScale x) (label) block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row) →
        (region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial positive lseLogits distinctOutputs zLse zLoss (KernelReal.arguments p) abi
  simp only [rowPID,blockPID] at lseRead lossRead zRead frame
  have lse := RealSource.forward_lse p (scalarMemory initial) row block x input positive start
  have loss := RealSource.forward_split_loss p (scalarMemory initial) row block x input
    (label) splitFlag noSmooth
  rw [KernelReal.lse_scalar p initial abi,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID,labelRead] at lossRead
  exact ⟨final,execution,lseRead.trans lse,lossRead.trans loss,fun r i hl he => frame r i hl he (Or.inl splitFlag)⟩

/-- Standalone backward specification is over the supplied raw LSE and
upstream. Its connection to a forward aggregation is outside this kernel. -/
def gradient (p : Parameters) (x : Nat → ℝ) (target : Int) (lse upstream : ℝ) (col : Nat) : ℝ :=
  if target=p.ignored then 0 else (upstream*p.logitScale)*
    ((1+2*p.squareScale*lse)*Real.exp (x col*p.logitScale-lse)-
      (if (col : Int)=target-p.classStart then 1-p.smoothing else 0)-p.smoothing/p.totalClasses)

theorem backward (p : Parameters)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p))
    (row block : Nat) (hr : row<p.rows) (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ) (lse upstream : ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (label : Int) (labelRead : CrossEntropy1.Source.label p.toParameters (scalarMemory initial) row=label)
    (lseRead : ((scalarMemory initial) p.lse (p.lseBase+row)).read .real=↑lse)
    (upstreamRead : ((scalarMemory initial) p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream)
    (lseLogits : p.lse≠p.logits) (lseUpstream : p.lse≠p.upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        (final.mem p.gradient (p.gradientBase+row*p.gradientStride+(block*p.block+lane.val))).read .real=
          ↑(gradient p x (label) lse upstream (block*p.block+lane.val))) ∧
      (∀ region offset, (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        region≠p.gradient ∨ offset≠p.gradientBase+row*p.gradientStride+(block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs KernelReal.algebra p initial (KernelReal.arguments p) abi
  simp only [KernelNative.gradientOffset,rowPID,blockPID] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  have computed := reads lane active
  rw [KernelReal.gradient_scalar p initial abi lseLogits lseUpstream,rowPID,blockPID] at computed
  exact computed.trans (RealSource.backward_lane p _ row block x input _ lse upstream
    labelRead lseRead upstreamRead lseLogits lseUpstream noSmooth lane active)

theorem gradient_from_lse (p : Parameters) (x : Nat → ℝ) (target : Int) (upstream : ℝ) (col : Nat)
    (start : p.classStart=0) (total : p.totalClasses=p.cols) :
    gradient p x target (Spec.lse p.cols p.logitScale x) upstream col=
      Spec.gradient p.cols p.logitScale p.smoothing p.squareScale upstream x target p.ignored col := by
  simp only [gradient,Spec.gradient,start,sub_zero,total]

#axiomsClean forward_unsplit
#axiomsClean forward_split
#axiomsClean backward
#specNonCircular gradient avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.RawKernel

import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.CrossEntropy1.RawCorrect
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

theorem forward_unsplit_kernel (p : Parameters) (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=0)
    (x : Nat → ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : Source.label p (scalarMemory initial) row=p.ignored ∨
      (0≤Source.label p (scalarMemory initial) row ∧ Source.label p (scalarMemory initial) row<(p.cols : Int))) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+row)).read .real=↑(Math.lse p.cols x) ∧
      (final.mem p.loss (p.lossBase+row)).read .real=
        ↑(Math.loss p.cols x (Source.label p (scalarMemory initial) row) p.ignored p.smoothing p.squareScale) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+row) →
        (region≠p.lse ∨ offset≠p.lseBase+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial (by omega) separate outputs (KernelReal.arguments p) abi
  simp only [rowPID,blockPID,Nat.zero_mul,Nat.add_zero] at lseRead lossRead frame
  have lse := RealSource.forward_lse p (scalarMemory initial) row 0 x input (by omega) (by simpa using positive)
  rw [Math.block_zero_lse p.cols p.block covered] at lse
  have loss := RealSource.unsplit_loss p (scalarMemory initial) row x input
    (Source.label p (scalarMemory initial) row) positive covered splitFlag noSmooth startZero total valid
  rw [KernelReal.lse_scalar,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID] at lossRead
  exact ⟨final,execution,lseRead.trans lse,lossRead.trans loss,frame⟩

theorem forward_split_kernel (p : Parameters) (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=↑(Math.blockLSE p.cols p.block block x) ∧
      (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
        ↑(RealSource.partialLoss p x (Source.label p (scalarMemory initial) row) block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row) →
        (region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial positive separate outputs (KernelReal.arguments p) abi
  simp only [rowPID,blockPID] at lseRead lossRead frame
  have lse := RealSource.forward_lse p (scalarMemory initial) row block x input positive start
  have loss := RealSource.split_loss p (scalarMemory initial) row block x input
    (Source.label p (scalarMemory initial) row) splitFlag noSmooth
  rw [KernelReal.lse_scalar,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID] at lossRead
  exact ⟨final,execution,lseRead.trans lse,lossRead.trans loss,frame⟩

/-- Complete typed backward source execution with the independent Real
gradient, supplied raw logit/LSE/upstream reads, and every exterior cell. -/
theorem backward_kernel (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ) (lse upstream : ℝ)
    (input : RealSource.Input p (scalarMemory initial) row x)
    (lseRead : ((scalarMemory initial) p.lse (p.lseBase+row)).read .real=↑lse)
    (upstreamRead : Source.label p (scalarMemory initial) row≠p.ignored →
      ((scalarMemory initial) p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, Source.column p block lane.val<p.cols →
        (final.mem p.gradient (Source.gradientAddress p row (Source.column p block lane.val))).read .real=
          ↑(KernelMath.gradient p x (Source.label p (scalarMemory initial) row) lse upstream
            (Source.column p block lane.val))) ∧
      (∀ region offset, (∀ lane : Fin p.block, Source.column p block lane.val<p.cols →
        region≠p.gradient ∨ offset≠Source.gradientAddress p row (Source.column p block lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs KernelReal.algebra p initial
    (KernelReal.arguments p) abi
  simp only [KernelNative.gradientOffset,rowPID,blockPID] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  have computed := reads lane active
  rw [KernelReal.gradient_scalar p initial abi,rowPID,blockPID] at computed
  apply computed.trans
  by_cases ignored : Source.label p (scalarMemory initial) row=p.ignored
  · simp [Source.backwardValues,ignored,KernelMath.gradient,RealSource.fp_coe]
  · exact RealSource.backward_lane p _ row block x input _ lse upstream rfl lseRead
      (upstreamRead ignored) noSmooth lane active

#axiomsClean forward_unsplit_kernel
#axiomsClean forward_split_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RawCorrect

import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelCorrect
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual typed forward execution, both independent mathematical outputs,
and every untouched memory cell for an unsplit launch. -/
theorem forward_unsplit (p : Parameters) (views : ForwardViews p)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=0)
    (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : targetValue initial views.labels row=p.ignored ∨
      (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.cols : Int))) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+row)).read .real=↑(Math.lse p.cols x) ∧
      (final.mem p.loss (p.lossBase+row)).read .real=
        ↑(Math.loss p.cols x (targetValue initial views.labels row) p.ignored p.smoothing p.squareScale) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+row) →
        (region≠p.lse ∨ offset≠p.lseBase+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial (by omega) views.lseLogits views.distinctOutputs (KernelReal.arguments p) abi
  simp only [rowPID,blockPID,Nat.zero_mul,Nat.add_zero] at lseRead lossRead frame
  have input := KernelMath.row_input p initial views.logits views.logitsView row hr x values
  have target := label_binding p initial views.labels views.labelsView row hr
  have lse := RealSource.forward_lse p (scalarMemory initial) row 0 x input (by omega) (by simpa using positive)
  rw [Math.block_zero_lse p.cols p.block covered] at lse
  have loss := RealSource.unsplit_loss p (scalarMemory initial) row x input
    (targetValue initial views.labels row) positive covered splitFlag noSmooth startZero total valid
  rw [KernelReal.lse_scalar,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID,target] at lossRead
  exact ⟨final,execution,lseRead.trans lse,lossRead.trans loss,frame⟩

/-- A split launch stores its own block LSE and local loss contribution.
No host aggregation is part of this local kernel claim. -/
theorem forward_split (p : Parameters) (views : ForwardViews p)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=↑(Math.blockLSE p.cols p.block block x) ∧
      (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
        ↑(RealSource.partialLoss p x (targetValue initial views.labels row) block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row) →
        (region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row) → final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    KernelReal.algebra p initial positive views.lseLogits views.distinctOutputs (KernelReal.arguments p) abi
  simp only [rowPID,blockPID] at lseRead lossRead frame
  have input := KernelMath.row_input p initial views.logits views.logitsView row hr x values
  have target := label_binding p initial views.labels views.labelsView row hr
  have lse := RealSource.forward_lse p (scalarMemory initial) row block x input positive start
  have loss := RealSource.split_loss p (scalarMemory initial) row block x input
    (targetValue initial views.labels row) splitFlag noSmooth
  rw [KernelReal.lse_scalar,rowPID,blockPID] at lseRead
  rw [KernelReal.loss_scalar p initial abi,rowPID,blockPID,target] at lossRead
  exact ⟨final,execution,lseRead.trans lse,lossRead.trans loss,frame⟩

/-- Actual backward execution computes each independent gradient value and
frames all other storage, including the allowed in-place logit allocation. -/
theorem backward (p : Parameters) (views : BackwardViews p)
    (initial : State (WithBot ℝ)) (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block)
    (x : Nat → ℝ) (lse upstream : ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (lseValue : fpValue initial views.lse row=↑lse)
    (upstreamValue : targetValue initial views.labels row≠p.ignored → fpValue initial views.upstream row=↑upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, Source.column p block lane.val<p.cols →
        (final.mem p.gradient (Source.gradientAddress p row (Source.column p block lane.val))).read .real=
          ↑(KernelMath.gradient p x (targetValue initial views.labels row) lse upstream (Source.column p block lane.val))) ∧
      (∀ region offset, (∀ lane : Fin p.block, Source.column p block lane.val<p.cols →
        region≠p.gradient ∨ offset≠Source.gradientAddress p row (Source.column p block lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs KernelReal.algebra p initial (KernelReal.arguments p) abi
  simp only [KernelNative.gradientOffset,rowPID,blockPID] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  have computed := reads lane active
  rw [KernelReal.gradient_scalar p initial abi,rowPID,blockPID] at computed
  have mathematical := (KernelMath.scalar_backward p views initial row block hr x lse upstream
    values lseValue upstreamValue noSmooth).1 lane active
  rw [Program.backward_output RealSource.algebra p (scalarMemory initial) row block lane active] at mathematical
  exact computed.trans mathematical

#axiomsClean forward_unsplit
#axiomsClean forward_split
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelCorrect

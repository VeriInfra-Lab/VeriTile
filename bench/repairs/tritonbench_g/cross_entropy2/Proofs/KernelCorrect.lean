import bench.repairs.tritonbench_g.cross_entropy2.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelCorrect
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

theorem forward (p : Parameters) (views : ForwardViews p) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block) (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (positive : 0<p.block) (start : block*p.block<p.cols) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=↑(KernelMath.lse p x block) ∧
      (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
        ↑(KernelMath.loss p x (targetValue initial views.labels row) block) ∧
      (final.mem p.zLoss (p.zLossBase+block*p.rows+row)).read .real=
        ↑(KernelMath.zLoss p x (targetValue initial views.labels row) block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+block*p.rows+row) →
        (region≠p.lse ∨ offset≠p.lseBase+block*p.rows+row) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+block*p.rows+row) →
        final.mem region offset=initial.mem region offset) := by
  have input := CrossEntropy1.KernelMath.row_input p.toParameters initial views.logits views.logitsView row hr x values
  have target := label_binding p.toParameters initial views.labels views.labelsView row hr
  exact RawCorrect.forward p views.lseLogits ⟨views.distinctOutputs,views.zDistinct⟩ initial abi row block hr
    rowPID blockPID x input (targetValue initial views.labels row) target positive start

theorem backward (p : Parameters) (views : BackwardViews p) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (KernelReal.arguments p)) (row block : Nat) (hr : row<p.rows)
    (rowPID : initial.pids 0=row) (blockPID : initial.pids 1=block) (x : Nat → ℝ) (L upstream : ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (lseValue : fpValue initial views.lse row=↑L) (upstreamValue : fpValue initial views.upstream row=↑upstream) :
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        (final.mem p.gradient (p.gradientBase+row*p.gradientStride+(block*p.block+lane.val))).read .real=
          ↑(KernelMath.gradient p x (targetValue initial views.labels row) L upstream (block*p.block+lane.val))) ∧
      (∀ region offset, (∀ lane : Fin p.block, block*p.block+lane.val<p.cols →
        region≠p.gradient ∨ offset≠p.gradientBase+row*p.gradientStride+(block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  have input := CrossEntropy1.KernelMath.row_input p.toParameters initial views.logits views.logitsView row hr x values
  have target := label_binding p.toParameters initial views.labels views.labelsView row hr
  have lse : ((scalarMemory initial) p.lse (p.lseBase+row)).read .real=↑L := by
    simpa only [fpValue,vector_index _ _ _ _ _ _ views.lseView hr,views.lseView.regionEq,Nat.mul_one,
      scalarMemory,CrossEntropy1.Source.Cell.read] using lseValue
  have up : ((scalarMemory initial) p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream := by
    simpa only [fpValue,vector_index _ _ _ _ _ _ views.upstreamView hr,views.upstreamView.regionEq,
      scalarMemory,CrossEntropy1.Source.Cell.read] using upstreamValue
  exact RawCorrect.backward p initial abi row block hr rowPID blockPID x L upstream input
    (targetValue initial views.labels row) target lse up

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelCorrect

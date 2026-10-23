import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelTensor
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelActual
open VeriTile.Triton FP.Structural KernelContract KernelTensor
noncomputable section
set_option maxHeartbeats 4000000

theorem real_forward (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : ForwardInputs p views alloc bytes words)
    (_scalarProjection : ScalarProjection p interpret (forwardScalars inputs))
    (actual : KernelBinding.Forward p views alloc)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret (Region.cast p.labels) words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
    (∀ col, col<p.cols → x col=interpret (KernelBinding.read words p.logits actual.logits (launch.row*p.cols+col))) ∧
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      KernelBinding.read final p.lse actual.lse (launch.block*p.rows+launch.row)=↑(CrossEntropy1.Math.blockLSE p.cols p.block launch.block (Spec.scaled p.logitScale x)) ∧
      KernelBinding.read final p.loss actual.loss (launch.block*p.rows+launch.row)=
        ↑(forwardLoss p x (targetValue words views.labels launch.row) launch.block) ∧
      (p.split=Bool.false → KernelBinding.read final p.zloss actual.zloss (launch.block*p.rows+launch.row)=
        ↑(Spec.zLoss p.cols p.logitScale p.squareScale x (targetValue words views.labels launch.row) p.ignored)) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        (p.split=Bool.true ∨ region≠p.zloss ∨ offset≠p.zlossBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  constructor
  · intro col hc
    exact congrArg interpret (KernelBinding.matrix_read alloc views.logits actual.logits actual.logitsBinding words
      _ _ _ _ _ _ _ views.logitsView launch.rowBound hc).symm
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelTensor.forward interpret p views alloc bytes words inputs _scalarProjection launch pids
  refine ⟨final,execution,?_,?_,?_,frame⟩
  · exact (KernelBinding.scalar_read alloc views.lse actual.lse actual.lseBinding final _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound).trans lseRead
  · exact (KernelBinding.scalar_read alloc views.loss actual.loss actual.lossBinding final _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound).trans lossRead
  · intro unsplit
    exact (KernelBinding.scalar_read alloc views.zloss actual.zloss actual.zlossBinding final _ _ _ _ _ _ views.zlossView launch.rowBound launch.blockBound).trans (zRead unsplit)

theorem real_backward (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : BackwardInputs p views alloc bytes words)
    (_scalarProjection : ScalarProjection p interpret (backwardScalars inputs))
    (actual : KernelBinding.Backward p views alloc)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret (Region.cast p.labels) words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
    (∀ col, col<p.cols → x col=interpret (KernelBinding.read words p.logits actual.logits (launch.row*p.cols+col))) ∧
    logicalReal interpret words views.lse launch.row=interpret (KernelBinding.read words p.lse actual.lse launch.row) ∧
    logicalReal interpret words views.upstream launch.row=interpret (KernelBinding.read words p.upstream actual.upstream launch.row) ∧
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        KernelBinding.read final p.gradient actual.gradient (launch.row*p.cols+(launch.block*p.block+lane.val))=
          ↑(KernelCorrect.gradient p x (targetValue words views.labels launch.row)
            (logicalReal interpret words views.lse launch.row) (logicalReal interpret words views.upstream launch.row)
            (launch.block*p.block+lane.val))) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  constructor
  · intro col hc
    exact congrArg interpret (KernelBinding.matrix_read alloc views.logits actual.logits actual.logitsBinding words
      _ _ _ _ _ _ _ views.logitsView launch.rowBound hc).symm
  refine ⟨?_,?_,?_⟩
  · exact congrArg interpret (KernelBinding.vector_read alloc views.lse actual.lse actual.lseBinding words
      _ _ _ _ _ views.lseView launch.rowBound).symm
  · exact congrArg interpret (KernelBinding.vector_read alloc views.upstream actual.upstream actual.upstreamBinding words
      _ _ _ _ _ views.upstreamView launch.rowBound).symm
  obtain ⟨final,execution,reads,frame⟩ := KernelTensor.backward interpret p views alloc bytes words inputs _scalarProjection launch pids
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  exact (KernelBinding.matrix_read alloc views.gradient actual.gradient actual.gradientBinding final
    _ _ _ _ _ _ _ views.gradientView launch.rowBound active).trans (reads lane active)

#axiomsClean real_forward
#axiomsClean real_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelActual

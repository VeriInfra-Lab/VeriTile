import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelProfile
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelActual
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

open KernelProfile

theorem native_forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (actual : KernelBinding.Forward p views alloc)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := KernelProfile.entry p (forwardScalars inputs) words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S (KernelBinding.read final p.lse actual.lse (launch.block*p.rows+launch.row))=
        Compiler.eval I S (KernelProfile.forwardLSE R p (forwardScalars inputs) words) ∧
      Compiler.eval I S (KernelBinding.read final p.loss actual.loss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (KernelProfile.forwardLoss R p (forwardScalars inputs) words) ∧
      (p.split=Bool.false → Compiler.eval I S (KernelBinding.read final p.zloss actual.zloss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (KernelProfile.forwardZ R p (forwardScalars inputs) words)) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        (p.split=Bool.true ∨ region≠p.zloss ∨ offset≠p.zlossBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelProfile.forward R I S p views alloc bytes words inputs launch pids
  refine ⟨final,execution,?_,?_,?_,frame⟩
  · exact (congrArg (Compiler.eval I S) (KernelBinding.scalar_read alloc views.lse actual.lse actual.lseBinding final _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound)).trans lseRead
  · exact (congrArg (Compiler.eval I S) (KernelBinding.scalar_read alloc views.loss actual.loss actual.lossBinding final _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound)).trans lossRead
  · intro unsplit
    exact (congrArg (Compiler.eval I S) (KernelBinding.scalar_read alloc views.zloss actual.zloss actual.zlossBinding final _ _ _ _ _ _ views.zlossView launch.rowBound launch.blockBound)).trans (zRead unsplit)

theorem native_backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (actual : KernelBinding.Backward p views alloc)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := KernelProfile.entry p (backwardScalars inputs) words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        Compiler.eval I S (KernelBinding.read final p.gradient actual.gradient (launch.row*p.cols+(launch.block*p.block+lane.val)))=
          Compiler.eval I S (KernelProfile.backwardLane R p (backwardScalars inputs) words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelProfile.backward R I S p views alloc bytes words inputs launch pids
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  exact (congrArg (Compiler.eval I S) (KernelBinding.matrix_read alloc views.gradient actual.gradient actual.gradientBinding final
    _ _ _ _ _ _ _ views.gradientView launch.rowBound active)).trans (reads lane active)

#axiomsClean native_forward
#axiomsClean native_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelActual

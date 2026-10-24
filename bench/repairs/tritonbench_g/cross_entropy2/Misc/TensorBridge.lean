import bench.repairs.tritonbench_g.cross_entropy2.Correct
import bench.repairs.tritonbench_g.cross_entropy2.Misc.NativeCorrect
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelTensor
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.CrossEntropy2.TensorBridge
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
section Real
open KernelTensor

theorem forward_real (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : ForwardInputs p views alloc bytes words)
    (_scalars : ScalarProjection p interpret inputs.scalars) (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      fpValue final views.lse (launch.block*p.rows+launch.row)=↑(KernelMath.lse p x launch.block) ∧
      fpValue final views.loss (launch.block*p.rows+launch.row)=
        ↑(KernelMath.loss p x (targetValue words views.labels launch.row) launch.block) ∧
      fpValue final views.zLoss (launch.block*p.rows+launch.row)=
        ↑(KernelMath.zLoss p x (targetValue words views.labels launch.row) launch.block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) :=
  KernelTensor.forward interpret p views alloc bytes words inputs _scalars launch pids

theorem backward_real (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : BackwardInputs p views alloc bytes words)
    (_scalars : ScalarProjection p interpret inputs.scalars) (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        fpValue final views.gradient (launch.row*p.cols+(launch.block*p.block+lane.val))=
          ↑(KernelMath.gradient p x (targetValue words views.labels launch.row)
            (logicalReal interpret words views.lse launch.row) (logicalReal interpret words views.upstream launch.row)
            (launch.block*p.block+lane.val))) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) :=
  KernelTensor.backward interpret p views alloc bytes words inputs _scalars launch pids

end Real
section Native
open KernelProfile

theorem forward_native (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := entry p inputs.scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S (fpValue final views.lse (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardLSE R p inputs.scalars words) ∧
      Compiler.eval I S (fpValue final views.loss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardLoss R p inputs.scalars words) ∧
      Compiler.eval I S (fpValue final views.zLoss (launch.block*p.rows+launch.row))=
        Compiler.eval I S (forwardZ R p inputs.scalars words) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) :=
  KernelProfile.forward R I S p views alloc bytes words inputs launch pids

theorem backward_native (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := entry p inputs.scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
        Compiler.eval I S (fpValue final views.gradient (launch.row*p.cols+(launch.block*p.block+lane.val)))=
          Compiler.eval I S (backwardLane R p inputs.scalars words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+launch.row*p.gradientStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) :=
  KernelProfile.backward R I S p views alloc bytes words inputs launch pids

end Native
#axiomsClean forward_real
#axiomsClean backward_real
#axiomsClean forward_native
#axiomsClean backward_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.TensorBridge

import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelCorrect
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelAccess
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelTensor
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000

/-- Three input-only scalar equations identify the mathematical parameters
with the very same runtime FP32 words used by the native theorem. -/
structure ScalarProjection (p : Parameters) (interpret : Word → ℝ) (scalars : RuntimeArgs.Values Word) : Prop where
  smoothing : interpret scalars.smoothing=p.smoothing
  squareScale : interpret scalars.squareScale=p.squareScale
  logitScale : interpret scalars.logitScale=p.logitScale

def realState (p : Parameters) (interpret : Word → ℝ) (words : State Word) : State (WithBot ℝ) :=
  let s := CrossEntropy1.KernelTensor.realState p.toParameters interpret (Region.cast p.labels) words
  s.setReg "logit_scale" .real [] (fun _ => ↑p.logitScale)

theorem real_binding (p : Parameters) (interpret : Word → ℝ) (words : State Word) :
    RuntimeArgs.Bound p (realState p interpret words) (KernelReal.arguments p) := by
  constructor
  · exact {
      smoothing := by simp [realState,CrossEntropy1.KernelTensor.realState,State.setReg,KernelReal.arguments]
      squareScale := by simp [realState,CrossEntropy1.KernelTensor.realState,State.setReg,KernelReal.arguments]
      totalClasses := by simp [realState,CrossEntropy1.KernelTensor.realState,State.setReg] }
  · simp [realState,State.setReg,KernelReal.arguments]

abbrev logicalReal := CrossEntropy1.KernelTensor.logicalReal

theorem float_input (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (tensor : CrossEntropy1.TensorModel.Tensor) (separate : tensor.region≠Region.cast p.labels) (i : Nat) :
    fpValue (realState p interpret words) tensor i=↑(logicalReal interpret words tensor i) := by
  exact CrossEntropy1.KernelTensor.float_input p.toParameters interpret words _ tensor separate i

theorem label_input (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (tensor : CrossEntropy1.TensorModel.Tensor) (same : tensor.region=Region.cast p.labels) (i : Nat) :
    targetValue (realState p interpret words) tensor i=targetValue words tensor i := by
  exact CrossEntropy1.KernelTensor.label_input p.toParameters interpret words _ tensor same i

theorem forward (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p)
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
        final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
  have values : ∀ col, col<p.cols → fpValue initial views.logits (launch.row*p.cols+col)=↑(x col) := by
    intro col _
    exact float_input p interpret words views.logits (by simpa only [views.logitsView.regionEq] using inputs.typedViews) _
  have target : targetValue initial views.labels launch.row=targetValue words views.labels launch.row :=
    label_input p interpret words views.labels views.labelsView.regionEq launch.row
  have start := CrossEntropy1.Math.block_starts p.cols p.block launch.block inputs.parameters.geometry.widthPositive launch.blockBound
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelCorrect.forward p views initial
    (real_binding p interpret words) launch.row launch.block launch.rowBound pids.1 pids.2 x values
    inputs.parameters.geometry.widthPositive start
  refine ⟨final,execution,?_,?_,?_,frame⟩
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,
      views.lseView.regionEq] using lseRead
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,
      views.lossView.regionEq,target] using lossRead
  · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.zLossView launch.rowBound launch.blockBound,
      views.zLossView.regionEq,target] using zRead

theorem backward (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
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
        final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
  have values : ∀ col, col<p.cols → fpValue initial views.logits (launch.row*p.cols+col)=↑(x col) := by
    intro col _
    exact float_input p interpret words views.logits (by simpa only [views.logitsView.regionEq] using inputs.typedViews.1) _
  have lseRead := float_input p interpret words views.lse (by simpa only [views.lseView.regionEq] using inputs.typedViews.2.1) launch.row
  have upstreamRead := float_input p interpret words views.upstream (by simpa only [views.upstreamView.regionEq] using inputs.typedViews.2.2) launch.row
  have target : targetValue initial views.labels launch.row=targetValue words views.labels launch.row :=
    label_input p interpret words views.labels views.labelsView.regionEq launch.row
  obtain ⟨final,execution,reads,frame⟩ := KernelCorrect.backward p views initial (real_binding p interpret words)
    launch.row launch.block launch.rowBound pids.1 pids.2 x (logicalReal interpret words views.lse launch.row)
    (logicalReal interpret words views.upstream launch.row) values lseRead upstreamRead
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  simpa only [fpValue,matrix_index _ _ _ _ _ _ _ _ views.gradientView launch.rowBound active,
    views.gradientView.regionEq,target] using reads lane active

#axiomsClean forward
#axiomsClean backward
#specNonCircular ScalarProjection avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelTensor

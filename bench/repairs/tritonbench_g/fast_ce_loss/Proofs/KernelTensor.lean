import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelOutputs
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelMath
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelTensor
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelContract KernelContract.Storage
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Input-only idealization of the actual FP32 words. It adds no finite-tensor
guard and makes no claim that native rounding is exact real arithmetic. -/
def realState (p : Parameters) (interpret : Word → ℝ) (words : State Word) : State (WithBot ℝ) where
  mem := fun r i => if r=Region.cast p.labels then .mk .int ((words.mem r i).read .int)
    else .mk .real (some (interpret ((words.mem r i).read .real)))
  regs := fun _ _ _ => none
  pids := words.pids
  numPids := words.numPids
  undef := fun d _ _ => defaultValue d

def logicalReal (interpret : Word → ℝ) (words : State Word) (tensor : Tensor) (i : Nat) : ℝ :=
  interpret (tensorValue words tensor i)

 theorem float_input (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (tensor : Tensor) (separate : tensor.region≠Region.cast p.labels) (i : Nat) :
    tensorValue (realState p interpret words) tensor i=some (logicalReal interpret words tensor i) := by
  simp [tensorValue,realState,separate,logicalReal]

 theorem label_input (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (tensor : Tensor) (same : tensor.region=Region.cast p.labels) (i : Nat) :
    tensorLabel (realState p interpret words) tensor i=tensorLabel words tensor i := by
  simp [tensorLabel,realState,same]

 theorem row_input (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (t : Tensor) (view : MatrixView t p.logits p.rows p.vocab p.logitsStride)
    (typed : p.logits≠Region.cast p.labels) (row : Nat) (hr : row<p.rows) (pid : words.pids 0=row) :
    ∀ col<p.vocab,
      ((realState p interpret words).mem p.logits (words.pids 0*p.logitsStride+col)).read .real=
        some (logicalReal interpret words t (row*p.vocab+col)) := by
  intro col hc
  have h := float_input p interpret words t (by simpa only [view.regionEq] using typed) (row*p.vocab+col)
  simpa only [tensorValue,matrix_index _ _ _ _ _ _ _ view hr hc,view.regionEq,
    Nat.add_assoc,Nat.add_sub_cancel_left,pid] using h

 theorem row_label (p : Parameters) (interpret : Word → ℝ) (words : State Word)
    (t : Tensor) (view : FlatView t (Region.cast p.labels) p.rows)
    (row : Nat) (hr : row<p.rows) (pid : words.pids 0=row) :
    label p (realState p interpret words)=tensorLabel words t row := by
  simp [label,realState,tensorLabel,view.index row hr,view.regionEq,pid]

 theorem forward (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p 1)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (covered : p.vocab≤p.block)
    (launch : Launch p 1) (pids : LaunchState launch words) :
    let initial := realState p interpret words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.forward p) initial=some final ∧
      tensorValue final views.lse launch.row=some (KernelMath.fullLSE p x) ∧
      tensorValue final views.loss launch.row=some (KernelMath.rowLoss p x (tensorLabel words views.labels launch.row)) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row) → final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.vocab+col)
  have flags : KernelReal.Flags p := ⟨inputs.options.softcapFlag,inputs.options.scalingFlag⟩
  have values := row_input p interpret words views.logits views.logitsView inputs.labelSeparation.1 launch.row launch.rowBound pids.1
  have target := row_label p interpret words views.labels views.labelsView launch.row launch.rowBound pids.1
  change label p initial=tensorLabel words views.labels launch.row at target
  have valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int)) := by
    simpa only [target] using inputs.labelsValid launch.row launch.rowBound
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelExecution.forward_outputs
    KernelReal.algebra p initial inputs.parameters.blockPositive (valid.imp_right And.left) views.outputsDistinct
  have lse := KernelMath.lse_full p initial flags x inputs.parameters.vocabPositive covered values
  have loss := KernelMath.loss_full p initial flags x inputs.parameters.vocabPositive covered values valid
  have cast_id (a : WithBot ℝ) : KernelModel.cast KernelReal.algebra a=a := rfl
  simp only [cast_id] at lseRead lossRead
  rw [lse] at lseRead
  rw [loss,target] at lossRead
  simp only [initial,realState,pids.1] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · simpa only [tensorValue,views.lseView.index launch.row (by simpa using launch.rowBound),
      views.lseView.regionEq,Nat.add_sub_cancel_left] using lseRead
  · simpa only [tensorValue,views.lossView.index launch.row launch.rowBound,
      views.lossView.regionEq,Nat.add_sub_cancel_left] using lossRead

 theorem chunked (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p p.chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (chunks : p.chunks=KernelMath.CE.blocks p.vocab p.block)
    (launch : Launch p p.chunks) (pids : LaunchState launch words) :
    let initial := realState p interpret words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.chunked p) initial=some final ∧
      tensorValue final views.lse (launch.row*p.chunks+launch.block)=some (KernelMath.blockLSE p launch.block x) ∧
      (launch.block=0 → tensorValue final views.loss launch.row=
        some (KernelMath.partialLoss p x (tensorLabel words views.labels launch.row))) ∧
      (∀ region offset, (launch.block=0 → region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row*p.chunks+launch.block) →
        final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.vocab+col)
  have flags : KernelReal.Flags p := ⟨inputs.options.softcapFlag,inputs.options.scalingFlag⟩
  have values := row_input p interpret words views.logits views.logitsView inputs.labelSeparation.1 launch.row launch.rowBound pids.1
  have target := row_label p interpret words views.labels views.labelsView launch.row launch.rowBound pids.1
  change label p initial=tensorLabel words views.labels launch.row at target
  have valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int)) := by
    simpa only [target] using inputs.labelsValid launch.row launch.rowBound
  have start : launch.block*p.block<p.vocab :=
    KernelMath.chunk_start p inputs.parameters.blockPositive launch.block (by simpa only [← chunks] using launch.blockBound)
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelExecution.chunked_outputs
    KernelReal.algebra p initial inputs.parameters.blockPositive (valid.imp_right And.left) views.outputsDistinct
  have lse := KernelMath.lse_block p initial flags x launch.block inputs.parameters.blockPositive start values
  have loss := KernelMath.loss_partial p initial flags x values valid
  have cast_id (a : WithBot ℝ) : KernelModel.cast KernelReal.algebra a=a := rfl
  simp only [cast_id] at lseRead lossRead
  have blockPid : initial.pids 1=launch.block := pids.2
  rw [blockPid,lse] at lseRead
  simp only [loss,target] at lossRead
  simp only [initial,realState,pids.1,pids.2] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · have hi : launch.row*p.chunks+launch.block<p.rows*p.chunks := by
      have bound := Nat.mul_le_mul_right p.chunks (Nat.succ_le_of_lt launch.rowBound)
      have hb := launch.blockBound
      simp only [Nat.succ_mul] at bound
      omega
    simpa only [tensorValue,views.lseView.index _ hi,views.lseView.regionEq,Nat.add_sub_cancel_left] using lseRead
  · intro first
    simpa only [tensorValue,views.lossView.index launch.row launch.rowBound,
      views.lossView.regionEq,Nat.add_sub_cancel_left] using lossRead first

 theorem backward (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (launch : Launch p blocks) (pids : LaunchState launch words) :
    let initial := realState p interpret words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.vocab+col)
    ∃ final, exec (engine KernelReal.algebra) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
        tensorValue final views.logits (launch.row*p.vocab+(launch.block*p.block+lane.val))=
          some (KernelReal.gradient p (tensorLabel words views.labels launch.row)
            (launch.block*p.block+lane.val) (x (launch.block*p.block+lane.val))
            (logicalReal interpret words views.lse launch.row) (logicalReal interpret words views.upstream launch.row))) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠launch.row*p.logitsStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret words
  have flags : KernelReal.Flags p := ⟨inputs.options.softcapFlag,inputs.options.scalingFlag⟩
  have values := row_input p interpret words views.logits views.logitsView inputs.typedSeparation.1 launch.row launch.rowBound pids.1
  have target := row_label p interpret words views.labels views.labelsView launch.row launch.rowBound pids.1
  change label p initial=tensorLabel words views.labels launch.row at target
  have lse := float_input p interpret words views.lse (by simpa only [views.lseView.regionEq] using inputs.typedSeparation.2.1) launch.row
  have upstream := float_input p interpret words views.upstream (by simpa only [views.upstreamView.regionEq] using inputs.typedSeparation.2.2) launch.row
  simp only [tensorValue,views.lseView.index launch.row launch.rowBound,views.lseView.regionEq,Nat.add_sub_cancel_left] at lse
  simp only [tensorValue,vector_index _ _ _ _ _ views.upstreamView launch.rowBound,
    views.upstreamView.regionEq,Nat.add_sub_cancel_left] at upstream
  obtain ⟨final,execution,reads,frame⟩ := KernelExecution.backward_outputs KernelReal.algebra p initial
  simp only [KernelExecution.gradientOffset,initial,realState,pids.1,pids.2] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  have result := KernelReal.gradient_finite p initial flags lane
    (logicalReal interpret words views.logits (launch.row*p.vocab+(launch.block*p.block+lane.val)))
    (logicalReal interpret words views.lse launch.row) (logicalReal interpret words views.upstream launch.row)
    (by simpa only [initial,realState,pids.2] using active)
    (by simpa only [initial,realState,pids.2] using values _ active)
    (by simpa only [initial,realState,pids.1] using lse)
    (by simpa only [initial,realState,pids.1] using upstream)
  rw [target] at result
  have output := (reads lane active).trans result
  simpa only [tensorValue,matrix_index _ _ _ _ _ _ _ views.logitsView launch.rowBound active,
    views.logitsView.regionEq,Nat.add_assoc,Nat.add_sub_cancel_left,target,initial,realState,pids.2] using output

#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
#specNonCircular logicalReal avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelTensor

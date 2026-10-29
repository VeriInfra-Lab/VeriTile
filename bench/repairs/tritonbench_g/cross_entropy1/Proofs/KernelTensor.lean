import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelCorrect
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelTensor
open VeriTile.Triton FP.Structural KernelContract
open TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

/-- Input-only idealization of the same bound runtime FP32 scalar words.
This states no arithmetic, output or successful-execution equality. -/
structure ScalarProjection (p : Parameters) (interpret : Word → ℝ)
    (scalars : RuntimeArgs.Values Word) : Prop where
  smoothing : interpret scalars.smoothing=p.smoothing
  squareScale : interpret scalars.squareScale=p.squareScale

/-- Construct the ideal mathematical input from the actual FP32 words and
signed-int64 labels. Interpretation is an ordinary input abstraction; no
finite-value guard or hardware decoder claim is added. Logical typed regions
may share coherent physical backing. -/
def realState (p : Parameters) (interpret : Word → ℝ) (labels : RegionName) (words : State Word) : State (WithBot ℝ) :=
  let s : State (WithBot ℝ) := {
    mem := fun r i => if r=labels then .mk .int ((words.mem r i).read .int)
      else .mk .real (↑(interpret ((words.mem r i).read .real)))
    regs := fun _ _ _ => none
    pids := words.pids
    numPids := words.numPids
    undef := fun d _ _ => defaultValue d }
  let s := s.setReg "smoothing" .real [] (fun _ => ↑p.smoothing)
  let s := s.setReg "lse_square_scale" .real [] (fun _ => ↑p.squareScale)
  s.setReg "total_classes" .int [] (fun _ => (p.totalClasses : Int))

theorem real_binding (p : Parameters) (interpret : Word → ℝ) (labels : RegionName) (words : State Word) :
    RuntimeArgs.Bound p (realState p interpret labels words) (KernelReal.arguments p) := by
  constructor <;> simp [realState,State.setReg,KernelReal.arguments]

def logicalReal (interpret : Word → ℝ) (words : State Word) (tensor : Tensor) (i : Nat) : ℝ :=
  interpret (fpValue words tensor i)

theorem float_input (p : Parameters) (interpret : Word → ℝ) (words : State Word) (labels : RegionName)
    (tensor : Tensor) (separate : tensor.region≠labels) (i : Nat) :
    fpValue (realState p interpret labels words) tensor i=↑(logicalReal interpret words tensor i) := by
  simp [fpValue,realState,State.setReg,separate,logicalReal]

theorem label_input (p : Parameters) (interpret : Word → ℝ) (words : State Word) (labels : RegionName)
    (tensor : Tensor) (same : tensor.region=labels) (i : Nat) :
    targetValue (realState p interpret labels words) tensor i=targetValue words tensor i := by
  simp [targetValue,realState,State.setReg,same]

theorem no_smoothing (p : Parameters) (valid : ParametersValid p) :
    p.hasSmoothing=Bool.false → p.smoothing=0 := by
  intro disabled
  have flag := valid.smoothingFlag
  rw [disabled] at flag
  have nonpositive : ¬0<p.smoothing := by simpa using flag.symm
  exact le_antisymm (le_of_not_gt nonpositive) valid.smoothingRange.1

def forwardLoss (p : Parameters) (x : Nat → ℝ) (target : Int) (block : Nat) : ℝ :=
  if p.split then RealSource.partialLoss p x target block
  else Math.loss p.cols x target p.ignored p.smoothing p.squareScale

/-- Public mathematical forward endpoint on the actual bound Tensor words.
It constructs the ideal state and derives both independent Tensor outputs
internally for every launched row/block, with the full storage frame. -/
theorem forward (interpret : Word → ℝ) (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : ForwardInputs p views alloc bytes words)
    (_scalarProjection : ScalarProjection p interpret inputs.scalars)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret (Region.cast p.labels) words
    let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
    ∃ final, exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
      fpValue final views.lse (launch.block*p.rows+launch.row)=↑(Math.blockLSE p.cols p.block launch.block x) ∧
      fpValue final views.loss (launch.block*p.rows+launch.row)=
        ↑(forwardLoss p x (targetValue words views.labels launch.row) launch.block) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠p.lossBase+launch.block*p.rows+launch.row) →
        (region≠p.lse ∨ offset≠p.lseBase+launch.block*p.rows+launch.row) →
        final.mem region offset=initial.mem region offset) := by
  let initial := realState p interpret (Region.cast p.labels) words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
  have values : ∀ col, col<p.cols → fpValue initial views.logits (launch.row*p.cols+col)=↑(x col) := by
    intro col _
    exact float_input p interpret words _ views.logits (by simpa only [views.logitsView.regionEq] using inputs.typedViews) _
  have target := label_input p interpret words (Region.cast p.labels) views.labels views.labelsView.regionEq launch.row
  change targetValue initial views.labels launch.row=targetValue words views.labels launch.row at target
  have start := Math.block_starts p.cols p.block launch.block inputs.parameters.geometry.widthPositive launch.blockBound
  cases splitFlag : p.split
  · have covered := inputs.splitCoverage splitFlag
    have zero : launch.block=0 := by
      by_contra nonzero
      have lower := Nat.mul_le_mul_right p.block (show 1≤launch.block by omega)
      simp only [Nat.one_mul] at lower
      omega
    obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelCorrect.forward_unsplit
      p views initial (real_binding p interpret _ words) launch.row launch.rowBound pids.1 (pids.2.trans zero) x values
      inputs.parameters.classesPositive covered splitFlag (no_smoothing p inputs.parameters)
      inputs.parameters.classStart inputs.parameters.totalClasses (by simpa only [target] using inputs.labelsValid launch.row launch.rowBound)
    refine ⟨final,execution,?_,?_,?_⟩
    · rw [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,views.lseView.regionEq]
      simpa only [zero,Nat.zero_mul,Nat.add_zero,Nat.zero_add,Math.block_zero_lse p.cols p.block covered] using lseRead
    · rw [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,views.lossView.regionEq]
      simpa only [zero,Nat.zero_mul,Nat.add_zero,Nat.zero_add,forwardLoss,splitFlag,Bool.false_eq_true,if_false,target] using lossRead
    · simpa only [zero,Nat.zero_mul,Nat.add_zero] using frame
  · obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelCorrect.forward_split
      p views initial (real_binding p interpret _ words) launch.row launch.block launch.rowBound pids.1 pids.2 x values
      inputs.parameters.geometry.widthPositive start splitFlag (no_smoothing p inputs.parameters)
    refine ⟨final,execution,?_,?_,frame⟩
    · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lseView launch.rowBound launch.blockBound,
        views.lseView.regionEq] using lseRead
    · simpa only [fpValue,output_index _ _ _ _ _ _ _ views.lossView launch.rowBound launch.blockBound,
        views.lossView.regionEq,forwardLoss,splitFlag,if_true,target] using lossRead

/-- Public backward endpoint uses the actual logical logits, LSE, targets,
and strided upstream Tensor. The source run and every gradient output are
proved, including ignored rows and the allowed in-place alias. -/
theorem backward (interpret : Word → ℝ) (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (inputs : BackwardInputs p views alloc bytes words)
    (_scalarProjection : ScalarProjection p interpret inputs.scalars)
    (launch : Launch p) (pids : LaunchState p launch words) :
    let initial := realState p interpret (Region.cast p.labels) words
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
  let initial := realState p interpret (Region.cast p.labels) words
  let x := fun col => logicalReal interpret words views.logits (launch.row*p.cols+col)
  have values : ∀ col, col<p.cols → fpValue initial views.logits (launch.row*p.cols+col)=↑(x col) := by
    intro col _
    exact float_input p interpret words _ views.logits (by simpa only [views.logitsView.regionEq] using inputs.typedViews.1) _
  have lseRead := float_input p interpret words _ views.lse (by simpa only [views.lseView.regionEq] using inputs.typedViews.2.1) launch.row
  have upstreamRead := float_input p interpret words _ views.upstream (by simpa only [views.upstreamView.regionEq] using inputs.typedViews.2.2) launch.row
  have target := label_input p interpret words (Region.cast p.labels) views.labels views.labelsView.regionEq launch.row
  change targetValue initial views.labels launch.row=targetValue words views.labels launch.row at target
  obtain ⟨final,execution,reads,frame⟩ := KernelCorrect.backward p views initial (real_binding p interpret _ words) launch.row launch.block launch.rowBound
    pids.1 pids.2 x (logicalReal interpret words views.lse launch.row) (logicalReal interpret words views.upstream launch.row)
    values lseRead (fun _ => upstreamRead) (no_smoothing p inputs.parameters)
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  simpa only [fpValue,matrix_index _ _ _ _ _ _ _ _ views.gradientView launch.rowBound active,
    views.gradientView.regionEq,target,Source.column,Source.gradientAddress] using reads lane active

#axiomsClean forward
#axiomsClean backward
#specNonCircular logicalReal avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelTensor

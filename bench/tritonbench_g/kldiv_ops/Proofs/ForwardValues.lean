import bench.tritonbench_g.kldiv_ops.Proofs.HostForward
import bench.tritonbench_g.kldiv_ops.Proofs.FPGraph
import bench.tritonbench_g.kldiv_ops.Proofs.NativeBinding

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

 theorem dense_neg_irrelevant {Word : Type} (codec : Codec Word) (neg : Word→Word) (m : Heap)
    (t : Tensor) (d : Dense t) (i : Nat) : value codec neg m t i=value codec (fun x=>x) m t i := by
  simp [value,d.normal]

namespace ForwardResult
variable {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
  {M : Algebra Word} {api : Services Word} {compiler : Compiler}
  {prediction target : Tensor} {flag : Bool} {name : String} {eps : Word}
  {entry : State} {returned : Tensor} {final : State}
variable (f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final)

theorem prediction_read (i : Nat) (hi : i<prediction.numel) :
    ((represented codec f.launch.plan.arena f.trace.ready.heap).mem "Y" i).read .real=
      value codec num.neg entry.heap prediction i := by
  rw [represented_prediction]
  change load codec f.trace.ready.heap ((denseView f.trace.py f.launch.plan.predictionDense).address i)=_
  rw [dense_value codec num.neg _ f.trace.py f.launch.plan.predictionDense i (by
    simpa only [Tensor.numel,f.launch.plan.predictionShape,f.predictionShape] using hi)]
  exact f.predictionValues i hi

theorem target_read (i : Nat) (hi : i<target.numel) :
    ((represented codec f.launch.plan.arena f.trace.ready.heap).mem "T" i).read .real=
      value codec num.neg entry.heap target i := by
  rw [represented_target]
  change load codec f.trace.ready.heap ((denseView f.trace.pt f.launch.plan.targetDense).address i)=_
  rw [dense_value codec num.neg _ f.trace.pt f.launch.plan.targetDense i (by
    simpa only [Tensor.numel,f.targetPackedShape] using hi)]
  exact f.targetValues i hi

theorem output_read (i : Nat) (hi : i<f.trace.out.numel) :
    value codec num.neg f.trace.launched.heap f.trace.out i=(f.launch.final.mem "O" i).read .real := by
  rw [dense_neg_irrelevant codec num.neg _ f.trace.out f.launch.plan.outputDense i]
  exact f.launch.readback i hi

theorem output_element (capacity : prediction.numel≤target.numel) (mode : f.trace.red=.none)
    (i : Nat) (hi : i<prediction.numel) :
    value codec num.neg f.trace.launched.heap f.trace.out i=
      Tile.scalar M flag eps (value codec num.neg entry.heap prediction i) (value codec num.neg entry.heap target i) := by
  have hn : prediction.numel=f.trace.rows*f.trace.cols := by simp [Tensor.numel,f.predictionShape]
  have hc : 0<f.trace.cols := by rw [hn] at hi; nlinarith
  have hr : i/f.trace.cols<f.trace.rows := (Nat.div_lt_iff_lt_mul hc).mpr (by simpa only [hn] using hi)
  have hm : i%f.trace.cols<f.trace.cols := Nat.mod_lt i hc
  have decomp : (i/f.trace.cols)*f.trace.cols+i%f.trace.cols=i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i f.trace.cols
  rw [f.output_read i (by simpa [Tensor.numel,f.launch.plan.outputShape,Reduction.outputShape,mode,f.predictionShape] using hi),f.launch.model]
  have value := Program.grid_value M (params f.trace.rows f.trace.cols) flag eps (List.range f.trace.rows)
    (represented codec f.launch.plan.arena f.trace.ready.heap).mem (i/f.trace.cols) (i%f.trace.cols)
    (List.mem_range.mpr hr) (block_positive f.trace.cols) hm (by change ("Y" : VeriTile.Triton.RegionName)≠"O"; decide) (by change ("T" : VeriTile.Triton.RegionName)≠"O"; decide)
  have cell := congrArg (fun c : VeriTile.Triton.FP.Structural.Cell Word=>c.read .real) value
  simpa only [params,mode,Reduction.code,decomp,f.prediction_read i hi,
    f.target_read i (by omega),VeriTile.Triton.FP.Structural.Cell.read_same] using cell

theorem output_row (capacity : prediction.numel≤target.numel) (mode : f.trace.red≠.none)
    (row : Nat) (hr : row<f.trace.rows) :
    value codec num.neg f.trace.launched.heap f.trace.out row=
      FPGraph.row M f.trace.cols (blockSize f.trace.cols) flag eps
        (fun col=>value codec num.neg entry.heap prediction (row*f.trace.cols+col))
        (fun col=>value codec num.neg entry.heap target (row*f.trace.cols+col)) := by
  have hcode : f.trace.red.code≠0 := by cases h : f.trace.red <;> simp_all [Reduction.code]
  rw [f.output_read row (by simpa [Tensor.numel,f.launch.plan.outputShape,Reduction.outputShape,mode] using hr),f.launch.model]
  have computed := FPGraph.grid_binding M (params f.trace.rows f.trace.cols) flag f.trace.red.code eps
    (List.range f.trace.rows) (represented codec f.launch.plan.arena f.trace.ready.heap).mem row
    (List.mem_range.mpr hr) hcode (by change ("Y" : VeriTile.Triton.RegionName)≠"O"; decide) (by change ("T" : VeriTile.Triton.RegionName)≠"O"; decide)
  have read := congrArg (fun c : VeriTile.Triton.FP.Structural.Cell Word=>c.read .real) computed
  dsimp only [params] at read ⊢
  rw [read]
  change FPGraph.row M f.trace.cols (blockSize f.trace.cols) flag eps _ _=_
  apply FPGraph.accumulate_congr
  · intro col hc
    apply f.prediction_read
    simpa [Tensor.numel,f.predictionShape] using flat_index _ _ _ _ hr hc
  · intro col hc
    apply f.target_read
    have hi : row*f.trace.cols+col<prediction.numel := by
      simpa [Tensor.numel,f.predictionShape] using flat_index _ _ _ _ hr hc
    omega

end ForwardResult

/-- Full independent FP graph from the logical entry tensors. Reduced rows
retain vector sum + sequential carried add and the source's FP32 zero. -/
def forwardKernelValue (M : Algebra Word) (red : Reduction) (cols : Nat) (flag : Bool)
    (eps : Word) (prediction target : Nat→Word) (i : Nat) : Word :=
  if red=.none then Tile.scalar M flag eps (prediction i) (target i)
  else FPGraph.row M cols (blockSize cols) flag eps
    (fun col=>prediction (i*cols+col)) (fun col=>target (i*cols+col))

namespace ForwardResult
variable {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
  {M : Algebra Word} {api : Services Word} {compiler : Compiler}
  {prediction target : Tensor} {flag : Bool} {name : String} {eps : Word}
  {entry : State} {returned : Tensor} {final : State}
variable (f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final)

theorem returned_values (capacity : prediction.numel≤target.numel) (i : Nat) (hi : i<returned.numel) :
    value codec num.neg final.heap returned i=
      reducedWord num f.trace.red f.trace.rows f.trace.cols
        (forwardKernelValue M f.trace.red f.trace.cols flag eps
          (value codec num.neg entry.heap prediction) (value codec num.neg entry.heap target)) i := by
  rw [f.reduction.values i hi]
  have rowsEq : ∀ row,row<f.trace.rows → f.trace.red≠.none →
      value codec num.neg f.trace.launched.heap f.trace.out row=
      forwardKernelValue M f.trace.red f.trace.cols flag eps
        (value codec num.neg entry.heap prediction) (value codec num.neg entry.heap target) row := by
    intro row hr mode
    simpa only [forwardKernelValue,if_neg mode] using f.output_row capacity mode row hr
  cases mode : f.trace.red with
  | none =>
    simpa only [mode,reducedWord,forwardKernelValue,if_pos rfl] using f.output_element capacity mode i (by
      simpa [Tensor.numel,f.reduction.shape,mode,f.predictionShape] using hi)
  | sum =>
    simp only [reducedWord]
    congr 1
    apply List.map_congr_left
    intro row hr
    simpa only [mode] using rowsEq row (List.mem_range.mp hr) (by simp [mode])
  | mean =>
    simp only [reducedWord]
    congr 2
    apply List.map_congr_left
    intro row hr
    simpa only [mode] using rowsEq row (List.mem_range.mp hr) (by simp [mode])
  | batchmean =>
    simp only [reducedWord]
    congr 2
    apply List.map_congr_left
    intro row hr
    simpa only [mode] using rowsEq row (List.mem_range.mp hr) (by simp [mode])

end ForwardResult
#specNonCircular forwardKernelValue avoiding [Program.grid,Program.gridModel,Program.loopModel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host

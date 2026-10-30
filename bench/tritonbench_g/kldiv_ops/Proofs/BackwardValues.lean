import bench.tritonbench_g.kldiv_ops.Proofs.ForwardValues
import bench.tritonbench_g.kldiv_ops.Proofs.HostBackward
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardCoverage
import bench.tritonbench_g.kldiv_ops.Proofs.Broadcast

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

/-- The extra destination tail is the actual untouched empty_like payload.
Only the target-sized prefix has a derivative; no output-shape gate is added. -/
def backwardKernelValue (M : Algebra Word) (flag : Bool) (count : Nat)
    (target initialWork : Nat→Word) (i : Nat) : Word :=
  if i<count then Backward.scalar M flag (target i) else initialWork i

namespace BackwardResult
variable {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
  {M : Algebra Word} {api : Services Word} {compiler : Compiler}
  {target grad dst : Tensor} {flag : Bool} {entry : State} {returned : Tensor} {final : State}
variable (f : BackwardResult codec num M api compiler target grad dst flag entry returned final)

theorem target_read (i : Nat) (hi : i<target.numel) :
    ((represented codec f.launch.plan.arena f.trace.ready.heap).mem "T" i).read .real=
      value codec num.neg entry.heap target i := by
  rw [represented_target]
  change load codec f.trace.ready.heap ((denseView f.trace.pt f.launch.plan.targetDense).address i)=_
  rw [dense_value codec num.neg _ f.trace.pt f.launch.plan.targetDense i (by
    simpa only [Tensor.numel,f.launch.plan.targetShape,f.targetShape] using hi)]
  exact f.targetValues i hi

theorem output_read (i : Nat) (hi : i<f.trace.work.numel) :
    value codec num.neg f.trace.launched.heap f.trace.work i=(f.launch.final.mem "O" i).read .real := by
  rw [dense_neg_irrelevant codec num.neg _ f.trace.work f.launch.plan.outputDense i]
  exact f.launch.readback i hi

theorem work_prefix (capacity : target.numel≤dst.numel) (i : Nat) (hi : i<target.numel) :
    value codec num.neg f.trace.launched.heap f.trace.work i=
      Backward.scalar M flag (value codec num.neg entry.heap target i) := by
  have hn : target.numel=f.trace.rows*f.trace.cols := by simp [Tensor.numel,f.targetShape]
  have hc : 0<f.trace.cols := by rw [hn] at hi; nlinarith
  have hr : i/f.trace.cols<f.trace.rows := (Nat.div_lt_iff_lt_mul hc).mpr (by simpa only [hn] using hi)
  have hm : i%f.trace.cols<f.trace.cols := Nat.mod_lt i hc
  have decomp : (i/f.trace.cols)*f.trace.cols+i%f.trace.cols=i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i f.trace.cols
  rw [f.output_read i (by simpa [Tensor.numel,f.workShape] using (show i<dst.numel by omega)),f.launch.model]
  have value := Backward.grid_value M (params f.trace.rows f.trace.cols) flag (List.range f.trace.rows)
    (represented codec f.launch.plan.arena f.trace.ready.heap).mem (i/f.trace.cols) (i%f.trace.cols)
    (List.mem_range.mpr hr) (block_positive f.trace.cols) hm (by change ("T" : VeriTile.Triton.RegionName)≠"O"; decide)
  have cell := congrArg (fun c : VeriTile.Triton.FP.Structural.Cell Word=>c.read .real) value
  simpa only [params,decomp,f.target_read i hi,VeriTile.Triton.FP.Structural.Cell.read_same] using cell

theorem work_tail (i : Nat) (lo : target.numel ≤ i) (hi : i<dst.numel) :
    value codec num.neg f.trace.launched.heap f.trace.work i=value codec num.neg f.trace.ready.heap f.trace.work i := by
  have hw : i<f.trace.work.numel := by simpa only [Tensor.numel,f.workShape] using hi
  rw [f.output_read i hw,f.launch.model]
  have tail := Backward.grid_tail M (params f.trace.rows f.trace.cols) flag
    (represented codec f.launch.plan.arena f.trace.ready.heap).mem i (by
      simpa only [params,Tensor.numel,f.targetShape,List.prod_cons,List.prod_nil,Nat.mul_one] using lo)
  dsimp only [params] at tail ⊢
  rw [tail]
  change load codec f.trace.ready.heap ((denseView f.trace.work f.launch.plan.outputDense).address i)=_
  exact dense_value codec num.neg _ f.trace.work f.launch.plan.outputDense i hw

theorem work_values (capacity : target.numel≤dst.numel) (i : Nat) (hi : i<dst.numel) :
    value codec num.neg f.trace.launched.heap f.trace.work i=
      backwardKernelValue M flag target.numel (value codec num.neg entry.heap target)
        (value codec num.neg f.trace.ready.heap f.trace.work) i := by
  by_cases active : i<target.numel
  · simpa only [backwardKernelValue,active,if_true] using f.work_prefix capacity i active
  · simpa only [backwardKernelValue,active,if_false] using f.work_tail i (by omega) hi

theorem destination_values (capacity : target.numel≤dst.numel) (i : Nat) (hi : i<dst.numel) :
    value codec num.neg final.heap dst i=
      backwardKernelValue M flag target.numel (value codec num.neg entry.heap target)
        (value codec num.neg f.trace.ready.heap f.trace.work) i := by
  rw [f.finish.destinationValues i hi]
  exact f.work_values capacity i hi

theorem gradient_compared (i : Nat) (hi : i<grad.numel) :
    value codec num.neg f.finish.compared.heap f.trace.pg i=value codec num.neg entry.heap grad i := by
  rw [f.finish.gradValues i (by simpa only [Tensor.numel,f.gradientShape] using hi)]
  exact f.gradientValues i hi

theorem scalar_test : f.finish.isOne=
    if grad.shape=[] then num.equal (value codec num.neg entry.heap grad 0) num.one else Bool.false := by
  rw [f.finish.test]
  by_cases shape : grad.shape=[]
  · have hi : 0<grad.numel := by simp [Tensor.numel,shape]
    simp [tensorEqual,f.gradientShape,shape,Tensor.numel,f.finish.scalar.shape,
      f.gradient_compared 0 hi,f.finish.scalarValue]
  · simp [tensorEqual,f.gradientShape,shape,f.finish.scalar.shape]

theorem returned_values (capacity : target.numel≤dst.numel) (i : Nat) (hi : i<returned.numel) :
    value codec num.neg final.heap returned i=
      if f.finish.isOne then backwardKernelValue M flag target.numel (value codec num.neg entry.heap target)
        (value codec num.neg f.trace.ready.heap f.trace.work) i
      else num.multiply
        (backwardKernelValue M flag target.numel (value codec num.neg entry.heap target)
          (value codec num.neg f.trace.ready.heap f.trace.work) (broadcastIndex returned.shape dst.shape i))
        (value codec num.neg entry.heap grad (broadcastIndex returned.shape grad.shape i)) := by
  cases mode : f.finish.isOne with
  | «true» =>
    have identity : returned=dst := (by simpa only [mode,if_true] using f.finish.returnCase : returned=dst ∧ final=f.finish.compared).1
    subst returned
    simpa only [mode,if_true] using f.destination_values capacity i hi
  | «false» =>
    have product : Product codec num dst f.trace.pg returned f.finish.compared final := by
      simpa only [mode,if_false] using f.finish.returnCase
    have bounds := broadcast_indices dst.shape f.trace.pg.shape returned.shape product.shape i hi
    rw [product.values i hi,f.finish.comparedValues _ bounds.1,f.work_values capacity _ bounds.1]
    rw [f.gradient_compared _ (by simpa only [Tensor.numel,f.gradientShape] using bounds.2)]
    simp [mode,f.gradientShape]

end BackwardResult
#specNonCircular backwardKernelValue avoiding [Backward.grid,Backward.gridModel,Backward.loopModel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host

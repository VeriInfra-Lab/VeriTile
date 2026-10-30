import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardValues
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- One source lane with actual input values. Loads remain barriers, and
products, subtraction, correction and final scaling retain source order. -/
def gradientScalar {α : Type} (M : FP32 α) (o : Options) (target : Int)
    (logit lse upstream : α) (col : Nat) : α :=
  let probability := M.exp (M.sub (M.mul (M.load logit) (M.literal o.logitScale)) (M.load lse))
  let corrected := M.add probability
    (M.mul (M.mul (M.mul (M.literal 2) (M.literal o.squareScale)) (M.load lse)) probability)
  let hit := (col : Int)=target-o.start
  let adjusted := if o.hasSmoothing then
    M.sub (if hit then M.sub corrected (M.sub (M.fromNat 1) (M.literal o.smoothing)) else corrected)
      (M.div (M.literal o.smoothing) (M.fromNat o.total))
    else if hit then M.sub corrected (M.literal 1) else corrected
  if target=o.ignored then M.literal 0 else M.mul (M.mul (M.load upstream) (M.literal o.logitScale)) adjusted

theorem backward_graph_lane {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {M : FP32 α} (A : Arithmetic α) {rt : Runtime α σ} (lib : Library obs A rt)
    {o : Options} {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    {e : BackwardData M rt o g x l y rows cols initial returned final} (k : BackwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (yLive : initial.live y.region) (gLive : initial.live g.region)
    (lLive : initial.live l.region) (row col : Nat) (hr : row<rows) (hc : col<cols) :
    k.gradientValues (row*cols+col)=gradientScalar M o (obs.integer initial y row)
      (value obs initial x (row*cols+col)) (value obs initial l row) (value obs initial g row) col := by
  have coordinates := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.block_coordinates cols col row hc
  let lane : Fin e.parameters.block := ⟨col%o.width,Nat.mod_lt col e.admitted.2.2.2.2.2.1⟩
  have columnEq : VeriTile.Bench.Repairs.CrossEntropy1.Source.column e.parameters.toParameters (col/o.width) lane.val=col := by
    simpa only [VeriTile.Bench.Repairs.CrossEntropy1.Source.column,lane,BackwardData.parameters,Host.parameters,Nat.mul_comm o.width]
      using Nat.div_add_mod col o.width
  have xRead := backward_kernel_logits k shape row col hr hc
  have yRead := backward_kernel_labels A lib k yLive row hr
  have gRead := backward_kernel_upstream k gLive row hr
  have lRead := backward_kernel_lse k lLive row hr
  have active : col<e.parameters.cols := hc
  simp only [BackwardKernelData.gradientValues,coordinates.1,coordinates.2]
  change backwardValues M e.parameters k.ready.memory row (col/o.width) lane=_
  simp only [backwardValues,VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits,
    columnEq,if_pos active,scaled,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.load,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.nat,VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.unary]
  change _=gradientScalar M o (obs.integer initial y row) _ _ _ col
  change BackwardGrid.label e.parameters k.ready.memory row=obs.integer initial y row at yRead
  simp only [BackwardGrid.label] at yRead
  simp only [BackwardGrid.logitAddress] at xRead
  rw [yRead,xRead,gRead,lRead]
  by_cases ignored : obs.integer initial y row=o.ignored
  · simp only [BackwardData.parameters,Host.parameters,gradientScalar,ignored,↓reduceIte]
  · simp only [BackwardData.parameters,Host.parameters,gradientScalar,ignored,↓reduceIte]

/-- Immediate composition supplies the backward normalizer from the actual
forward result. This theorem is polymorphic in arithmetic; no Real algebraic
identity or floating-point rewrite is used. -/
theorem complete_graph {α σ : Type} [Inhabited α] (obs : Observations α σ) (M : FP32 α)
    (A : Arithmetic α) (rt : Runtime α σ) (lib : Library obs A rt) (ints : IntegerViews obs)
    {o : Options} {g x y : Tensor} {rows cols : Nat} {initial saved final : World α σ}
    {returned : Return} {output : Tensor}
    {e : ForwardData M rt o x y rows cols initial returned saved} (k : ForwardKernelData obs e)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region)
    (b : BackwardData M rt o g x returned.2.1 y rows cols saved output final) :
    FloatArray obs saved returned.1 [rows] (finalLoss k A) ∧
    FloatArray obs saved returned.2.1 [rows] (finalLSE k A) ∧
    FloatArray obs saved returned.2.2 [rows] (finalZ k A) ∧
    FloatArray obs final output [rows,cols]
      (fun i => gradientScalar M o (obs.integer initial y (i/cols))
        (value obs initial x i) (finalLSE k A (i/cols)) (value obs saved g (i/cols)) (i%cols)) := by
  obtain ⟨loss,lse,z,labels,ls,lz,sz,frame⟩ := forward_values A lib ints k yLive
  obtain ⟨bk⟩ := backward_kernel_data obs A rt lib b shape (frame.live _ xLive) (frame.live _ yLive) gLive lse.live
  have array := backward_array bk shape
  refine ⟨loss,lse,z,⟨array.shapeEq,array.dtype,array.live,?_⟩⟩
  intro i hi
  have size : output.numel=rows*cols := by simp [Tensor.numel,array.shapeEq]
  have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot rows cols i (by simpa only [size] using hi)
  have result := backward_graph_lane A lib bk shape (frame.live _ yLive) gLive lse.live
    (i/cols) (i%cols) slot.2.1 slot.1
  rw [slot.2.2,frame.integers y (i/cols) yLive,frame.floatValue x xLive i,
    lse.values (i/cols) (by simpa [Tensor.numel,lse.shapeEq] using slot.2.1)] at result
  exact (array.values i hi).trans result

#axiomsClean backward_graph_lane
#axiomsClean complete_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host

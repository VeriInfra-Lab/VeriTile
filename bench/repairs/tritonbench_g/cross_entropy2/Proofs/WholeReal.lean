import bench.repairs.tritonbench_g.cross_entropy2.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations Preserves FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem forward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (v : Nat → Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (success : forward RealSource.algebra rt o x y initial=.ok returned final) :
    FloatArray obs final returned.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final returned.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final returned.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    returned.1.region≠returned.2.1.region ∧ returned.1.region≠returned.2.2.region ∧
    returned.2.1.region≠returned.2.2.region ∧ Preserves obs initial final := by
  obtain ⟨e⟩ := forward_success RealSource.algebra rt o x y rows cols initial returned final shape success
  obtain ⟨k⟩ := forward_kernel_data obs arithmetic rt lib e xLive yLive
  obtain ⟨loss,lse,z,labels,ls,lz,sz,frame⟩ := forward_values arithmetic lib ints k yLive
  have formulas := fun row hr => final_real_values lib k shape yLive row hr (v row) (caller row hr)
  refine ⟨⟨loss.shapeEq,loss.dtype,loss.live,?_⟩,⟨lse.shapeEq,lse.dtype,lse.live,?_⟩,
    ⟨z.shapeEq,z.dtype,z.live,?_⟩,ls,lz,sz,frame⟩
  · intro row hr
    exact (loss.values row hr).trans (formulas row (by simpa [Tensor.numel,loss.shapeEq] using hr)).1
  · intro row hr
    exact (lse.values row hr).trans (formulas row (by simpa [Tensor.numel,lse.shapeEq] using hr)).2.1
  · intro row hr
    exact (z.values row hr).trans (formulas row (by simpa [Tensor.numel,z.shapeEq] using hr)).2.2

theorem backward_row {σ : Type} {obs : Observations (WithBot ℝ) σ}
    {rt : Runtime (WithBot ℝ) σ} (lib : Library obs arithmetic rt)
    {o : Options} {g x l y : Tensor} {rows cols : Nat} {initial final : World (WithBot ℝ) σ} {returned : Tensor}
    {e : BackwardData RealSource.algebra rt o g x l y rows cols initial returned final}
    (k : BackwardKernelData obs e) (shape : x.shape=[rows,cols])
    (yLive : initial.live y.region) (gLive : initial.live g.region) (lLive : initial.live l.region)
    (row col : Nat) (hr : row<rows) (hc : col<cols) (v : Nat → ℝ) (upstream L : ℝ)
    (caller : ∀ j, j<cols → value obs initial x (row*cols+j)=↑(v j))
    (gValue : value obs initial g row=↑upstream) (lValue : value obs initial l row=↑L) :
    k.gradientValues (row*cols+col)=
      ↑(Math.gradientWithLSE o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale upstream L v col) := by
  have coordinates := VeriTile.Bench.Repairs.CrossEntropy1.RealHost.block_coordinates cols col row hc
  have input : RealSource.Input e.parameters.toParameters k.ready.memory row v := by
    intro j hj
    exact (backward_kernel_logits k shape row j hr hj).trans (caller j hj)
  have yRead := backward_kernel_labels arithmetic lib k yLive row hr
  have gRead := (backward_kernel_upstream k gLive row hr).trans gValue
  have lRead := (backward_kernel_lse k lLive row hr).trans lValue
  let lane : Fin e.parameters.block := ⟨col%o.width,Nat.mod_lt col e.admitted.2.2.2.2.2.1⟩
  have columnEq : VeriTile.Bench.Repairs.CrossEntropy1.Source.column e.parameters.toParameters (col/o.width) lane.val=col := by
    simpa only [VeriTile.Bench.Repairs.CrossEntropy1.Source.column,lane,BackwardData.parameters,Host.parameters,Nat.mul_comm o.width]
      using Nat.div_add_mod col o.width
  have result := RealSource.backward_lane e.parameters k.ready.memory row (col/o.width) v input (obs.integer initial y row)
    L upstream yRead lRead gRead lane (by simpa only [columnEq] using hc)
  simpa only [BackwardKernelData.gradientValues,coordinates.1,coordinates.2,columnEq,Math.gradientWithLSE,lane] using result

theorem backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (v : Nat → Nat → ℝ) (upstream L : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs initial g row=↑(upstream row))
    (lValues : ∀ row, row<rows → value obs initial l row=↑(L row))
    (success : backward RealSource.algebra rt o g x l y initial=.ok returned final) :
    FloatArray obs final returned [rows,cols]
      (fun i => ↑(Math.gradientWithLSE o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (L (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬initial.live returned.region ∧ Preserves obs initial final := by
  obtain ⟨e⟩ := backward_success RealSource.algebra rt o g x l y rows cols initial returned final shape success
  obtain ⟨k⟩ := backward_kernel_data obs arithmetic rt lib e shape xLive yLive gLive lLive
  have output := backward_array k shape
  refine ⟨⟨output.shapeEq,output.dtype,output.live,?_⟩,?_,backward_caller_frame ints k⟩
  · intro i hi
    have size : returned.numel=rows*cols := by simp [Tensor.numel,output.shapeEq]
    have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot rows cols i (by simpa only [size] using hi)
    have formula := backward_row lib k shape yLive gLive lLive (i/cols) (i%cols) slot.2.1 slot.1
      (v (i/cols)) (upstream (i/cols)) (L (i/cols)) (caller (i/cols) slot.2.1)
      (gValues (i/cols) slot.2.1) (lValues (i/cols) slot.2.1)
    rw [slot.2.2] at formula
    exact (output.values i hi).trans formula
  · intro live
    exact k.allocation.fresh
      (k.lse.materialized.frame.live _ (k.upstream.materialized.frame.live _
        (k.labels.materialized.frame.live _ (k.logits.materialized.frame.live _ live))))

/-- Immediate forward/backward composition supplies the saved normalizer
proved by the actual forward call; no LSE-output assumption is exposed. -/
theorem forward_backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World (WithBot ℝ) σ)
    (f : Return) (gradient : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region) (v : Nat → Nat → ℝ) (upstream : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs saved g row=↑(upstream row))
    (fSuccess : forward RealSource.algebra rt o x y initial=.ok f saved)
    (bSuccess : backward RealSource.algebra rt o g x f.2.1 y saved=.ok gradient final) :
    FloatArray obs final f.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final f.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final f.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final gradient [rows,cols]
      (fun i => ↑(Math.gradient cols o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬saved.live gradient.region ∧ Preserves obs initial final := by
  obtain ⟨loss,lse,z,ls,lz,sz,fFrame⟩ := forward_correct obs rt lib ints o x y rows cols initial saved f shape xLive yLive v caller fSuccess
  have savedCaller : ∀ row, row<rows → ∀ col, col<cols → value obs saved x (row*cols+col)=↑(v row col) := by
    intro row hr col hc
    exact (fFrame.floatValue x xLive _).trans (caller row hr col hc)
  have lseValues : ∀ row, row<rows → value obs saved f.2.1 row=↑(Math.LSE cols o.logitScale (v row)) := by
    intro row hr
    exact lse.values row (by simpa [Tensor.numel,lse.shapeEq] using hr)
  obtain ⟨grad,fresh,bFrame⟩ := backward_correct obs rt lib ints o g x f.2.1 y rows cols saved final gradient shape
    (fFrame.live _ xLive) (fFrame.live _ yLive) gLive lse.live v upstream (fun row => Math.LSE cols o.logitScale (v row))
    savedCaller gValues lseValues bSuccess
  have labels : obs.integer saved y=obs.integer initial y := funext (fun row => fFrame.integers y row yLive)
  refine ⟨loss.preserve bFrame,lse.preserve bFrame,z.preserve bFrame,?_,fresh,fFrame.trans bFrame⟩
  simpa only [labels,Math.gradientWithLSE,Math.gradient] using grad

#axiomsClean forward_correct
#axiomsClean backward_row
#axiomsClean backward_correct
#axiomsClean forward_backward_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RealHost

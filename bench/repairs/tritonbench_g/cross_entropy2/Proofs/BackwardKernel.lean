import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardKernel
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.DenseGeometry

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves Allocation FloatArray IntegerViews RowLayout rowStride value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

def BackwardData.parameters {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {o : Options}
    {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    (e : BackwardData M rt o g x l y rows cols initial returned final) : Parameters :=
  {Host.parameters o e.logits e.labels e.upstream e.lse e.lse rows cols Bool.false with
    gradient:=returned.region,gradientBase:=returned.storageOffset,gradientStride:=rowStride returned,
    upstream:=e.upstream.region,upstreamBase:=e.upstream.storageOffset,upstreamStride:=rowStride e.upstream}
def BackwardData.count {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {o : Options}
    {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    (_e : BackwardData M rt o g x l y rows cols initial returned final) : Nat :=
  VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width

theorem backward_empty {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (empty : p.rows=0) : backwardGrid M p m count=m := by
  simp only [backwardGrid,VeriTile.Bench.Repairs.CrossEntropy1.Grid.tasks,empty,List.range_zero]
  rfl

theorem backward_ready {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (M : FP32 α) (p : Parameters)
    (count : Nat) (initial computed : World α σ)
    (call : backwardLaunch M rt p count initial=.ok () computed) :
    ∃ ready, Preserves obs initial ready ∧
      computed={ready with memory:=BackwardGrid.backward M p ready.memory count} := by
  obtain ⟨trace⟩ := launch_trace p.rows (rt.setupBackward2 p count)
    (fun m => backwardGrid M p m count) initial computed call
  cases trace with
  | empty zero =>
    exact ⟨initial,Preserves.refl obs initial,by rw [← backward_models,backward_empty M p initial.memory count zero]⟩
  | dispatched nonzero ready hsetup =>
    exact ⟨ready,lib.setupBackward _ _ _ _ hsetup,by rw [backward_models]⟩

structure BackwardKernelData {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {g x l y : Tensor} {rows cols : Nat}
    {initial final : World α σ} {returned : Tensor}
    (e : BackwardData M rt o g x l y rows cols initial returned final) where
  logits : Normalized obs x e.logits initial e.normalizedLogits
  labels : Normalized obs y e.labels e.normalizedLogits e.normalizedLabels
  upstream : Normalized obs g e.upstream e.normalizedLabels e.normalizedUpstream
  lse : Normalized obs l e.lse e.normalizedUpstream e.normalizedLSE
  allocation : Allocation obs returned e.logits.shape e.logits.dtype e.logits.device e.normalizedLSE e.allocated
  dense : returned.Dense
  ready : World α σ
  setupFrame : Preserves obs e.allocated ready
  computedEq : final={ready with memory:=BackwardGrid.backward M e.parameters ready.memory e.count}
  separation : BackwardGrid.BackwardSeparation e.parameters

theorem backward_kernel_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) {M : FP32 α} {o : Options}
    {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    (e : BackwardData M rt o g x l y rows cols initial returned final) (shape : x.shape=[rows,cols])
    (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region) : Nonempty (BackwardKernelData obs e) := by
  have nx := normalize_values obs A rt lib _ _ _ _ xLive e.normalizeLogits
  have ny := normalize_values obs A rt lib _ _ _ _ (nx.materialized.frame.live _ yLive) e.normalizeLabels
  have fxy := nx.materialized.frame.trans ny.materialized.frame
  have ng := normalize_values obs A rt lib _ _ _ _ (fxy.live _ gLive) e.normalizeUpstream
  have fxyg := fxy.trans ng.materialized.frame
  have nl := normalize_values obs A rt lib _ _ _ _ (fxyg.live _ lLive) e.normalizeLSE
  have fx := ny.materialized.frame.trans (ng.materialized.frame.trans nl.materialized.frame)
  have fy := ng.materialized.frame.trans nl.materialized.frame
  have allocation := lib.base.emptyLike _ _ _ _ _ (fx.live _ nx.materialized.live) e.allocationCall
  have dense := allocation.2.1 rfl
  obtain ⟨ready,setup,computed⟩ := backward_ready obs A rt lib M e.parameters e.count _ _ e.launchCall
  have fresh := VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ allocation.1
  have sep : BackwardGrid.BackwardSeparation e.parameters := by
    refine ⟨?_,?_,?_,?_,?_⟩
    · exact fresh _ (fy.live _ ny.materialized.live)
    · exact fresh _ (nl.materialized.frame.live _ ng.materialized.live)
    · exact fresh _ nl.materialized.live
    · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.dense_rows returned rows cols dense
        (allocation.1.shapeEq.trans (nx.materialized.metadata.shape.trans shape)) allocation.1.rank
    · intro shared
      exact False.elim (fresh _ (fx.live _ nx.materialized.live) shared)
  exact ⟨⟨nx,ny,ng,nl,allocation.1,dense,ready,setup,computed,sep⟩⟩

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options}
  {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
  {e : BackwardData M rt o g x l y rows cols initial returned final}

def BackwardKernelData.lseFrame (k : BackwardKernelData obs e) : Preserves obs e.normalizedLSE k.ready :=
  k.allocation.frame.trans k.setupFrame
def BackwardKernelData.upstreamFrame (k : BackwardKernelData obs e) : Preserves obs e.normalizedUpstream k.ready :=
  k.lse.materialized.frame.trans k.lseFrame
def BackwardKernelData.labelsFrame (k : BackwardKernelData obs e) : Preserves obs e.normalizedLabels k.ready :=
  k.upstream.materialized.frame.trans k.upstreamFrame
def BackwardKernelData.logitsFrame (k : BackwardKernelData obs e) : Preserves obs e.normalizedLogits k.ready :=
  k.labels.materialized.frame.trans k.labelsFrame
def BackwardKernelData.initialFrame (k : BackwardKernelData obs e) : Preserves obs initial k.ready :=
  k.logits.materialized.frame.trans k.logitsFrame

def BackwardKernelData.gradientValues (k : BackwardKernelData obs e) (i : Nat) : α :=
  Source.backwardValues M e.parameters k.ready.memory (i/cols) ((i%cols)/o.width)
    ⟨(i%cols)%o.width,Nat.mod_lt _ e.admitted.2.2.2.2.2.1⟩

/-- Backward returns the first fresh contiguous allocation and derives every
logical element from the actual complete grid. Input regions may alias. -/
theorem backward_array (k : BackwardKernelData obs e) (shape : x.shape=[rows,cols]) :
    FloatArray obs final returned [rows,cols] k.gradientValues := by
  have shapeOut := k.allocation.shapeEq.trans (k.logits.materialized.metadata.shape.trans shape)
  have dtypeOut := k.allocation.dtypeEq.trans (k.logits.materialized.metadata.dtype.trans e.admitted.2.1)
  have layout := VeriTile.Bench.Repairs.CrossEntropy1.Host.dense_row_layout returned k.dense k.allocation.rank
  refine ⟨shapeOut,dtypeOut,?_,?_⟩
  · simpa only [k.computedEq] using k.setupFrame.live _ k.allocation.live
  intro i hi
  have size : returned.numel=rows*cols := by simp [Tensor.numel,shapeOut]
  have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot rows cols i (by simpa only [size] using hi)
  have positive : 0<e.parameters.block := e.admitted.2.2.2.2.2.1
  let col := i%cols
  let block := col/e.parameters.block
  let lane : Fin e.parameters.block := ⟨col%e.parameters.block,Nat.mod_lt col positive⟩
  have columnEq : BackwardGrid.column e.parameters block lane.val=col := by
    simpa only [BackwardGrid.column,VeriTile.Bench.Repairs.CrossEntropy1.Source.column,block,lane,Nat.mul_comm e.parameters.block]
      using Nat.div_add_mod col e.parameters.block
  have cover := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks_cover cols o.width positive
  have blockBound : block<e.count := by
    apply (Nat.div_lt_iff_lt_mul positive).mpr
    exact lt_of_lt_of_le slot.1 cover
  have stored := BackwardGrid.backward_output M e.parameters k.ready.memory e.count (i/cols,block)
    ((VeriTile.Bench.Repairs.CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨slot.2.1,blockBound⟩) k.separation lane
    (by simpa only [columnEq] using slot.1)
  have coordinate := layout rows cols shapeOut (i/cols) col slot.2.1 slot.1
  rw [slot.2.2] at coordinate
  simp only [k.computedEq,value,k.dense.1,Bool.false_eq_true,↓reduceIte,coordinate]
  change (BackwardGrid.backward M e.parameters k.ready.memory e.count e.parameters.gradient
    (BackwardGrid.gradientAddress e.parameters (i/cols) col)).read .real=_
  rw [columnEq] at stored
  rw [stored,VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell.read_same]
  rfl

theorem backward_caller_frame (ints : IntegerViews obs) (k : BackwardKernelData obs e) :
    Preserves obs initial final := by
  have before := (k.logits.materialized.frame.trans k.labels.materialized.frame).trans
    (k.upstream.materialized.frame.trans k.lse.materialized.frame)
  have fresh : ¬initial.live returned.region := fun live => k.allocation.fresh (before.live _ live)
  have unchanged : ∀ r, initial.live r → ∀ i,
      BackwardGrid.backward M e.parameters k.ready.memory e.count r i=k.ready.memory r i := by
    intro r live i
    apply BackwardGrid.backward_complete_frame M e.parameters k.ready.memory e.count k.separation
    intro q hq written
    obtain ⟨lane,active,equal⟩ := written
    have region := congrArg Prod.fst equal
    change r=e.parameters.gradient at region
    exact fresh (by simpa only [BackwardData.parameters] using (region ▸ live : initial.live e.parameters.gradient))
  rw [k.computedEq]
  refine ⟨k.initialFrame.live,?_,?_⟩
  · intro r i live
    exact (unchanged r live i).trans (k.initialFrame.memory r i live)
  · intro t i live
    exact (ints.localRead {k.ready with memory:=BackwardGrid.backward M e.parameters k.ready.memory e.count}
      k.ready t i rfl (unchanged t.region live)).trans (k.initialFrame.integers t i live)

#axiomsClean backward_kernel_data
#axiomsClean backward_array
#axiomsClean backward_caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host

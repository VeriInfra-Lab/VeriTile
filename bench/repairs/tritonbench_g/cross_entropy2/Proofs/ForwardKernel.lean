import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Trace
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Arrays
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.WholeGrids
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardArrays

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves Allocation FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

def ForwardData.parameters {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {o : Options}
    {x y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Return}
    (e : ForwardData M rt o x y rows cols initial returned final) : Parameters :=
  Host.parameters o e.logits e.labels e.loss e.lse e.zLoss rows cols e.split

/-- The skipped empty grid is exactly the identity; it needs no fabricated
setup call. This lets the readback theorem use one execution equation. -/
theorem forward_empty {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (empty : p.rows=0) : Grid.forward M p m count=m := by
  simp only [Grid.forward,VeriTile.Bench.Repairs.CrossEntropy1.Grid.tasks,empty,List.range_zero]
  rfl

theorem forward_ready {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (M : FP32 α) (p : Parameters)
    (count : Nat) (initial computed : World α σ)
    (call : forwardLaunch M rt p count initial=.ok () computed) :
    ∃ ready, Preserves obs initial ready ∧
      computed={ready with memory:=Grid.forward M p ready.memory count} := by
  obtain ⟨trace⟩ := launch_trace p.rows (rt.setupForward2 p count)
    (fun m => Grid.forward M p m count) initial computed call
  cases trace with
  | empty zero => exact ⟨initial,Preserves.refl obs initial,by rw [forward_empty M p initial.memory count zero]⟩
  | dispatched nonzero ready hsetup => exact ⟨ready,lib.setupForward _ _ _ _ hsetup,rfl⟩

structure ForwardKernelData {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {x y : Tensor} {rows cols : Nat}
    {initial final : World α σ} {returned : Return}
    (e : ForwardData M rt o x y rows cols initial returned final) where
  logits : Normalized obs x e.logits initial e.normalizedLogits
  labels : Normalized obs y e.labels e.normalizedLogits e.normalizedLabels
  lossAllocation : Allocation obs e.loss (if e.count>1 then [e.count,rows] else [rows])
    "torch.float32" e.logits.device e.normalizedLabels e.lossAllocated
  lseAllocation : Allocation obs e.lse (if e.count>1 then [e.count,rows] else [rows])
    "torch.float32" e.logits.device e.lossAllocated e.lseAllocated
  zAllocation : Allocation obs e.zLoss (if e.count>1 then [e.count,rows] else [rows])
    "torch.float32" e.logits.device e.lseAllocated e.zAllocated
  lossDense : e.loss.Dense
  lseDense : e.lse.Dense
  zDense : e.zLoss.Dense
  ready : World α σ
  setupFrame : Preserves obs e.zAllocated ready
  computedEq : e.computed={ready with memory:=Grid.forward M e.parameters ready.memory e.count}
  separation : Grid.Separation e.parameters

theorem forward_kernel_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) {M : FP32 α} {o : Options}
    {x y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Return}
    (e : ForwardData M rt o x y rows cols initial returned final)
    (xLive : initial.live x.region) (yLive : initial.live y.region) : Nonempty (ForwardKernelData obs e) := by
  have nx := normalize_values obs A rt lib _ _ _ _ xLive e.normalizeLogits
  have ny := normalize_values obs A rt lib _ _ _ _ (nx.materialized.frame.live _ yLive) e.normalizeLabels
  have loss := lib.base.empty _ _ _ _ _ _ e.lossCall
  have lse := lib.base.empty _ _ _ _ _ _ e.lseCall
  have z := lib.base.empty _ _ _ _ _ _ e.zCall
  obtain ⟨ready,setup,computed⟩ := forward_ready obs A rt lib M e.parameters e.count _ _ e.launchCall
  have ls : e.loss.region≠e.lse.region := by intro eq; apply lse.1.fresh; simpa only [← eq] using loss.1.live
  have lz : e.loss.region≠e.zLoss.region := by intro eq; apply z.1.fresh; simpa only [← eq] using lse.1.frame.live _ loss.1.live
  have sz : e.lse.region≠e.zLoss.region := by intro eq; apply z.1.fresh; simpa only [← eq] using lse.1.live
  have xReady := ny.materialized.frame.live _ nx.materialized.live
  have sep : Grid.Separation e.parameters := by
    refine ⟨?_,?_,?_⟩
    · intro a b eq
      cases a <;> cases b <;>
        simp_all [Grid.region,ForwardData.parameters,Host.parameters,Ne.symm ls,Ne.symm lz,Ne.symm sz]
    · intro role
      cases role
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ loss.1 _ xReady
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ xReady)
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ z.1 _ (lse.1.frame.live _ (loss.1.frame.live _ xReady))
    · intro role
      cases role
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ loss.1 _ ny.materialized.live
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ ny.materialized.live)
      · exact VeriTile.Bench.Repairs.CrossEntropy1.Host.fresh_ne_live obs _ _ _ _ _ _ z.1 _ (lse.1.frame.live _ (loss.1.frame.live _ ny.materialized.live))
  exact ⟨⟨nx,ny,loss.1,lse.1,z.1,loss.2,lse.2,z.2,ready,setup,computed,sep⟩⟩

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options}
  {x y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Return}
  {e : ForwardData M rt o x y rows cols initial returned final}

def ForwardKernelData.labelsFrame (k : ForwardKernelData obs e) : Preserves obs e.normalizedLabels k.ready :=
  k.lossAllocation.frame.trans (k.lseAllocation.frame.trans (k.zAllocation.frame.trans k.setupFrame))
def ForwardKernelData.logitsFrame (k : ForwardKernelData obs e) : Preserves obs e.normalizedLogits k.ready :=
  k.labels.materialized.frame.trans k.labelsFrame
def ForwardKernelData.initialFrame (k : ForwardKernelData obs e) : Preserves obs initial k.ready :=
  k.logits.materialized.frame.trans k.logitsFrame

theorem forward_kernel_logits (k : ForwardKernelData obs e) (shape : x.shape=[rows,cols])
    (row col : Nat) (hr : row<rows) (hc : col<cols) :
    (k.ready.memory e.parameters.logits
      (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress e.parameters.toParameters row col)).read .real=
        value obs initial x (row*cols+col) := by
  have bound : row*cols+col<x.numel := by
    simp only [Tensor.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have h := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at h
    omega
  have read := k.logits.materialized.floats e.admitted.2.1 _ bound
  rw [← k.logitsFrame.floatValue e.logits k.logits.materialized.live] at read
  simpa only [value,k.logits.packed.1,Bool.false_eq_true,↓reduceIte,
    k.logits.rows rows cols (k.logits.materialized.metadata.shape.trans shape) row col hr hc,
    ForwardData.parameters,Host.parameters,VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress] using read

theorem forward_kernel_labels (A : Arithmetic α) (lib : Library obs A rt)
    (k : ForwardKernelData obs e) (yLive : initial.live y.region) (row : Nat) (hr : row<rows) :
    VeriTile.Bench.Repairs.CrossEntropy1.Source.label e.parameters.toParameters k.ready.memory row=
      obs.integer initial y row := by
  have inputSize : y.numel=rows := by simp [Tensor.numel,e.admitted.2.2.2.1]
  have normalized := k.labels.int32 e.admitted.2.2.1 row (by simpa only [inputSize] using hr)
  have raw := lib.int32Raw e.labels e.normalizedLabels row k.labels.materialized.live
    (k.labels.materialized.metadata.dtype.trans e.admitted.2.2.1) k.labels.packed
    (by simpa only [k.labels.materialized.metadata.numel,inputSize] using hr)
  have frame := k.labelsFrame.memory e.labels.region (e.labels.storageOffset+row) k.labels.materialized.live
  change (k.ready.memory e.labels.region (e.labels.storageOffset+row)).read .int=_
  rw [frame,raw,normalized,k.logits.materialized.frame.integers y row yLive]

def ForwardKernelData.output (k : ForwardKernelData obs e) : Grid.Role → Tensor
  | .loss => e.loss | .lse => e.lse | .zLoss => e.zLoss
def ForwardKernelData.outputValues (k : ForwardKernelData obs e) (role : Grid.Role) (i : Nat) : α :=
  Grid.roleValue M e.parameters k.ready.memory (i%rows,i/rows) role

theorem forward_arrays (k : ForwardKernelData obs e) (role : Grid.Role) :
    FloatArray obs e.computed (k.output role) (if e.count>1 then [e.count,rows] else [rows]) (k.outputValues role) := by
  have positive : 0<e.count := by
    rw [e.countEq]
    exact VeriTile.Bench.Repairs.CrossEntropy1.Host.block_count_positive cols o.width e.admitted.1 e.admitted.2.2.2.2.2.1
  have shape : (k.output role).shape=(if e.count>1 then [e.count,rows] else [rows]) := by
    cases role
    exacts [k.lossAllocation.shapeEq,k.lseAllocation.shapeEq,k.zAllocation.shapeEq]
  have dtype : (k.output role).dtype="torch.float32" := by
    cases role
    exacts [k.lossAllocation.dtypeEq,k.lseAllocation.dtypeEq,k.zAllocation.dtypeEq]
  have dense : (k.output role).Dense := by
    cases role
    exacts [k.lossDense,k.lseDense,k.zDense]
  have live : k.ready.live (k.output role).region := by
    cases role
    · exact (k.lseAllocation.frame.trans (k.zAllocation.frame.trans k.setupFrame)).live _ k.lossAllocation.live
    · exact (k.zAllocation.frame.trans k.setupFrame).live _ k.lseAllocation.live
    · exact k.setupFrame.live _ k.zAllocation.live
  rw [k.computedEq]
  refine ⟨shape,dtype,live,?_⟩
  intro i hi
  have size := VeriTile.Bench.Repairs.CrossEntropy1.Host.output_numel (k.output role) e.count rows positive shape
  have slot := VeriTile.Bench.Repairs.CrossEntropy1.Host.flattened_slot e.count rows i (by simpa only [size] using hi)
  have base := VeriTile.Bench.Repairs.CrossEntropy1.Host.dense_base (k.output role) dense (by omega)
  have stored := Grid.complete_output M e.parameters k.ready.memory e.count k.separation
    (i%rows,i/rows) ((VeriTile.Bench.Repairs.CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨slot.1,slot.2.1⟩) role
  have region : Grid.region e.parameters role=(k.output role).region := by cases role <;> rfl
  have offset : Grid.base e.parameters role=(k.output role).storageOffset := by cases role <;> rfl
  rw [region,offset,base,Nat.zero_add] at stored
  change Grid.forward M e.parameters k.ready.memory e.count (k.output role).region (i/rows*rows+i%rows)=_ at stored
  rw [slot.2.2] at stored
  simpa only [value,dense.1,Bool.false_eq_true,↓reduceIte,dense.2 i hi,stored,
    VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell.read_same,ForwardKernelData.outputValues]

#axiomsClean forward_kernel_data
#axiomsClean forward_kernel_logits
#axiomsClean forward_kernel_labels
#axiomsClean forward_arrays
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host

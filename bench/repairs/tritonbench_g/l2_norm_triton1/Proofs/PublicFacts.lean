import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Execution

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Source result at original caller-entry logical values. The later Real and
native specifications replace this expression with independent formulas. -/
def expected {M α β γ : Type} [Inhabited α] (ops : Ops M α β) (neg : Negation α)
    (a : Source.Adapter α γ) (call : Call γ) (entry : World M) (row col : Nat) : α :=
  let N := call.input.shape.getLastD 0
  a.project col (Tile.scalar a.algebra N (blockSize N) call.eps
    (fun j => a.inject (entryRow ops neg call N row entry j)) col)

theorem entry_preserved {M α β γ : Type} [Inhabited α] (ops : Ops M α β)
    (neg : Negation α) (call : Call γ) (s t : World M) (domain : Domain call s)
    (keep : Keep ops s t) (cols row : Nat) (bounded : ∀ j,j<cols → row*cols+j<call.input.numel) :
    entryRow ops neg call cols row t=entryRow ops neg call cols row s := by
  funext j
  unfold entryRow
  split
  · rename_i hj
    unfold logical
    rw [keep.reads _ domain.live _ (domain.backed _ (bounded j hj))]
  · rfl

structure PublicFacts {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (neg : Negation α) (a : Source.Adapter α γ)
    (rt : Runtime M) (call : Call γ) (entry : World M) (out : View) (after : World M) : Prop where
  trace : SuccessTrace ops a rt call entry out after
  frame : Keep ops entry after
  shape : out.shape=call.input.shape
  count : out.numel=call.input.numel
  normal : out.negative=Bool.false
  device : out.storage.owner=call.input.storage.owner
  backed : out.backed
  live : out.storage∈after.allocations
  fresh : ∀ old∈entry.allocations,out.storage.separate old
  positive : 0<call.input.shape.getLastD 0
  accepted : call.input.shape.getLastD 0≤16384
  values : ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
    col<call.input.shape.getLastD 0 →
    logical ops neg after.memory out (row*(call.input.shape.getLastD 0)+col)=
      expected ops neg a call entry row col

theorem execute_success {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (neg : Negation α) (a : Source.Adapter α γ)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call γ)
    (entry : World M) (domain : Domain call entry) (out : View) (after : World M)
    (success : execute ops a rt call entry=.ok out after) :
    PublicFacts ops neg a rt call entry out after := by
  have trace := execute_trace ops a rt call entry out after success
  obtain ⟨selected,token,entered,q,ready,done,he,hp,hk,hl,hr⟩ := trace
  have enteredKeep : Keep ops entry entered := by simpa [he,Result.state] using lib.enter call.device entry
  have prepared := prepare_sound ops neg rt lib call entered
    ⟨domain.backed,enteredKeep.live _ domain.live,domain.device,domain.direct⟩ q ready hp
  have launched := launch_frame ops neg a rt lib q prepared.valid call.eps ready
    entered.allocations prepared.keep.live prepared.outputFresh
  have doneLive : q.output.view.storage∈done.allocations := by
    have h := launched.2 _ prepared.outputLive
    simpa [hk,Result.state] using h
  have left : Keep ops done after := by simpa [hl,Result.state] using lib.leave token done
  obtain ⟨shape,storage,normal,size,keys⟩ := lib.reshape q.output call.input.shape out hr
  have outputCount : q.output.count=q.snapshot.count := by
    simp only [Dense.count,prepared.outputRows,prepared.outputCols]
  have count : out.numel=call.input.numel := size.trans (outputCount.trans prepared.count)
  have boundedOutput : ∀ i,i<out.numel → i<q.output.count := by intro i hi; rwa [size] at hi
  refine ⟨⟨selected,token,entered,q,ready,done,he,hp,hk,hl,hr⟩,
    ?_,shape,count,normal,by rw [storage]; exact prepared.outputDevice,?_,?_,?_,?_,?_,?_⟩
  · simpa [success,Result.state] using execute_frame ops neg a rt lib call entry domain
  · intro i hi
    rw [storage,keys i (boundedOutput i hi)]
    exact Allocation.key_contains _ _ (lt_of_lt_of_le (boundedOutput i hi) q.output.capacity)
  · rw [storage]
    exact left.live _ doneLive
  · intro old hold
    rw [storage]
    exact prepared.outputFresh old (enteredKeep.live old hold)
  · rw [←prepared.cols]
    exact prepared.valid.positive
  · rw [←prepared.cols]
    exact le_trans prepared.valid.blockCovers prepared.valid.blockLimit
  · intro row col active hc
    have colBound : col<q.snapshot.cols := by rwa [prepared.cols]
    have indexBound : row*q.snapshot.cols+col<q.snapshot.count := by rwa [prepared.cols,prepared.count]
    have rowBound : row<q.snapshot.rows := by
      by_contra bad
      have bound := Nat.mul_le_mul_right q.snapshot.cols (Nat.le_of_not_gt bad)
      change row*q.snapshot.cols+col<q.snapshot.rows*q.snapshot.cols at indexBound
      omega
    have outputBound : row*q.snapshot.cols+col<q.output.count := by rwa [outputCount]
    have preserved := entry_preserved ops neg call entry entered domain enteredKeep q.snapshot.cols row
      (fun j hj => by rw [←prepared.count]; exact index_bound q.layout row j rowBound hj)
    have values := launch_values ops neg a rt lib q prepared.valid call.eps ready done prepared.inputLive hk row col rowBound colBound
    rw [prepared_inputs ops neg call entered q ready prepared row rowBound,preserved] at values
    unfold logical
    rw [normal]
    simp only [Negation.encode,Bool.false_eq_true,if_false]
    rw [←prepared.cols,keys _ outputBound]
    rw [left.reads _ doneLive _ (Allocation.key_contains _ _ (lt_of_lt_of_le outputBound q.output.capacity))]
    change ops.read done.memory (q.layout.ykey row col)=_
    rw [values]
    simp only [Source.Adapter.compute,Prepared.layout,expected,prepared.block,prepared.cols]

#axiomsClean execute_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host

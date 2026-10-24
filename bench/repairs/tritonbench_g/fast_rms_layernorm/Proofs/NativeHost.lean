import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeGraph
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPCurrent

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeHost
open VeriTile Triton FP.Structural Host NativeGraph
noncomputable section
set_option maxHeartbeats 6000000

/-- Separate source-stage algebras retain each launch's actual fusion policy. -/
def backwardRun (R : Reductions) (h : Call) (heap : FPGridForward.Heap Term) :=
  (FPGridWeight.run (algebra R .weight) h heap).bind (FPGridBackward.run (algebra R .input) h)

theorem backward_complete (R : Reductions) (h : Call) (heap : FPGridForward.Heap Term) :
    ∃ final,backwardRun R h heap=some final ∧
      (∀ row : Fin h.rows,∀ j,final (FPGridBackward.address h row j)=FPGridBackward.expected (algebra R .input) h heap row j) ∧
      (∀ pid : Fin (Geometry.featureCount h),∀ j,final (FPGridWeight.address h pid j)=FPGridWeight.expected (algebra R .weight) h heap pid j) ∧
      (∀ addr,(∀ row : Fin h.rows,¬FPGridBackward.writes h row addr) →
        (∀ pid : Fin (Geometry.featureCount h),¬FPGridWeight.writes h pid addr) → final addr=heap addr) := by
  obtain ⟨weights,hw,wo,wf⟩ := FPGridWeight.complete (algebra R .weight) h heap
  obtain ⟨final,hx,xo,xf⟩ := FPGridBackward.complete (algebra R .input) h weights
  refine ⟨final,by simp [backwardRun,hw,hx],?_,?_,?_⟩
  · intro row j
    rw [xo]
    apply FPGridBackward.expected_stable (algebra R .input) h row weights heap
    intro addr read
    apply wf
    intro pid
    rintro ⟨lane,eq⟩
    subst addr
    simp only [FPGridBackward.reads,FPGridBackward.writes,FPGridBackward.address,
      FPGridWeight.address] at read
    simp at read
  · intro pid j
    rw [xf]
    · exact wo pid j
    · intro row
      rintro ⟨lane,eq⟩
      exact (by decide : ("dw":RegionName)≠"dx") (congrArg Prod.fst eq)
  · intro addr noX noW
    exact (xf addr noX).trans (wf addr noW)

def zero : Cell Term := .mk .real (.bits 0)
theorem zero_bytes (I : Instructions) (S : Selection) (byte : Fin 4) : encode I S zero byte=0 := by
  apply Fin.ext
  simp [encode,observe,zero,Cell.read,eval,lower,Expr.eval]

structure BackwardSpec (R : Reductions) (ctx : Context (Cell Term)) (g : Upstream (Cell Term))
    (actual : FPGridForward.Heap Term) : Prop where
  input : ∀ row : Fin ctx.call.rows,∀ j,actual (FPGridBackward.address ctx.call row j)=
    FPGridBackward.expected (algebra R .input) ctx.call (seed ctx g) row j
  weight : ∀ pid : Fin (Geometry.featureCount ctx.call),∀ j,actual (FPGridWeight.address ctx.call pid j)=
    if ctx.call.rows=0 then zero else FPGridWeight.expected (algebra R .weight) ctx.call (seed ctx g) pid j

theorem backward_source {β : Type} (R : Reductions) (ctx : Context (Cell Term))
    (g : Upstream (Cell Term)) (metadata : Contract.TensorMetadata) (admitted : Contract.BackwardAdmitted ctx g metadata)
    (live : Caller) (encode : Cell Term → Fin 4 → β) (entry : Storage.Heap β)
    (p : BackwardPrepared ctx live) :
    ∃ actual,CheckedHost.backward (FPGridWeight.run (algebra R .weight)) (FPGridBackward.run (algebra R .input)) zero
        ctx g metadata live encode entry (.ok p) (.ok ()) (.ok ()) (.ok ())=
      finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual)) ∧
      BackwardSpec R ctx g actual := by
  rw [CheckedHost.backward_admitted _ _ _ ctx g metadata admitted]
  have valid:=Contract.backward_validated ctx g metadata admitted
  by_cases empty : ctx.call.rows=0
  · refine ⟨zeroWeight ctx.call zero (seed ctx g),Host.backward_empty _ _ _ _ _ _ _ _ _ valid empty (.ok ()) (.ok ()),?_,?_⟩
    · intro row;have hr:=row.isLt;omega
    · intro pid j;simp [zeroWeight,FPGridWeight.address,j.property,empty]
  · obtain ⟨actual,run,dx,dw,_⟩:=backward_complete R ctx.call (seed ctx g)
    change (FPGridWeight.run (algebra R .weight) ctx.call (seed ctx g)).bind (FPGridBackward.run (algebra R .input) ctx.call)=some actual at run
    cases weightRun : FPGridWeight.run (algebra R .weight) ctx.call (seed ctx g) with
    | none => simp [weightRun] at run
    | some weighted =>
      simp only [weightRun,Option.bind_some] at run
      refine ⟨actual,Host.backward_success _ _ _ _ _ _ _ _ _ valid empty weighted actual weightRun run (.ok ()),dx,?_⟩
      intro pid j
      simpa [empty] using dw pid j

/-- Whole checked forward, saved statistic, all four actual returned bytes and
caller frame. Only fixed-arity primitive instruction meanings remain open. -/
theorem forward (R : Reductions) (S : Selection) (I : Instructions) (binding : CompilerBinding R S)
    (r : Request (Cell Term)) (m : Contract.Metadata) (ok : Contract.Admitted r m)
    (scratch : FPGridForward.Heap Term) (live : Caller) (entry : Storage.Heap (Fin 256))
    (prepare : ∀ h,Preparation h .forward live)
    (p : Prepared (Contract.checkedCall r m ok) .forward live)
    (ready : prepare (Contract.checkedCall r m ok)=.ok p)
    (views : CheckedHost.DirectViews r m (Contract.checkedCall r m ok) live p)
    (inputs : CheckedHost.InputsAtEntry r (Contract.checkedCall r m ok) live p (compiledEncode binding I) entry) :
    let h:=Contract.checkedCall r m ok
    let outcome:=CheckedHost.forward (FPGridForward.run (algebra R .forward)) r m scratch live (compiledEncode binding I) entry prepare (.ok ())
    ∃ actual,outcome=finish p (compiledEncode binding I) entry actual (.ok (makeContext r h p actual,outputTensor r h p actual)) ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,∀ byte : Fin 4,
        outcome.bytes (Storage.address p.storage "y" (by simp [Storage.buffers]) (row.val*h.cols+j.val) byte)=
          encode I S (FPGridForward.expected (algebra R .forward) h (packed r scratch) row (.inl j)) byte) ∧
      (∀ row : Fin h.rows,observe I S (actual ("r",row.val))=
        observe I S (FPEndToEnd.referenceSaved (algebra R .forward) h (packed r scratch) row.val)) ∧
      (∀ address,live address → outcome.bytes address=entry address) := by
  dsimp only
  let h:=Contract.checkedCall r m ok
  obtain ⟨actual,run,source,spec⟩:=FPHost.forward_complete (algebra R .forward) r h scratch live
    (compiledEncode binding I) entry prepare p (Contract.checked_valid r m ok) ready
  have checked : CheckedHost.forward (FPGridForward.run (algebra R .forward)) r m scratch live
      (compiledEncode binding I) entry prepare (.ok ())=
      finish p (compiledEncode binding I) entry actual (.ok (makeContext r h p actual,outputTensor r h p actual)) := by
    rw [CheckedHost.forward_admitted _ r m ok];exact run
  refine ⟨actual,checked,?_,?_,?_⟩
  · intro row j byte
    rw [checked]
    have inside : row.val*h.cols+j.val<h.cells := by
      have hr:=row.isLt;have hj:=j.isLt;unfold Call.cells;nlinarith
    rw [HostStorage.forward_readback r h live p actual (compiledEncode binding I) entry _ inside byte]
    change compiledEncode binding I (actual ("y",row.val*h.cols+j.val)) byte=_
    rw [spec.output row j,compiled_encode]
  · intro row
    rw [spec.inv row]
  · intro address active
    rw [checked]
    exact Host.finish_frame p _ entry actual _ address active

/-- Whole checked backward over the currently retrieved saved X/W/r, with dW
completed before dX and an explicit all-zero empty-row result. -/
theorem backward (R : Reductions) (S : Selection) (I : Instructions) (binding : CompilerBinding R S)
    (ctx : Context (Cell Term)) (g : Upstream (Cell Term)) (metadata : Contract.TensorMetadata)
    (ok : Contract.BackwardAdmitted ctx g metadata) (live : Caller) (entry : Storage.Heap (Fin 256))
    (current : HostStorage.SavedAtEntry ctx (compiledEncode binding I) entry) (p : BackwardPrepared ctx live) :
    let outcome:=CheckedHost.backward (FPGridWeight.run (algebra R .weight)) (FPGridBackward.run (algebra R .input)) zero
      ctx g metadata live (compiledEncode binding I) entry (.ok p) (.ok ()) (.ok ()) (.ok ())
    ∃ actual,outcome=finish p.toPrepared (compiledEncode binding I) entry actual (.ok (gradientTensors ctx p actual)) ∧
      (∀ row : Fin ctx.call.rows,∀ j : Fin ctx.call.cols,∀ byte : Fin 4,
        outcome.bytes (Storage.address p.storage "dx" (by simp [Storage.buffers]) (row.val*ctx.call.cols+j.val) byte)=
          encode I S (FPGridBackward.expected (algebra R .input) ctx.call (seed ctx g) row j) byte) ∧
      (∀ pid : Fin (Geometry.featureCount ctx.call),∀ j : FPGridWeight.Output ctx.call pid,∀ byte : Fin 4,
        outcome.bytes (Storage.address p.storage "dw" (by simp [Storage.buffers]) (Geometry.column pid.val j.val) byte)=
          if ctx.call.rows=0 then 0 else encode I S (FPGridWeight.expected (algebra R .weight) ctx.call (seed ctx g) pid j) byte) ∧
      (∀ address,live address → outcome.bytes address=entry address) := by
  dsimp only
  obtain ⟨actual,run,spec⟩:=backward_source R ctx g metadata ok live (compiledEncode binding I) entry p
  refine ⟨actual,run,?_,?_,?_⟩
  · intro row j byte
    rw [run]
    have inside : row.val*ctx.call.cols+j.val<ctx.call.cells := by
      have hr:=row.isLt;have hj:=j.isLt;unfold Call.cells;nlinarith
    rw [(HostStorage.backward_readback ctx live p actual (compiledEncode binding I) entry).1 _ inside byte]
    change compiledEncode binding I (actual (FPGridBackward.address ctx.call row j)) byte=_
    rw [spec.input row j,compiled_encode]
  · intro pid j byte
    rw [run,(HostStorage.backward_readback ctx live p actual (compiledEncode binding I) entry).2 _ j.property byte]
    change compiledEncode binding I (actual (FPGridWeight.address ctx.call pid j)) byte=_
    rw [spec.weight pid j]
    split <;> simp [compiled_encode,zero_bytes]
  · intro address active
    rw [run]
    exact Host.finish_frame p.toPrepared _ entry actual _ address active

#axiomsClean backward_complete
#axiomsClean zero_bytes
#axiomsClean forward
#axiomsClean backward
#specNonCircular NativeGraph.eval avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeHost

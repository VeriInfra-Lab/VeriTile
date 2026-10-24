import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealEndToEnd
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostExecution

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealHost
open VeriTile.Triton Host
open RealGridForward (Heap read)
noncomputable section

def input (r : Request MemCell) (h : Call) (row : Fin h.rows) (j : Fin h.cols) : ℝ :=
  read (r.x (row.val*h.cols+j.val))
def weights (r : Request MemCell) (h : Call) (j : Fin h.cols) : ℝ := read (r.w j.val)
def upstream (g : Upstream MemCell) (h : Call) (row : Fin h.rows) (j : Fin h.cols) : ℝ :=
  read (g.values (row.val*h.cols+j.val))

structure ForwardSpec (r : Request MemCell) (h : Call) (actual : Heap) : Prop where
  output : ∀ row : Fin h.rows,∀ j : Fin h.cols,read (actual ("y",row.val*h.cols+j.val))=
    RealReference.output h.gemma h.eps (input r h row) (weights r h) j
  inv : ∀ row : Fin h.rows,read (actual ("r",row.val))=RealReference.inv h.eps (input r h row)
  snapshotX : ∀ off,actual ("x",off)=r.x off
  snapshotW : ∀ off,actual ("w",off)=r.w off

structure BackwardSpec (r : Request MemCell) (h : Call) (g : Upstream MemCell) (actual : Heap) : Prop where
  input : ∀ row : Fin h.rows,∀ j : Fin h.cols,read (actual ("dx",row.val*h.cols+j.val))=
    RealReference.inputGradient h.gemma h.eps (RealHost.input r h row) (weights r h) (upstream g h row) j
  weight : ∀ j : Fin h.cols,read (actual ("dw",j.val))=
    RealReference.weightGradient h.eps (RealHost.input r h) (upstream g h) j

theorem forward_source (r : Request MemCell) (h : Call) (scratch : Heap) :
    ∃ actual,RealGridForward.run h (packed r scratch)=some actual ∧ ForwardSpec r h actual := by
  obtain ⟨actual,he,ho,hf⟩:=RealGridForward.complete h (packed r scratch)
  have rawx (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry (packed r scratch) row.val h.rows) "x" h.cols=input r h row := by
    funext j
    simp [RealRows.raw,RealGridForward.entry,BlockState.readMem,input,RealGridForward.read,packed]
    rfl
  have raww (row : Fin h.rows) : RealRows.raw h (RealGridForward.entry (packed r scratch) row.val h.rows) "w" 0=weights r h := by
    funext j
    simp [RealRows.raw,RealGridForward.entry,BlockState.readMem,weights,RealGridForward.read,packed]
    rfl
  have snapshot (role : RegionName) (hr : role="x" ∨ role="w") (off : Nat) :
      actual (role,off)=packed r scratch (role,off) := by
    apply hf
    intro row
    rintro (⟨j,eq⟩|eq)
    all_goals
      have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
  refine ⟨actual,he,⟨?_,?_,?_,?_⟩⟩
  · intro row j
    simpa [RealGridForward.address,RealGridForward.expected,rawx,raww] using ho row (.inl j)
  · intro row
    simpa [RealGridForward.address,RealGridForward.expected,rawx] using ho row (.inr ())
  · intro off
    simpa using snapshot "x" (Or.inl rfl) off
  · intro off
    simpa using snapshot "w" (Or.inr rfl) off

def gHeap (g : Upstream MemCell) : Heap := fun a => g.values a.2

/-- The saved statistic is obtained by replaying the already proved actual
forward result, never by assuming a saved-output equation. -/
theorem backward_source (r : Request MemCell) (h : Call) (scratch forward : Heap)
    (forwardRun : RealGridForward.run h (packed r scratch)=some forward) (g : Upstream MemCell) :
    ∃ actual,RealSequence.run h (RealEndToEnd.seed forward (gHeap g))=some actual ∧ BackwardSpec r h g actual := by
  obtain ⟨f,b,run,_,dx,dw,_⟩:=RealEndToEnd.complete h (packed r scratch) (gHeap g)
  unfold RealEndToEnd.run at run
  rw [forwardRun] at run
  change (RealSequence.run h (RealEndToEnd.seed forward (gHeap g))).bind (fun b => some (forward,b))=some (f,b) at run
  cases back : RealSequence.run h (RealEndToEnd.seed forward (gHeap g)) with
  | none => simp [back] at run
  | some actual =>
    simp only [back,Option.bind_some] at run
    have same := congrArg Prod.snd (Option.some.inj run)
    change actual=b at same
    subst b
    have xs : RealSequence.data h (packed r scratch) "x"=input r h := by
      funext row j
      simp [RealSequence.data,input]
    have ws : RealEndToEnd.weights h (packed r scratch)=weights r h := by
      funext j
      simp [RealEndToEnd.weights,weights]
    have gs : RealSequence.data h (gHeap g) "dx"=upstream g h := by
      rfl
    rw [xs,ws,gs] at dx
    rw [xs,gs] at dw
    exact ⟨actual,rfl,⟨dx,dw⟩⟩

theorem empty_spec (r : Request MemCell) (h : Call) (g : Upstream MemCell) (saved : Heap)
    (empty : h.rows=0) : BackwardSpec r h g (zeroWeight h (.real 0) saved) := by
  letI : IsEmpty (Fin h.rows) := ⟨fun row => by have := row.isLt; omega⟩
  constructor
  · intro row
    exact isEmptyElim row
  · intro j
    rw [zeroWeight_read]
    change 0=RealReference.weightGradient h.eps (input r h) (upstream g h) j
    simp [RealReference.weightGradient]

theorem forward_complete {β : Type} (r : Request MemCell) (h : Call) (scratch : Heap)
    (live : Caller) (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared h .forward live)
    (valid : validate r=.ok h) (ready : prepare h=.ok p) :
    ∃ actual,Host.forward RealGridForward.run r scratch live encode entry prepare (.ok ())=
      finish p encode entry actual (.ok (makeContext r h p actual,outputTensor r h p actual)) ∧
      RealGridForward.run h (packed r scratch)=some actual ∧ ForwardSpec r h actual := by
  obtain ⟨actual,he,correct⟩:=forward_source r h scratch
  exact ⟨actual,Host.forward_success _ r scratch live encode entry prepare h p valid ready actual he,he,correct⟩

theorem backward_complete {β : Type} (r : Request MemCell) (h : Call) (scratch actualForward : Heap)
    (forwardLive : Caller) (fp : Prepared h .forward forwardLive)
    (forwardRun : RealGridForward.run h (packed r scratch)=some actualForward)
    (g : Upstream MemCell) (live : Caller) (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β)
    (p : BackwardPrepared (makeContext r h fp actualForward) live)
    (valid : backwardValidate (makeContext r h fp actualForward) g=.ok ()) :
    ∃ actual,Host.backward RealGridWeight.run RealGridBackward.run (.real 0)
      (makeContext r h fp actualForward) g live encode entry (.ok p) (.ok ()) (.ok ()) (.ok ())=
      finish p.toPrepared encode entry actual (.ok (gradientTensors (makeContext r h fp actualForward) p actual)) ∧
      BackwardSpec r h g actual := by
  let ctx:=makeContext r h fp actualForward
  by_cases empty : h.rows=0
  · exact ⟨zeroWeight h (.real 0) (seed ctx g),
      Host.backward_empty _ _ _ ctx g live encode entry p valid empty (.ok ()) (.ok ()),
      empty_spec r h g (seed ctx g) empty⟩
  · obtain ⟨actual,run,correct⟩:=backward_source r h scratch actualForward forwardRun g
    change (RealGridWeight.run h (seed ctx g)).bind (RealGridBackward.run h)=some actual at run
    cases weightedRun : RealGridWeight.run h (seed ctx g) with
    | none => simp [weightedRun] at run
    | some weighted =>
      simp only [weightedRun,Option.bind_some] at run
      exact ⟨actual,Host.backward_success _ _ _ ctx g live encode entry p valid empty weighted actual weightedRun run (.ok ()),correct⟩

#axiomsClean forward_source
#axiomsClean backward_source
#axiomsClean empty_spec
#axiomsClean forward_complete
#axiomsClean backward_complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealHost

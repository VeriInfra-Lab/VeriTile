import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPEndToEnd
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostExecution

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPHost
open VeriTile.Triton FP.Structural Host
open FPGridForward (Heap)
noncomputable section
variable {α : Type} [Inhabited α]

/-- These expected values are the independent scalar graph: fp32Load, every
cast, exact source reduction, rsqrt, and separate arithmetic are retained. -/
structure ForwardSpec (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch actual : Heap α) : Prop where
  output : ∀ row : Fin h.rows,∀ j : Fin h.cols,actual ("y",row.val*h.cols+j.val)=
    FPGridForward.expected M h (packed r scratch) row (.inl j)
  inv : ∀ row : Fin h.rows,actual ("r",row.val)=FPEndToEnd.referenceSaved M h (packed r scratch) row.val
  snapshotX : ∀ off,actual ("x",off)=r.x off
  snapshotW : ∀ off,actual ("w",off)=r.w off

def gHeap (g : Upstream (Cell α)) : Heap α := fun a => g.values a.2

/-- zero is the actual FP32 storage-zero primitive, not an asserted identity
between opaque literal/cast callbacks and all-bits-zero. -/
structure BackwardSpec (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch : Heap α)
    (g : Upstream (Cell α)) (zero : Cell α) (actual : Heap α) : Prop where
  input : ∀ row : Fin h.rows,∀ j : Fin h.cols,actual ("dx",row.val*h.cols+j.val)=
    FPGridBackward.expected M h (FPEndToEnd.referenceSeed M h (packed r scratch) (gHeap g)) row j
  weight : ∀ pid : Fin (Geometry.featureCount h),∀ j : FPGridWeight.Output h pid,
    actual (FPGridWeight.address h pid j)=
      if h.rows=0 then zero else FPGridWeight.expected M h (FPEndToEnd.referenceSeed M h (packed r scratch) (gHeap g)) pid j

theorem forward_source (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch : Heap α) :
    ∃ actual,FPGridForward.run M h (packed r scratch)=some actual ∧ ForwardSpec M r h scratch actual := by
  obtain ⟨actual,he,ho,hf⟩:=FPGridForward.complete M h (packed r scratch)
  have snapshot (role : RegionName) (hr : role="x" ∨ role="w") (off : Nat) :
      actual (role,off)=packed r scratch (role,off) := by
    apply hf
    intro row
    rintro (⟨j,eq⟩|eq)
    all_goals
      have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
  refine ⟨actual,he,⟨fun row j => ho row (.inl j),fun row => ho row (.inr ()),?_,?_⟩⟩
  · intro off
    simpa using snapshot "x" (Or.inl rfl) off
  · intro off
    simpa using snapshot "w" (Or.inr rfl) off

theorem backward_source (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch forward : Heap α)
    (forwardRun : FPGridForward.run M h (packed r scratch)=some forward) (g : Upstream (Cell α))
    (zero : Cell α) (nonempty : h.rows≠0) :
    ∃ actual,FPSequence.run M h (FPEndToEnd.seed forward (gHeap g))=some actual ∧
      BackwardSpec M r h scratch g zero actual := by
  obtain ⟨f,b,run,_,dx,dw,_⟩:=FPEndToEnd.complete M h (packed r scratch) (gHeap g)
  unfold FPEndToEnd.run at run
  rw [forwardRun] at run
  change (FPSequence.run M h (FPEndToEnd.seed forward (gHeap g))).bind (fun b => some (forward,b))=some (f,b) at run
  cases back : FPSequence.run M h (FPEndToEnd.seed forward (gHeap g)) with
  | none => simp [back] at run
  | some actual =>
    simp only [back,Option.bind_some] at run
    have same := congrArg Prod.snd (Option.some.inj run)
    change actual=b at same
    subst b
    exact ⟨actual,rfl,⟨dx,fun pid j => by simpa [nonempty] using dw pid j⟩⟩

theorem empty_spec (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch saved : Heap α)
    (g : Upstream (Cell α)) (zero : Cell α) (empty : h.rows=0) :
    BackwardSpec M r h scratch g zero (zeroWeight h zero saved) := by
  letI : IsEmpty (Fin h.rows) := ⟨fun row => by have := row.isLt; omega⟩
  constructor
  · intro row
    exact isEmptyElim row
  · intro pid j
    simp [zeroWeight,FPGridWeight.address,j.property,empty]

theorem forward_complete {β : Type} (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch : Heap α)
    (live : Caller) (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared h .forward live)
    (valid : validate r=.ok h) (ready : prepare h=.ok p) :
    ∃ actual,Host.forward (FPGridForward.run M) r scratch live encode entry prepare (.ok ())=
      finish p encode entry actual (.ok (makeContext r h p actual,outputTensor r h p actual)) ∧
      FPGridForward.run M h (packed r scratch)=some actual ∧ ForwardSpec M r h scratch actual := by
  obtain ⟨actual,he,correct⟩:=forward_source M r h scratch
  exact ⟨actual,Host.forward_success _ r scratch live encode entry prepare h p valid ready actual he,he,correct⟩

theorem backward_complete {β : Type} (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch actualForward : Heap α)
    (forwardLive : Caller) (fp : Prepared h .forward forwardLive)
    (forwardRun : FPGridForward.run M h (packed r scratch)=some actualForward)
    (g : Upstream (Cell α)) (zero : Cell α) (live : Caller) (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β)
    (p : BackwardPrepared (makeContext r h fp actualForward) live)
    (valid : backwardValidate (makeContext r h fp actualForward) g=.ok ()) :
    ∃ actual,Host.backward (FPGridWeight.run M) (FPGridBackward.run M) zero
      (makeContext r h fp actualForward) g live encode entry (.ok p) (.ok ()) (.ok ()) (.ok ())=
      finish p.toPrepared encode entry actual (.ok (gradientTensors (makeContext r h fp actualForward) p actual)) ∧
      BackwardSpec M r h scratch g zero actual := by
  let ctx:=makeContext r h fp actualForward
  by_cases empty : h.rows=0
  · exact ⟨zeroWeight h zero (seed ctx g),
      Host.backward_empty _ _ _ ctx g live encode entry p valid empty (.ok ()) (.ok ()),
      empty_spec M r h scratch (seed ctx g) g zero empty⟩
  · obtain ⟨actual,run,correct⟩:=backward_source M r h scratch actualForward forwardRun g zero empty
    change (FPGridWeight.run M h (seed ctx g)).bind (FPGridBackward.run M h)=some actual at run
    cases weightedRun : FPGridWeight.run M h (seed ctx g) with
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
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPHost

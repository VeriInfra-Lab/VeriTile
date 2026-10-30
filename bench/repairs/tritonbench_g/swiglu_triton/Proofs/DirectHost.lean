import bench.repairs.tritonbench_g.swiglu_triton.Proofs.InputGuards
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectNative
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Math

namespace VeriTile.Bench.Repairs.SwigluTriton.DirectHost
open VeriTile.Triton FP.Structural
open Memory DirectPhysical
noncomputable section
set_option maxHeartbeats 4000000

/-- A returned Torch view carries its actual object token, shape and shared
logical storage map. Forward returns flattened A/B and original-shaped C;
backward returns original-upstream-shaped views of the mutated A/B. -/
structure ReturnedTensor where
  object : Nat
  shape : List Nat
  key : Nat → Key

def returned (p : Prepared backward) : List ReturnedTensor :=
  let view := fun r shape => ReturnedTensor.mk (p.returnedObject r) shape (p.tensor r).key
  if backward then [view .a p.originalShape,view .b p.originalShape]
  else [view .a [p.rows,p.cols],view .b [p.rows,p.cols],view .third p.originalShape]

structure Plan (p : Prepared backward) where
  order : List Nat
  schedule : order.Perm (List.range p.rows)

/-- Ordinary Torch preparation/allocation/view receipt and runtime schedule.
Neither stage supplies computed values, a final heap, or a source result. -/
structure Runtime (backward : Bool) where
  prepare : Except String (Prepared backward)
  setup : (p : Prepared backward) → Except String (Plan p)

inductive Outcome (M : Type) where
  | failure (reason : String)
  | returned (tensors : List ReturnedTensor) (memory : M)

def execute {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (rt : Runtime backward) (s : M) : Outcome M :=
  match rt.prepare with
  | .error reason => .failure reason
  | .ok p =>
    if hc : inputCheck p = Bool.true then
      if blockSize p.cols ≤ 65536 then
        match rt.setup p with
        | .error reason => .failure reason
        | .ok plan => .returned (returned p)
          (DirectPhysical.run ops compute backward (layout p ((inputCheck_admitted_iff p).mp hc)) plan.order s)
      else .failure "block exceeds 65536"
    else .failure "input contract"

def Outside (p : Prepared backward) (h : Admitted p) (owner address : Nat) : Prop :=
  ∀ ch, enabled backward ch → ∀ row col, row < p.rows → col < p.cols →
    ¬((layout p h).key (outputRole ch) row col).covers owner address

def Values {M α β : Type} [Inhabited α] (ops : Ops M α β) (formula : Channel → α → α → α → α)
    (p : Prepared backward) (entry final : M) : Prop :=
  ∀ ch, enabled backward ch → ∀ row col, row < p.rows → col < p.cols →
    (p.tensor (outputRole ch)).read ops final (row*p.cols+col) =
      formula ch ((p.tensor .a).read ops entry (row*p.cols+col))
        ((p.tensor .b).read ops entry (row*p.cols+col))
        (if backward then (p.tensor .third).read ops entry (row*p.cols+col) else default)

def RunSpec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (formula : Channel → α → α → α → α) (p : Prepared backward) (h : Admitted p)
    (entry final : M) : Prop :=
  Values ops formula p entry final ∧
  (∀ owner address, Outside p h owner address → ops.observe final owner address = ops.observe entry owner address)

theorem run_spec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute formula : Channel → α → α → α → α) (same : ∀ ch a b d, compute ch a b d = formula ch a b d)
    (p : Prepared backward) (h : Admitted p) (plan : Plan p) (s : M) :
    RunSpec ops formula p h s (DirectPhysical.run ops compute backward (layout p h) plan.order s) := by
  have bounded : ∀ r ∈ plan.order, r < p.rows := by
    intro r hr; exact List.mem_range.mp (plan.schedule.mem_iff.mp hr)
  constructor
  · intro ch on row col hr hc
    have result := DirectPhysical.run_values ops compute backward (layout p h) (valid p h)
      plan.order (plan.schedule.nodup_iff.mpr List.nodup_range) bounded s ch on row col
      (plan.schedule.mem_iff.mpr (List.mem_range.mpr hr)) hc
    simpa only [value,key_binding p h _ row col hr hc,Tensor.read,same] using result
  · intro owner address outside
    apply DirectPhysical.run_bytes
    intro r hr ch on c hc
    exact outside ch on r c (bounded r hr) hc

/-- Source linkage records full execution and every backed cell for each row
in the same state sequence used by the returned physical memory. -/
def SourceSteps {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) : List Nat → M → Prop
  | [],_ => True
  | row::rest,s =>
    (∃ t, run algebra (directSource backward p.block p.cols).surfaceBody
        (DirectTile.rowState row (DirectSource.state ops backward p 0 s)) = some t ∧
      ∀ role offset, offset < (p.allocation role).words →
        t.mem (DirectSource.region backward role) offset = Cell.mk .real
          (ops.read (DirectPhysical.row ops (DirectTile.scalar algebra) backward p row s)
            ((p.allocation role).key offset))) ∧
    SourceSteps ops algebra backward p rest (DirectPhysical.row ops (DirectTile.scalar algebra) backward p row s)

theorem source_steps {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (valid : Valid p) (order : List Nat)
    (bound : ∀ r ∈ order, r < p.rows) (s : M) : SourceSteps ops algebra backward p order s := by
  induction order generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨?_,ih (fun r hr => bound r (by simp [hr])) _⟩
    simpa using DirectSource.row_refines ops algebra backward p valid 0 row (by simpa using bound row (by simp)) s

def NativeSteps {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.InstructionSemantics)
    (backward : Bool) (p : Layout) : List Nat → M → Prop
  | [],_ => True
  | row::rest,s =>
    (∃ t, run VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.graphAlgebra
        (directSource backward p.block p.cols).surfaceBody
        (DirectTile.rowState row (DirectNative.graphState ops backward p 0 s)) = some t ∧
      ∀ role offset, offset < (p.allocation role).words →
        Native.interpret (Native.compiledPorts instructions) ((t.mem (DirectSource.region backward role) offset).read .real) =
          ops.read (DirectPhysical.row ops (Native.sourceScalar (Native.compiledPorts instructions)) backward p row s)
            ((p.allocation role).key offset)) ∧
    NativeSteps ops instructions backward p rest
      (DirectPhysical.row ops (Native.sourceScalar (Native.compiledPorts instructions)) backward p row s)

theorem native_steps {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.InstructionSemantics)
    (backward : Bool) (p : Layout) (valid : Valid p) (order : List Nat)
    (bound : ∀ r ∈ order, r < p.rows) (s : M) : NativeSteps ops instructions backward p order s := by
  induction order generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨?_,ih (fun r hr => bound r (by simp [hr])) _⟩
    simpa using DirectNative.source_row_native_full ops instructions backward p valid 0 row
      (by simpa using bound row (by simp)) s

/-- Observable failures have their exact source stage; success includes the
returned view descriptors, complete formula/frame and source-linkage claim. -/
def HostSpec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (formula : Channel → α → α → α → α)
    (linked : (p : Prepared backward) → Admitted p → Plan p → M → Prop)
    (rt : Runtime backward) (entry : M) (outcome : Outcome M) : Prop :=
  (∃ reason, rt.prepare = .error reason ∧ outcome = .failure reason) ∨
  (∃ p, rt.prepare = .ok p ∧ inputCheck p = Bool.false ∧ outcome = .failure "input contract") ∨
  (∃ p, rt.prepare = .ok p ∧ inputCheck p = Bool.true ∧ ¬blockSize p.cols ≤ 65536 ∧ outcome = .failure "block exceeds 65536") ∨
  (∃ p reason, rt.prepare = .ok p ∧ inputCheck p = Bool.true ∧ blockSize p.cols ≤ 65536 ∧
    rt.setup p = .error reason ∧ outcome = .failure reason) ∨
  (∃ p h plan final, rt.prepare = .ok p ∧ rt.setup p = .ok plan ∧
    blockSize p.cols ≤ 65536 ∧ outcome = .returned (returned p) final ∧
    RunSpec ops formula p h entry final ∧ linked p h plan entry ∧
    (∀ row lane, row < p.rows → lane < blockSize p.cols →
      (BitVec.ofNat 64 row * BitVec.ofNat 64 p.cols + BitVec.ofNat 64 lane).toInt = (row*p.cols+lane : Nat)) ∧
    (∀ r row col, row < p.rows → col < p.cols →
      (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat + BitVec.ofNat 64 (4*(row*p.cols+col))).toNat =
        ((p.tensor r).key (row*p.cols+col)).address ∧ ((p.tensor r).key (row*p.cols+col)).address % 4 = 0) ∧
    (∀ lane, lane < blockSize p.cols →
      (BitVec.ofNat 32 lane).slt (BitVec.ofNat 32 p.cols) = decide (lane < p.cols)))

theorem host_spec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute formula : Channel → α → α → α → α) (same : ∀ ch a b d, compute ch a b d = formula ch a b d)
    (linked : (p : Prepared backward) → Admitted p → Plan p → M → Prop)
    (source : ∀ p h plan s, linked p h plan s) (rt : Runtime backward) (s : M) :
    HostSpec ops formula linked rt s (execute ops compute rt s) := by
  cases hp : rt.prepare with
  | error reason => exact Or.inl ⟨reason,hp,by simp [execute,hp]⟩
  | ok p =>
    cases hc : inputCheck p with
    | «false» => exact Or.inr (Or.inl ⟨p,hp,hc,by simp [execute,hp,hc]⟩)
    | «true» =>
      by_cases hs : blockSize p.cols ≤ 65536
      · cases setup : rt.setup p with
        | error reason => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨p,reason,hp,hc,hs,setup,by simp [execute,hp,hc,hs,setup]⟩)))
        | ok plan =>
          let h := (inputCheck_admitted_iff p).mp hc
          apply Or.inr; apply Or.inr; apply Or.inr; apply Or.inr
          refine ⟨p,h,plan,_,hp,setup,hs,?_,run_spec ops compute formula same p h plan s,
            source p h plan s,indices p h hs,pointer p h,mask p h hs⟩
          simp [execute,hp,hc,hs,setup,h]
      · exact Or.inr (Or.inr (Or.inl ⟨p,hp,hc,hs,by simp [execute,hp,hc,hs]⟩))

#axiomsClean host_spec
#axiomsClean source_steps
#axiomsClean native_steps
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.DirectHost

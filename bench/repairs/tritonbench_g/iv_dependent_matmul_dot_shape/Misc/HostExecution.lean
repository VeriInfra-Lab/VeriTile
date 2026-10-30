import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.GridExecution
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.Targets

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.HostExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000

def returned {α : Type} {r : Request} {before : Wrapper.Heap α}
    (p : Wrapper.Prepared α r before) (final : State α) : Wrapper.Returned α r :=
  ⟨if CheckedHost.Empty r then [.device,.randomA,.randomB,.emptyC,.returned]
    else [.device,.randomA,.randomB,.emptyC,.launch,.returned],
    p.c.tensor, {p.c.after with state := final}⟩

/-- Actual checked dispatch followed by the shared interpreter. The successful
allocation and submission boundary contains no numerical result. -/
def call {α Device Error : Type} [Inhabited α] (M : Algebra α) (g : Graph)
    (api : Wrapper.Calls α Device Error) (r : Request) (before : Wrapper.Heap α)
    (defaultFP32 cuda : Bool) : Option (Wrapper.Returned α r) :=
  if Contract.check r defaultFP32 cuda=.ok () then
    (Wrapper.prepare api r before).toOption.bind fun (d,p) =>
      let finish := (Kernel.grid M g p.request p.c.after.state).map (returned p)
      if CheckedHost.Empty r then finish
      else (Wrapper.launch api d p).toOption.bind fun _ => finish
  else none

theorem call_prepared {α Device Error : Type} [Inhabited α] (M : Algebra α) (g : Graph)
    (api : Wrapper.Calls α Device Error) (r : Request) (before : Wrapper.Heap α)
    (defaultFP32 cuda : Bool) (h : Contract.Preconditions r defaultFP32 cuda)
    (d : Device) (p : Wrapper.Prepared α r before) (hp : Wrapper.PreparedBy api r before d p)
    (hl : CheckedHost.LaunchReady api d p) (final : State α)
    (executed : Kernel.grid M g p.request p.c.after.state=some final) :
    call M g api r before defaultFP32 cuda=some (returned p final) := by
  have prepared : Wrapper.prepare api r before=.ok (d,p) := by
    simp only [Wrapper.prepare,hp.1,hp.2.1,hp.2.2.1,hp.2.2.2]
  have checked := (Contract.check_iff r defaultFP32 cuda).mpr h
  by_cases empty : CheckedHost.Empty r
  · simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,
      if_pos empty,executed,Option.map_some]
  · obtain ⟨support,launch⟩ := hl empty
    simp only [call,if_pos checked,prepared,Except.toOption,Option.bind_some,
      if_neg empty,launch,Option.bind_some,executed,Option.map_some]

theorem post_transfer {α : Type} {r : Request} {before : Wrapper.Heap α}
    (p : Wrapper.Prepared α r before) (old : Wrapper.Returned α r) (values : Nat → Nat → α)
    (logical : CheckedHost.Post p old values) (final : State α)
    (hm : final.mem=old.heap.state.mem) : CheckedHost.Post p (returned p final) values := by
  refine ⟨rfl,?_,?_,?_,?_⟩
  · change p.c.tensor.metadata=outputMetadata p.request
    rw [←logical.identity]
    exact logical.metadata
  · intro i hi j hj
    change final.mem _ _=_
    rw [hm]
    exact logical.values i hi j hj
  · intro region offset outside
    change final.mem _ _=_
    rw [hm]
    exact logical.frame region offset outside
  · intro old member offset
    change final.mem _ _=_
    rw [hm]
    exact logical.caller old member offset

theorem target_transfer (R : Instructions.Reference) {r : Request} {before : Wrapper.Heap Word}
    (p : Wrapper.Prepared Word r before) (out : Wrapper.Returned Word r)
    (logical : CheckedHost.Post p out
      (reference R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
        (Instructions.accumulatorPort (Instructions.referenceInstructions R)) .accumulator p.request p.b.after.state)) :
    CheckedHost.Post p out (Target.native R r.inner r.blockInner
      (Target.inputA p.request p.a.after.state) (Target.inputB p.request p.b.after.state)) := by
  refine ⟨logical.identity,logical.metadata,?_,logical.frame,logical.caller⟩
  intro i hi j hj
  have ha : Target.inputA p.request p.b.after.state=Target.inputA p.request p.a.after.state := by
    funext row inner
    unfold Target.inputA
    have frame := p.c.frame p.a.tensor.region (row*max r.inner 1+inner)
      (by intro he; exact p.ca he.1.symm)
    exact congrArg (fun cell => cell.read .fp32)
      (frame.symm.trans (p.memoryA (row*max r.inner 1+inner)))
  rw [logical.values i hi j hj,Target.reference_eq R p.request p.b.after.state i j hi hj,ha]
  rfl

/-- The byte machine executes each retained finite output-pointer store.
Its values equal the actual quotation's memory by the proved grid projection. -/
theorem bytes_from_quotation (rest : Algebra Word) (R : Instructions.Reference)
    {r : Request} {before : Wrapper.Heap Word} (p : Wrapper.Prepared Word r before)
    (h : Contract.Admitted r) (prepared : OutputBytes.Read)
    (bound : OutputBytes.Bound prepared (ByteHost.cView p) p.c.after.state)
    (final : State Word)
    (hm : final.mem=(Execution.host
      (Execution.fp R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
        (Instructions.accumulatorPort (Instructions.referenceInstructions R)) p.request)
      .accumulator p.a.tensor.base p.b.tensor.base p.c.tensor.base p.c.after.state).2.mem) :
    (ByteExecution.grid
      (Execution.fp R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
        (Instructions.accumulatorPort (Instructions.referenceInstructions R)) p.request)
      .accumulator p.a.tensor.base p.b.tensor.base p.c.tensor.base
      (List.range (programs p.request)) (p.c.after.state,prepared)).2 =
    OutputBytes.render prepared (ByteHost.cView p) final := by
  let a := p.allocation p.c.after.state p.typedA p.typedB
  have d := p.addressable p.c.after.state p.typedA p.typedB
  have l := IvDependentMatmul.Contract.launchable p.request
    (CheckedHost.prepared_admitted p h.toAdmitted)
  have run := ByteExecution.grid_render
    (Execution.fp R.cfg (Instructions.dotPort (Instructions.referenceInstructions R))
      (Instructions.accumulatorPort (Instructions.referenceInstructions R)) p.request)
    .accumulator p.c.after.state a d l prepared (List.range (programs p.request))
    (fun _ hi => List.mem_range.mp hi) p.c.after.state
  change ByteExecution.grid _ _ _ _ _ _
    (p.c.after.state,OutputBytes.render prepared (ByteHost.cView p) p.c.after.state)=_ at run
  rw [OutputBytes.render_initial prepared (ByteHost.cView p) p.c.after.state bound] at run
  have ha : a.baseA=p.a.tensor.base := rfl
  have hb : a.baseB=p.b.tensor.base := rfl
  have hc : a.baseC=p.c.tensor.base := rfl
  rw [ha,hb,hc] at run
  rw [run]
  funext address
  simp only [OutputBytes.render,hm,Execution.host,ByteHost.cView]

#axiomsClean call_prepared
#axiomsClean post_transfer
#axiomsClean target_transfer
#axiomsClean bytes_from_quotation
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.HostExecution

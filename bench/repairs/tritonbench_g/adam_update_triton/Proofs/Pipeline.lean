import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Preparation
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
noncomputable section
open Classical

structure PipelinePath {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (rt : Runtime M) (q : Request) (before after : World M) where
  buffers : Buffers
  prepared : World M
  preparation : prepare rt q before=.ok buffers prepared
  kernels : ChunksFacts ops model rt q buffers (Index.chunks 0 q.count) prepared after

theorem pipeline_path {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} (rt : Runtime M)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (args : Scalars (BitVec 64)) (params : Scalars α) (binding : Scalar.bind convert args=.ok params)
    (q : Request) (before after : World M)
    (ran : pipeline rt driver convert args q before=.ok () after) :
    Nonempty (PipelinePath ops (model params) rt q before after) := by
  unfold pipeline at ran
  cases prep : prepare rt q before with
  | error reason failed => simp [prep,Result.bind] at ran
  | ok b prepared =>
    simp only [prep,Result.bind] at ran
    exact ⟨⟨b,prepared,prep,chunks_facts driver convert args params binding q b _ prepared after ran⟩⟩

inductive RunPath {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Scalars α → Machine.Model α τ) (rt : Runtime M)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (call : Call) (args : Scalars (BitVec 64)) (params : Scalars α) (before after : World M) : Prop
  | empty (zero : call.request.count=0) (same : after=before)
  | nonempty (nonzero : call.request.count≠0)
      (context : ContextPath rt call.request.p.storage.owner
        (pipeline rt driver convert args call.request) before () after)
      (body : PipelinePath ops (model params) rt call.request context.entered context.completed)

theorem run_path {M α β τ : Type} [Inhabited α] [Inhabited τ]
    {ops : Ops M α β} {model : Scalars α → Machine.Model α τ} (rt : Runtime M)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (call : Call) (h : Admitted call) (args : Scalars (BitVec 64)) (params : Scalars α)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params)
    (before after : World M) (success : run rt driver convert call before=.ok () after) :
    RunPath ops model rt driver convert call args params before after := by
  unfold run at success
  simp only [checkInputs,(checkTensorInputs_iff _ _ _).mpr h] at success
  by_cases zero : call.request.count=0
  · rw [if_pos zero] at success
    exact .empty zero (Result.ok.inj success).2.symm
  · rw [if_neg zero,normalized zero] at success
    change contextual rt call.request.p.storage.owner (pipeline rt driver convert args call.request) before=.ok () after at success
    have context:=contextual_path rt call.request.p.storage.owner
      (pipeline rt driver convert args call.request) before
    rw [success] at context
    obtain ⟨context⟩:=context
    obtain ⟨body⟩:=pipeline_path rt driver convert args params (binding zero) call.request
      context.entered context.completed context.body
    exact .nonempty zero context body

structure WholeSpec {M α β τ : Type} [Inhabited τ] (ops : Ops M α β) (neg : Negation α)
    (model : Machine.Model α τ) (q : Request) (before after : World M) : Prop where
  values : ∀ output i,i<q.count → logical ops neg after.memory (outputView q output) i=
    Machine.scalar model output (logical ops neg before.memory q.p i)
      (logical ops neg before.memory q.grad i) (logical ops neg before.memory q.momentum i)
  frame : ∀ owner address,CallerBytes q owner address → Outside q owner address →
    ops.observe after.memory owner address=ops.observe before.memory owner address
  device : after.currentDevice=before.currentDevice
  streams : after.currentStreams=before.currentStreams

theorem run_spec {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (neg : Negation α) (model : Scalars α → Machine.Model α τ)
    (rt : Runtime M) (lib : Library rt) (driver : ChunkRuntime M α τ ops model rt)
    (convert : Scalar.Conversions α) (call : Call) (h : Admitted call) (tensor : Bindings call)
    (args : Scalars (BitVec 64)) (params : Scalars α)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params)
    (before after : World M) (success : run rt driver convert call before=.ok () after) :
    WholeSpec ops neg (model params) call.request before after := by
  have path:=run_path rt driver convert call h args params normalized binding before after success
  cases path with
  | empty zero same =>
    subst after
    refine ⟨?_,fun _ _ _ _=>rfl,rfl,rfl⟩
    intro output i active
    omega
  | nonempty nonzero context body =>
    obtain ⟨prepared,state⟩:=prepare_sound rt lib call.request context.entered body.prepared body.buffers body.preparation
    have valid:=prepared.valid call h tensor body.buffers (Nat.pos_of_ne_zero nonzero)
    have entered:=lib.enter _ _ _ _ context.enter
    have exited:=lib.exit _ _ _ context.exit
    have enough (role : Role) : call.request.count≤(call.request.view role).numel := by
      have cp:=tensor.count .a
      have cg:=tensor.count .third
      have cm:=tensor.count .b
      cases role
      · exact Nat.le_refl _
      · have := h.enough.2;change call.momentum.numel=call.request.momentum.numel at cm
        change call.parameter.numel=call.request.count at cp;change call.request.count≤call.request.momentum.numel;omega
      · have := h.enough.1;change call.gradient.numel=call.request.grad.numel at cg
        change call.parameter.numel=call.request.count at cp;change call.request.count≤call.request.grad.numel;omega
    have normal (role : Role) : (call.request.view role).negative=Bool.false := by
      cases role
      · exact tensor.parameter.negative.symm.trans h.noNeg.1
      · exact tensor.momentum.negative.symm.trans h.noNeg.2.2
      · exact tensor.gradient.negative.symm.trans h.noNeg.2.1
    have np : call.request.p.negative=Bool.false := normal .a
    have ng : call.request.grad.negative=Bool.false := normal .third
    have nm : call.request.momentum.negative=Bool.false := normal .b
    have entry (role : Role) (i : Nat) (hi : i<call.request.count) :
        ops.read body.prepared.memory (((body.buffers.layout call.request).allocation role).key i)=
          logical ops neg before.memory (call.request.view role) i := by
      rw [prepared.keys _ _ role i (lt_of_lt_of_le hi (enough role)),state,entered.2.1]
      simp [logical,Negation.encode,normal role]
    refine ⟨?_,?_,exited.2.1.trans entered.1,?_⟩
    · intro output i active
      have values:=Autotune.whole_values ops (model params) (body.buffers.layout call.request) valid
        (CallerBytes call.request) body.prepared.memory context.completed.memory body.kernels.source output i active
      rw [entry .a i active,entry .third i active,entry .b i active] at values
      have keys:=prepared.keys call.request body.buffers (Private.outputRole output) i
        (lt_of_lt_of_le active (enough _))
      rw [keys] at values
      rw [exited.1]
      cases output <;> simpa [logical,Negation.encode,outputView,Request.view,Private.outputRole,np,ng,nm] using values
    · intro owner address kept outside
      rw [exited.1]
      have keep:=Autotune.launches_bytes ops (model params) (body.buffers.layout call.request)
        (CallerBytes call.request) (Index.chunks 0 call.request.count) body.prepared.memory context.completed.memory
        body.kernels.source owner address kept
      rw [keep,state,entered.2.1]
      intro chunk member i selected output
      have bounds:=(Index.chunks_bounds 0 call.request.count chunk member).2.2.2
      have ir:=List.mem_range'_1.mp selected
      have active : i<call.request.count := by omega
      rw [prepared.keys _ _ (Private.outputRole output) i (lt_of_lt_of_le active (enough _))]
      cases output
      · exact outside .parameter i active
      · exact outside .momentum i active
    · exact exited.2.2.1.trans (body.kernels.streams.trans ((congrArg World.currentStreams state).trans entered.2.2.2.1))

#axiomsClean run_path
#axiomsClean run_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Host

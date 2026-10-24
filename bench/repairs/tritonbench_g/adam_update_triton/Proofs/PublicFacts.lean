import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Pipeline
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.InvocationSafety
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Observation
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
noncomputable section

/-- Same successful direct-view path, with integer/address certificates on
all actual autotune trials and selected invocations, including their queues. -/
def Execution {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Scalars α → Machine.Model α τ)
    (rt : Runtime M) (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (call : Call) (args : Scalars (BitVec 64)) (params : Scalars α) (before after : World M)
    (extra : Buffers → Queue → Index.Chunk → Config → M → Prop) : Prop :=
  (call.request.count=0 ∧ after=before) ∨
  ∃ context : ContextPath rt call.request.p.storage.owner
      (pipeline rt driver convert args call.request) before () after,
    ∃ body : PipelinePath ops (model params) rt call.request context.entered context.completed,
      Prepared call.request body.buffers ∧
      context.entered.currentDevice=call.request.p.storage.owner ∧
      (∀ role,((body.buffers.layout call.request).allocation role).owner=call.request.p.storage.owner) ∧
      InvocationSafety.Launches ops (model params) (body.buffers.layout call.request)
        (extra body.buffers (context.entered.queue call.request.p.storage.owner))
        (CallerBytes call.request) body.kernels.source

theorem execution {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (model : Scalars α → Machine.Model α τ)
    (rt : Runtime M) (lib : Library rt) (driver : ChunkRuntime M α τ ops model rt)
    (convert : Scalar.Conversions α) (call : Call) (h : Admitted call) (tensor : Bindings call)
    (args : Scalars (BitVec 64)) (params : Scalars α)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params)
    (before after : World M) (ran : run rt driver convert call before=.ok () after)
    (extra : Buffers → Queue → Index.Chunk → Config → M → Prop)
    (bound : ∀ b queue,∀ chunk∈Index.chunks 0 call.request.count,∀ config state,extra b queue chunk config state) :
    Execution ops model rt driver convert call args params before after extra := by
  have path:=run_path rt driver convert call h args params normalized binding before after ran
  cases path with
  | empty zero same => exact Or.inl ⟨zero,same⟩
  | nonempty nonzero context body =>
    have prepared:=(prepare_sound rt lib call.request context.entered body.prepared body.buffers body.preparation).1
    have valid:=prepared.valid call h tensor body.buffers (Nat.pos_of_ne_zero nonzero)
    refine Or.inr ⟨context,body,prepared,(lib.enter _ _ _ _ context.enter).2.2.1,?_,?_⟩
    · intro role
      rw [(prepared.window role).owner]
      cases role
      · rfl
      · change call.request.momentum.storage.owner=call.request.p.storage.owner
        rw [←tensor.momentum.device,←tensor.parameter.device,h.sameDevice.2]
      · change call.request.grad.storage.owner=call.request.p.storage.owner
        rw [←tensor.gradient.device,←tensor.parameter.device,h.sameDevice.1]
    · exact InvocationSafety.launches ops (model params) (body.buffers.layout call.request) valid
        _ (bound body.buffers _) _ _ (fun _ member=>member) _ _ body.kernels.source

theorem read_only {M α β τ : Type} [Inhabited α] [Inhabited τ]
    (ops : Ops M α β) (neg : Negation α) (readBytes : Observation.ReadByBytes ops)
    (model : Scalars α → Machine.Model α τ) (rt : Runtime M) (lib : Library rt)
    (driver : ChunkRuntime M α τ ops model rt) (convert : Scalar.Conversions α)
    (call : Call) (h : Admitted call) (tensor : Bindings call)
    (args : Scalars (BitVec 64)) (params : Scalars α)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params)
    (before after : World M) (ran : run rt driver convert call before=.ok () after)
    (whole : WholeSpec ops neg (model params) call.request before after)
    (role : Role) (i : Nat) (active : i<(call.request.view role).numel)
    (readonly : role=.third ∨ call.request.count≤ i) :
    logical ops neg after.memory (call.request.view role) i=
      logical ops neg before.memory (call.request.view role) i := by
  have path:=run_path rt driver convert call h args params normalized binding before after ran
  cases path with
  | empty zero same => rw [same]
  | nonempty nonzero context body =>
    have prepared:=(prepare_sound rt lib call.request context.entered body.prepared body.buffers body.preparation).1
    have valid:=prepared.valid call h tensor body.buffers (Nat.pos_of_ne_zero nonzero)
    have enough (r : Role) : call.request.count≤(call.request.view r).numel := by
      have cp:=tensor.count .a
      have cg:=tensor.count .third
      have cm:=tensor.count .b
      change call.parameter.numel=call.request.count at cp
      change call.gradient.numel=call.request.grad.numel at cg
      change call.momentum.numel=call.request.momentum.numel at cm
      cases r
      · exact Nat.le_refl _
      · have := h.enough.2;change call.request.count≤call.request.momentum.numel;omega
      · have := h.enough.1;change call.request.count≤call.request.grad.numel;omega
    have contained (r : Role) (j : Nat) (hj : j<(call.request.view r).numel) :
        ((body.buffers.layout call.request).allocation r).contains
          (((body.buffers.layout call.request).allocation r).key j) := by
      apply Allocation.key_contains
      simpa [Allocation.words,(prepared.window r).bytes] using hj
    unfold logical
    congr 1
    apply readBytes
    intro byte
    have backed:=(tensor.tensor role).backed i active
    have inside : (call.request.view role).storage.covers ((call.request.view role).key i).owner
        (((call.request.view role).key i).address+byte.val) := by
      unfold Allocation.contains at backed
      unfold Allocation.covers
      exact ⟨backed.1,by omega,by have := byte.isLt;omega⟩
    apply whole.frame _ _ ⟨(call.request.view role).storage,by cases role <;> simp [Request.view,Request.caller],inside⟩
    intro output j selected
    have separate :
        (((body.buffers.layout call.request).allocation role).key i).disjoint
          (((body.buffers.layout call.request).allocation (Private.outputRole output)).key j) := by
      by_cases same : role=Private.outputRole output
      · rw [←same]
        apply Allocation.indices_disjoint
        rcases readonly with gradient | surplus
        · cases output <;> simp_all [Private.outputRole]
        · omega
      · exact Allocation.spans_separate _ _ (valid.separate role _ same) _ _
          (contained role i active) (contained _ j (lt_of_lt_of_le selected (enough _)))
    rw [prepared.keys _ _ role i active,
      prepared.keys _ _ (Private.outputRole output) j (lt_of_lt_of_le selected (enough _))] at separate
    have byteInside : ((call.request.view role).key i).covers ((call.request.view role).key i).owner
        (((call.request.view role).key i).address+byte.val) := ⟨rfl,by omega,by have := byte.isLt;omega⟩
    have disjoint:=Key.disjoint_bytes separate _ _ byteInside
    cases output <;> exact disjoint

#axiomsClean execution
#axiomsClean read_only
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Host

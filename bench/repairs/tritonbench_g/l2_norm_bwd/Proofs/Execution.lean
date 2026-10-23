import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Whole

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Actual handles, intermediate states, setup queue and internally executed
memory transition, all belonging to the same body invocation. -/
structure BodyPath {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (q : Request) (eps : α) (before after : World M) where
  x : Matrix
  dy : Matrix
  afterX : World M
  launch : World M
  resolvedX : rt.perform (.resolve q.x) before=.ok x afterX
  resolvedDY : rt.perform (.resolve q.dy) afterX=.ok dy launch
  setup : rt.perform (.setup ⟨x,dy,q.dx,q.block⟩ launch.queue) launch=.ok () launch
  computed : after={launch with memory:=Physical.run ops red ⟨x,dy,q.dx,q.block⟩ eps (rt.schedule ⟨x,dy,q.dx,q.block⟩ launch) launch.memory}
  valid : Valid ⟨x,dy,q.dx,q.block⟩
  xMetadata : Integer.Metadata x
  dyMetadata : Integer.Metadata dy
  dxMetadata : Integer.Metadata q.dx
  rows : x.rows=q.x.rows
  cols : x.cols=q.x.cols
  blockBound : q.block≤16384

def BodyPath.layout {M α β : Type} {ops : Ops M α β} {red : Reduction α} {rt : Runtime M}
    {q : Request} {eps : α} {before after : World M} (p : BodyPath ops red rt q eps before after) : Layout :=
  ⟨p.x,p.dy,q.dx,q.block⟩

private theorem resolved_fresh {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {v out : View} {before after : World M} (copy : Copy ops neg v out before after)
    (a : Allocation) (live : before.live a) (separate : v.storage.separate a) : out.storage.separate a := by
  rcases copy.storage with same | fresh
  · rwa [same]
  · exact fresh a live

theorem body_path {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (initial before after : World M)
    (q : Request) (eps : α) (norm : Normalized ops neg call initial q before)
    (success : body ops red rt q eps before=.ok () after) :
    Nonempty (BodyPath ops red rt q eps before after) := by
  unfold body at success
  cases h1 : rt.perform (.resolve q.x) before with
  | error why failed => simp [h1,Result.bind] at success
  | ok x s1 =>
    rw [h1] at success
    simp only [Result.bind] at success
    obtain ⟨xr,xc,xneg,cx,mx⟩ := lib.resolve _ _ _ _ h1
    have liveDX : s1.live q.dx.view.storage := by
      have preserved := (lib.frame (.resolve q.x) before [] (by simp)).live _ norm.dxLive
      simpa only [h1,Result.state] using preserved
    cases h2 : rt.perform (.resolve q.dy) s1 with
    | error why failed => simp [h2,Result.bind] at success
    | ok dy s2 =>
      rw [h2] at success
      simp only [Result.bind] at success
      obtain ⟨yr,yc,yneg,cy,my⟩ := lib.resolve _ _ _ _ h2
      let p : Layout := ⟨x,dy,q.dx,q.block⟩
      have valid : Valid p := {
        positive := by dsimp [p];rw [xc];exact norm.positive
        block := by dsimp [p];rw [xc];exact norm.block
        dyCols := by dsimp [p];rw [yc,xc];exact norm.dyCols
        dyRows := by dsimp [p];rw [yr,xr];exact norm.dyRows
        dxCols := by dsimp [p];rw [xc];exact norm.dxCols
        dxRows := by dsimp [p];rw [xr];exact norm.dxRows
        xBacked := x.from_backed cx.backed
        dyBacked := dy.from_backed cy.backed
        dxBacked := q.dx.from_backed norm.dxBacked
        dxUnique := norm.dxUnique
        xFresh := resolved_fresh cx q.dx.view.storage norm.dxLive norm.xFresh
        dyFresh := resolved_fresh cy q.dx.view.storage liveDX norm.dyFresh
        xResolved := xneg
        dyResolved := yneg
        dxResolved := norm.dxPositive }
      cases h3 : rt.perform (.setup p s2.queue) s2 with
      | error why failed => simp [p,h3,Result.bind] at success
      | ok u s3 =>
        cases u
        have same := lib.setup _ _ _ _ h3
        subst s3
        change (rt.perform (.setup p s2.queue) s2).bind _=_ at success
        rw [h3] at success
        simp only [Result.bind,Result.ok.injEq,true_and] at success
        exact ⟨⟨x,dy,s1,s2,h1,h2,h3,success.symm,valid,mx,my,norm.dxMetadata,xr,xc,norm.blockBound⟩⟩

structure ContextPath {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (q : Request) (eps : α) (before after : World M) where
  token : Context
  inside : World M
  bodyAfter : World M
  entered : rt.perform (.enter q.x.view.storage.owner) before=.ok token inside
  body : BodyPath ops red rt q eps inside bodyAfter
  exited : rt.perform (.exit token) bodyAfter=.ok () after

theorem context_path {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (initial before after : World M)
    (q : Request) (eps : α) (norm : Normalized ops neg call initial q before)
    (domain : InputDomain call initial)
    (success : contextual ops red rt q eps before=.ok () after) :
    Nonempty (ContextPath ops red rt q eps before after) := by
  unfold contextual at success
  cases he : rt.perform (.enter q.x.view.storage.owner) before with
  | error why failed => simp [he,Result.bind] at success
  | ok token inside =>
    rw [he] at success
    simp only [Result.bind] at success
    have fe := lib.frame (.enter q.x.view.storage.owner) before call.caller
      (fun a ha => norm.caller.live a (domain.live a ha))
    simp only [he,Result.state] at fe
    have memory := (lib.enter _ _ _ _ he).2.1
    have insideNorm := normalized_memory ops neg call initial before inside q norm memory fe
    cases hb : body ops red rt q eps inside with
    | error why failed =>
      rw [hb] at success
      unfold finallyExit at success
      cases hx : rt.perform (.exit token) failed <;> simp [hx,Result.bind] at success
    | ok u computed =>
      cases u
      rw [hb] at success
      change rt.perform (.exit token) computed=.ok () after at success
      obtain ⟨bp⟩ := body_path ops neg red rt lib call initial inside computed q eps insideNorm hb
      exact ⟨⟨token,inside,computed,he,bp,success⟩⟩

/-- One complete successful call, including the actual final returned object.
Reshape is permitted to allocate its actual copy. -/
structure WholePath {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (call : Call) (eps : α) (initial : World M) (output : View) (after : World M) where
  request : Request
  prepared : World M
  computed : World M
  normalized : normalize rt call initial=.ok request prepared
  context : ContextPath ops red rt request eps prepared computed
  returned : rt.perform (.returnShape request.dx request.originalShape) computed=.ok output after

theorem successful_path {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (eps : α) (initial after : World M)
    (output : View) (domain : InputDomain call initial)
    (success : run ops red rt call eps initial=.ok output after) :
    Nonempty (WholePath ops red rt call eps initial output after) := by
  have spec := whole_spec ops neg red rt lib call eps initial domain
  rw [success] at spec
  change WholeSpec ops neg red rt call eps initial output after at spec
  obtain ⟨path⟩ := spec.path
  have norm := normalize_spec ops neg rt lib call initial domain
  rw [path.normalize] at norm
  change Normalized ops neg call initial path.request path.prepared at norm
  obtain ⟨context⟩ := context_path ops neg red rt lib call initial path.prepared path.computed
    path.request eps norm domain path.context
  exact ⟨⟨path.request,path.prepared,path.computed,path.normalize,context,path.returned⟩⟩

#axiomsClean body_path
#axiomsClean context_path
#axiomsClean successful_path
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host

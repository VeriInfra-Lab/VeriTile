import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Host

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate
open Memory Native Host
noncomputable section
set_option maxHeartbeats 4000000

/-- Forward has no incoming cotangent. The unused field is fixed independently
of all caller memory, rather than supplied as a future backward input. -/
def primalInput (c : Shape) (s : Memory Word) : Native.Input Word :=
  ⟨Native.load c s .q,Native.load c s .k,Native.load c s .v,fun _ _ _ => 0⟩

theorem forward_reference_primal (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (s : Memory Word) (scale : Word) :
    forwardReference c p owners (Native.input c s) scale =
      forwardReference c p owners (primalInput c s) scale := by rfl

def attachGradient (gradient : Nat → Word) (s : Memory Word) : Memory Word :=
  fun b o => if b = .g then gradient o else s b o

theorem primal_no_future_gradient (c : Shape) (gradient : Nat → Word) (s : Memory Word) :
    primalInput c (attachGradient gradient s) = primalInput c s := by rfl

def forwardWritten (c : Shape) (r : Buffer) (o : Nat) : Prop :=
  (r = .o ∧ o < c.elements) ∨ ((r = .l ∨ r = .m) ∧ o < c.groups*c.rows)

theorem forward_frame (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (s : Memory Word) (scale : Word) (r : Buffer) (o : Nat)
    (outside : ¬forwardWritten c r o) : Native.forward c p owners s scale r o = s r o := by
  have tensorFrame (s' : Memory Word) (values : Coord c → Word) :
      writeTensor c s' .o values r o = s' r o := by
    by_cases hr : r = .o
    · apply writeTensor_outside
      by_contra! h
      exact outside (Or.inl ⟨hr,h⟩)
    · exact writeTensor_other _ _ _ _ _ _ hr
  have rowFrame (s' : Memory Word) (b : Buffer)
      (values : Fin c.groups × Fin c.rows → Word) (hb : b = .l ∨ b = .m) :
      writeRows c s' b values r o = s' r o := by
    by_cases hr : r = b
    · apply writeRows_outside
      by_contra! h
      exact outside (Or.inr ⟨hr ▸ hb,h⟩)
    · exact writeRows_other _ _ _ _ _ _ hr
  simp only [Native.forward]
  rw [rowFrame _ _ _ (by simp),rowFrame _ _ _ (by simp),tensorFrame]

def ForwardCorrect (c : Shape) (reference : Instructions) (owners : Owners)
    (before after : Memory Word) (scale : Word) : Prop :=
  (∀ i : Coord c, after .o (offset c i) =
    (forwardReference c (Native.ops reference) owners (primalInput c before) scale).y
      i.1.val i.2.1.val i.2.2.val) ∧
  (∀ i : Fin c.groups × Fin c.rows,
    after .l (rowAddr c i) =
      (forwardReference c (Native.ops reference) owners (primalInput c before) scale).l i.1.val i.2.val ∧
    after .m (rowAddr c i) =
      (forwardReference c (Native.ops reference) owners (primalInput c before) scale).m i.1.val i.2.val) ∧
  ∀ r o, ¬forwardWritten c r o → after r o = before r o

theorem forward_correct (c : Shape) (p : Instructions) (owners : Owners)
    (s : Memory Word) (scale : Word) :
    ForwardCorrect c p owners s (Native.forward c (Native.ops p) owners s scale) scale := by
  refine ⟨?_,?_,forward_frame c (Native.ops p) owners s scale⟩
  · intro i
    simp only [Native.forward,forward_reference_primal]
    rw [writeRows_other _ _ _ _ _ _ (by decide : Buffer.o ≠ .m),
      writeRows_other _ _ _ _ _ _ (by decide : Buffer.o ≠ .l),writeTensor_readback]
  · intro i
    constructor
    · simp only [Native.forward,forward_reference_primal]
      rw [writeRows_other _ _ _ _ _ _ (by decide : Buffer.l ≠ .m),writeRows_readback]
    · simp only [Native.forward,forward_reference_primal]
      rw [writeRows_readback]

def savedContext (c : Shape) (a : Host.Arguments) : Host.Context :=
  ⟨a.q,a.k,a.v,Host.tensor c a.q.device .o,Host.rowTensor c a.q.device .l,
    Host.rowTensor c a.q.device .m,a.scale,Host.forwardLaunch c⟩

structure ForwardResult where
  output : Host.Tensor
  context : Host.Context
  memory : Memory Word

def forwardResult (c : Shape) (a : Host.Arguments) (s : Memory Word) : ForwardResult :=
  ⟨Host.tensor c a.q.device .o,savedContext c a,s⟩

def forwardCall (c : Shape) (a : Host.Arguments) (runtime : Host.Runtime)
    (s : Memory Word) : Option ForwardResult := by
  classical
  exact if Host.InputChecks c a then do
    let f ← runtime.launch (Host.forwardLaunch c) a.scale s
    pure (forwardResult c a f)
  else none

/-- Ordinary binding of this one launch to its actual instruction graph. -/
structure ForwardBinding (c : Shape) (actual : Instructions) (owners : Owners)
    (runtime : Host.Runtime) : Prop where
  launch : ∀ scale s, runtime.launch (Host.forwardLaunch c) scale s =
    some (Native.forward c (Native.ops actual) owners s scale)

/-- A separate forward endpoint with output/saved-tensor values, exact forward
frame and returned context, requiring no incoming gradient or backward call. -/
theorem forward_host (c : Shape) (a : Host.Arguments) (runtime : Host.Runtime)
    (s : Memory Word) (actual reference : Instructions) (owners : Owners)
    (checks : Host.InputChecks c a) (binding : ForwardBinding c actual owners runtime)
    (numeric : LocalConformance actual reference) :
    ∃ r, forwardCall c a runtime s = some r ∧
      r.output = Host.tensor c a.q.device .o ∧ r.context = savedContext c a ∧
      ForwardCorrect c reference owners s r.memory a.scale := by
  refine ⟨forwardResult c a (Native.forward c (Native.ops actual) owners s a.scale),?_,rfl,rfl,?_⟩
  · simp [forwardCall,checks,binding.launch]
  · rw [local_conformance_eq actual reference numeric]
    exact forward_correct c reference owners s a.scale

/-- Later backward may have arbitrary new scratch contents. Only saved
primal/output/normalizer/maximum logical views are bound to the saved snapshot. -/
structure SavedMatch (c : Shape) (snapshot current : Memory Word) : Prop where
  q : Native.load c current .q = Native.load c snapshot .q
  k : Native.load c current .k = Native.load c snapshot .k
  v : Native.load c current .v = Native.load c snapshot .v
  o : Native.load c current .o = Native.load c snapshot .o
  l : Native.loadRow c current .l = Native.loadRow c snapshot .l
  m : Native.loadRow c current .m = Native.loadRow c snapshot .m

def gradientView (c : Shape) (gradient : Nat → Word) : Native.Tensor Word :=
  fun h i e => gradient (offset c (Native.coord c h i e))

/-- Input-only backward graph from saved forward tensors and the cotangent
supplied to this call, independent of current scratch or execution results. -/
def backwardSnapshot (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (snapshot : Memory Word) (gradient : Nat → Word) : BackwardInput Word :=
  let pre := preprocessReference c p owners (gradientView c gradient) (Native.saved c snapshot)
  ⟨Native.load c snapshot .q,Native.load c snapshot .k,Native.load c snapshot .v,
    Native.loadRow c snapshot .m,pre.newDO,pre.delta⟩

/-- A snapshot actually produced by forward composes with a cotangent chosen
later. The resulting reference uses original Q/K/V and that new cotangent. -/
theorem backwardSnapshot_forward (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (initial : Memory Word) (gradient : Nat → Word) (scale : Word) :
    backwardSnapshot c p owners (Native.forward c p owners initial scale) gradient =
      wholeBackwardInput c p owners (Native.input c (attachGradient gradient initial)) scale := by
  have hf : forwardReference c p owners (Native.input c (attachGradient gradient initial)) scale =
      forwardReference c p owners (Native.input c initial) scale := by
    rw [forward_reference_primal,primal_no_future_gradient,←forward_reference_primal]
  simp only [backwardSnapshot,wholeBackwardInput,saved_forward,hf]
  simp [Native.forward,Native.input]
  refine ⟨rfl,rfl,rfl,?_,rfl,rfl⟩
  funext h i
  simp [forwardReference,Native.rowCoord]

theorem preprocess_attached (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (snapshot current : Memory Word) (gradient : Nat → Word)
    (saved : SavedMatch c snapshot current) :
    backwardInput c (Native.preprocess c p owners (attachGradient gradient current)) =
      backwardSnapshot c p owners snapshot gradient := by
  have hsaved : Native.saved c (attachGradient gradient current) = Native.saved c snapshot := by
    change Native.Saved.mk (Native.load c current .o) (Native.loadRow c current .l)
      (Native.loadRow c current .m) = Native.saved c snapshot
    rw [saved.o,saved.l,saved.m]
    rfl
  have hq : Native.load c (attachGradient gradient current) .q = Native.load c snapshot .q := saved.q
  have hk : Native.load c (attachGradient gradient current) .k = Native.load c snapshot .k := saved.k
  have hv : Native.load c (attachGradient gradient current) .v = Native.load c snapshot .v := saved.v
  have hm : Native.loadRow c (attachGradient gradient current) .m = Native.loadRow c snapshot .m := saved.m
  have hg : Native.load c (attachGradient gradient current) .g = gradientView c gradient := rfl
  simp [backwardInput,Native.preprocess,backwardSnapshot,hsaved,hq,hk,hv,hm,hg,
    preprocessReference,Native.coord,Native.rowCoord]

def backwardWritten (c : Shape) (r : Buffer) (o : Nat) : Prop :=
  ((r = .newDO ∨ r = .dq ∨ r = .dk ∨ r = .dv) ∧ o < c.elements) ∨
  (r = .delta ∧ o < c.groups*c.rows)

def BackwardCorrect (c : Shape) (reference : Instructions) (owners : Owners)
    (snapshot before after : Memory Word) (gradient : Nat → Word) (scale : Word) : Prop :=
  (∀ i : Coord c,
    after .dq (offset c i) = dqReference c (Native.ops reference)
      (backwardSnapshot c (Native.ops reference) owners snapshot gradient) scale c.blocks
      i.1.val i.2.1.val i.2.2.val ∧
    after .dk (offset c i) = dkReference c (Native.ops reference)
      (backwardSnapshot c (Native.ops reference) owners snapshot gradient) scale
      i.1.val i.2.1.val i.2.2.val ∧
    after .dv (offset c i) = dvReference c (Native.ops reference)
      (backwardSnapshot c (Native.ops reference) owners snapshot gradient) scale
      i.1.val i.2.1.val i.2.2.val) ∧
  ∀ r o, ¬backwardWritten c r o → after r o = (attachGradient gradient before) r o

def backwardRun (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (current : Memory Word) (gradient : Nat → Word) (scale : Word) : Memory Word :=
  backwardFrom c p scale (Native.preprocess c p owners (zeroDQ c p (attachGradient gradient current))) c.blocks

theorem backward_run_prefix (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (current : Memory Word) (gradient : Nat → Word) (scale : Word) :
    backwardRun c p owners current gradient scale =
      backwardPrefix c p scale (Native.preprocess c p owners (attachGradient gradient current)) c.blocks := by
  rw [backwardRun,preprocess_zeroDQ,←backwardPrefix_eq_from]

theorem backward_frame (c : Shape) (p : Native.Ops Word) (owners : Owners)
    (current : Memory Word) (gradient : Nat → Word) (scale : Word)
    (r : Buffer) (o : Nat) (outside : ¬backwardWritten c r o) :
    backwardRun c p owners current gradient scale r o = (attachGradient gradient current) r o := by
  have tensorFrame (s : Memory Word) (b : Buffer) (values : Coord c → Word)
      (hb : b = .newDO ∨ b = .dq ∨ b = .dk ∨ b = .dv) :
      writeTensor c s b values r o = s r o := by
    by_cases hr : r = b
    · apply writeTensor_outside
      by_contra! h
      exact outside (Or.inl ⟨hr ▸ hb,h⟩)
    · exact writeTensor_other _ _ _ _ _ _ hr
  have whereFrame (s : Memory Word) (b : Buffer) (keep : Coord c → Prop)
      (values : Coord c → Word) (hb : b = .dq ∨ b = .dk ∨ b = .dv) :
      writeTensorWhere c s b keep values r o = s r o := by
    by_cases hr : r = b
    · apply writeTensorWhere_outside
      by_contra! h
      exact outside (Or.inl ⟨Or.inr (hr ▸ hb),h⟩)
    · exact writeTensorWhere_other _ _ _ _ _ _ _ hr
  have prefixFrame (s : Memory Word) (n : Nat) : backwardPrefix c p scale s n r o = s r o := by
    induction n with
    | zero => exact tensorFrame s .dq _ (by simp)
    | succ n ih =>
      simp only [backwardPrefix,backwardStep]
      rw [whereFrame _ _ _ _ (by simp),whereFrame _ _ _ _ (by simp),
        whereFrame _ _ _ _ (by simp),ih]
  rw [backward_run_prefix,prefixFrame]
  simp only [Native.preprocess]
  have rowFrame (s : Memory Word) (values : Fin c.groups × Fin c.rows → Word) :
      writeRows c s .delta values r o = s r o := by
    by_cases hr : r = .delta
    · apply writeRows_outside
      by_contra! h
      exact outside (Or.inr ⟨hr,h⟩)
    · exact writeRows_other _ _ _ _ _ _ hr
  rw [rowFrame,tensorFrame _ _ _ (by simp)]

theorem backward_correct (c : Shape) (p : Instructions) (owners : Owners)
    (snapshot current : Memory Word) (gradient : Nat → Word) (scale : Word)
    (saved : SavedMatch c snapshot current) :
    BackwardCorrect c p owners snapshot current
      (backwardRun c (Native.ops p) owners current gradient scale) gradient scale := by
  refine ⟨?_,backward_frame c (Native.ops p) owners current gradient scale⟩
  intro i
  rw [backward_run_prefix]
  rw [backwardPrefix_dq,backwardPrefix_dk _ _ _ _ _ _ (grid_coverage c i.2.1).1,
    backwardPrefix_dv _ _ _ _ _ _ (grid_coverage c i.2.1).1,
    preprocess_attached c (Native.ops p) owners snapshot current gradient saved]
  exact ⟨rfl,rfl,rfl⟩

structure BackwardResult where
  dq : Host.Tensor
  dk : Host.Tensor
  dv : Host.Tensor
  scaleGradient : Option Host.Tensor
  memory : Memory Word

def backwardResult (c : Shape) (ctx : Host.Context) (s : Memory Word) : BackwardResult :=
  ⟨Host.tensor c ctx.q.device .dq,Host.tensor c ctx.q.device .dk,
    Host.tensor c ctx.q.device .dv,none,s⟩

def contextArguments (ctx : Host.Context) : Host.Arguments := ⟨ctx.q,ctx.k,ctx.v,ctx.scale⟩

def backwardCall (c : Shape) (ctx : Host.Context) (g : Host.Tensor) (runtime : Host.Runtime)
    (current : Memory Word) (gradient : Nat → Word) : Option BackwardResult := by
  classical
  exact if Host.GradientChecks c (contextArguments ctx) g then do
    let gc ← runtime.contiguousGradient g (attachGradient gradient current)
    let z ← runtime.zerosDQ gc
    let pre ← runtime.launch (Host.preprocessLaunch c) ctx.scale z
    let final ← runtime.launch (Host.backwardLaunch c) ctx.scale pre
    pure (backwardResult c ctx final)
  else none

/-- Gradient words denote the actual logical cotangent after contiguous copy;
stage bindings preserve exact actual instructions and actual current memory. -/
structure BackwardBinding (c : Shape) (actual : Instructions) (owners : Owners)
    (runtime : Host.Runtime) (g : Host.Tensor) : Prop where
  contiguous : ∀ s, runtime.contiguousGradient g s = some s
  zeros : ∀ s, runtime.zerosDQ s = some (zeroDQ c (Native.ops actual) s)
  preprocess : ∀ scale s, runtime.launch (Host.preprocessLaunch c) scale s =
    some (Native.preprocess c (Native.ops actual) owners s)
  backward : ∀ scale s, runtime.launch (Host.backwardLaunch c) scale s =
    some (backwardFrom c (Native.ops actual) scale s c.blocks)

/-- A later backward call consumes its actual current cotangent and the saved
forward views. Its scratch state is unconstrained, its DQ is initialized before
preprocess, and its exact frame is relative to the attached cotangent memory. -/
theorem backward_host (c : Shape) (a : Host.Arguments) (ctx : Host.Context)
    (g : Host.Tensor) (runtime : Host.Runtime) (snapshot current : Memory Word)
    (gradient : Nat → Word) (actual reference : Instructions) (owners : Owners)
    (context : ctx = savedContext c a)
    (checks : Host.GradientChecks c (contextArguments ctx) g)
    (saved : SavedMatch c snapshot current)
    (binding : BackwardBinding c actual owners runtime g)
    (numeric : LocalConformance actual reference) :
    ∃ r, backwardCall c ctx g runtime current gradient = some r ∧
      r.dq = Host.tensor c a.q.device .dq ∧ r.dk = Host.tensor c a.q.device .dk ∧
      r.dv = Host.tensor c a.q.device .dv ∧ r.scaleGradient = none ∧
      BackwardCorrect c reference owners snapshot current r.memory gradient a.scale := by
  subst ctx
  refine ⟨backwardResult c (savedContext c a)
    (backwardRun c (Native.ops actual) owners current gradient a.scale),?_,rfl,rfl,rfl,rfl,?_⟩
  · rw [backwardCall,if_pos checks]
    simp [binding.contiguous,binding.zeros,binding.preprocess,
      binding.backward,backwardRun,savedContext]
  · rw [local_conformance_eq actual reference numeric]
    exact backward_correct c reference owners snapshot current gradient a.scale saved

#axiomsClean primal_no_future_gradient
#axiomsClean forward_host
#axiomsClean preprocess_attached
#axiomsClean backwardSnapshot_forward
#axiomsClean backward_host
#specNonCircular ForwardCorrect avoiding [forwardCall,forward_host]
#specNonCircular BackwardCorrect avoiding [backwardRun,backwardCall,backward_host]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate

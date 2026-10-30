import bench.tritonbench_g.kldiv_compute.Proofs.Program
import bench.tritonbench_g.cross_entropy1.Proofs.TensorModel

namespace VeriTile.Bench.Repairs.KldivCompute.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor World SameMetadata Preserves)
noncomputable section
set_option maxHeartbeats 6000000

/-- FP32 is carried by the original source's .real channel, without any Real
interpretation of α. Logical negation is the tensor-library operation. -/
def value {α : Type} [Inhabited α] (neg : α → α) (s : State α) (t : Tensor) (i : Nat) : α :=
  let raw := (s.mem t.region (t.index i)).read .real
  if t.isNeg then neg raw else raw

theorem preserved_value {α σ : Type} [Inhabited α] (neg : α → α)
    {a b : World α σ} (h : Preserves a b) (t : Tensor) (live : a.live t.region) (i : Nat) :
    value neg b.kernel t i = value neg a.kernel t i := by
  unfold value
  rw [h.frame t.region (t.index i) live]

inductive Step (S A : Type) where
  | error (detail : String) (state : S)
  | ok (value : A) (state : S)

def bind {S A B : Type} (r : Step S A) (f : A → S → Step S B) : Step S B :=
  match r with
  | .error e s => .error e s
  | .ok a s => f a s

def Ensures {S A : Type} (post : A → S → Prop) : Step S A → Prop
  | .error _ _ => True
  | .ok a s => post a s

theorem bind_ensures {S A B : Type} (r : Step S A) (f : A → S → Step S B)
    (p : A → S → Prop) (q : B → S → Prop) (first : Ensures p r)
    (rest : ∀ a s, p a s → Ensures q (f a s)) : Ensures q (bind r f) := by
  cases r with
  | error => trivial
  | ok a s => exact rest a s first

theorem self_ensures {S A : Type} (r : Step S A) : Ensures (fun a s => r=.ok a s) r := by
  cases r <;> trivial

/-- The first failing operation stops the continuation and retains its actual
failure state, including allocation/normalization/setup/copy side effects. -/
theorem failed_prefix {S A B : Type} (e : String) (s : S) (f : A → S → Step S B) :
    bind (.error e s) f = .error e s := rfl

structure Materialized {α σ : Type} [Inhabited α] (neg : α → α)
    (a b : Tensor) (s t : World α σ) : Prop where
  metadata : SameMetadata a b
  frame : Preserves s t
  live : t.live b.region
  values : ∀ i, i<a.numel → value neg t.kernel b i=value neg s.kernel a i

structure Allocation {α σ : Type} (a b : Tensor) (s t : World α σ) : Prop where
  metadata : SameMetadata a b
  frame : Preserves s t
  live : t.live b.region
  fresh : ¬s.live b.region
  objectFresh : b.objectId ≠ a.objectId
  notNeg : b.isNeg=Bool.false

structure Runtime (α σ : Type) where
  emptyLike : Tensor → World α σ → Step (World α σ) Tensor
  resolveNeg : Tensor → World α σ → Step (World α σ) Tensor
  contiguous : Tensor → World α σ → Step (World α σ) Tensor
  emptyWork : Tensor → World α σ → Step (World α σ) Tensor
  setup : Tensor → Tensor → Tensor → Nat → World α σ → Step (World α σ) Unit
  copy : Tensor → Tensor → World α σ → Step (World α σ) Tensor

/-- All trusted laws are ordinary local tensor-library operations. In
particular copy is a coordinate copy between arbitrary tensors; no law
mentions KLDIV, scalar arithmetic, launch success, or expected kernel values. -/
structure Library {α σ : Type} [Inhabited α] (neg : α → α) (rt : Runtime α σ) : Prop where
  empty : ∀ a s b t, s.live a.region → rt.emptyLike a s=.ok b t → Allocation a b s t
  resolve : ∀ a s b t, s.live a.region → rt.resolveNeg a s=.ok b t →
    Materialized neg a b s t ∧ b.isNeg=Bool.false
  pack : ∀ a s b t, s.live a.region → a.isNeg=Bool.false → rt.contiguous a s=.ok b t →
    Materialized neg a b s t ∧ b.Dense
  work : ∀ a s b t, s.live a.region → rt.emptyWork a s=.ok b t →
    Allocation a b s t ∧ b.Dense
  setup : ∀ a b o n s t, rt.setup a b o n s=.ok () t →
    t.kernel.mem=s.kernel.mem ∧ t.live=s.live
  copy : ∀ dst src s out t, s.live dst.region → s.live src.region →
    dst.shape=src.shape → rt.copy dst src s=.ok out t →
    out=dst ∧ t.live=s.live ∧
    (∀ i, i<dst.numel → value neg t.kernel dst i=value neg s.kernel src i) ∧
    (∀ r i, r≠dst.region → t.kernel.mem r i=s.kernel.mem r i) ∧
    (∀ i, (∀ j, j<dst.numel → i≠dst.index j) →
      t.kernel.mem dst.region i=s.kernel.mem dst.region i)

def prepare {α σ : Type} (rt : Runtime α σ) (a : Tensor) (s : World α σ) :=
  bind (rt.resolveNeg a s) fun r sr => rt.contiguous r sr

theorem prepare_correct {α σ : Type} [Inhabited α] (neg : α → α) (rt : Runtime α σ)
    (lib : Library neg rt) (a b : Tensor) (s t : World α σ) (live : s.live a.region)
    (h : prepare rt a s=.ok b t) : Materialized neg a b s t ∧ b.Dense := by
  cases hr : rt.resolveNeg a s with
  | error e u => simp [prepare,bind,hr] at h
  | ok r sr =>
    have cr := lib.resolve a s r sr live hr
    have hc : rt.contiguous r sr=.ok b t := by simpa [prepare,bind,hr] using h
    have cc := lib.pack r sr b t cr.1.live cr.2 hc
    refine ⟨⟨⟨cc.1.metadata.shape.trans cr.1.metadata.shape,cc.1.metadata.dtype.trans cr.1.metadata.dtype,
      cc.1.metadata.cuda.trans cr.1.metadata.cuda,cc.1.metadata.device.trans cr.1.metadata.device⟩,
      cr.1.frame.trans cc.1.frame,cc.1.live,?_⟩,cc.2⟩
    intro i hi
    exact (cc.1.values i (by simpa [cr.1.metadata.numel] using hi)).trans (cr.1.values i hi)

def params (x y o : Tensor) (n : Nat) : Parameters := ⟨x.region,y.region,o.region,n,1024⟩

def launch {α σ : Type} [Inhabited α] (M : Algebra α) (rt : Runtime α σ)
    (x y o : Tensor) (n : Nat) (s : World α σ) : Step (World α σ) Unit :=
  bind (rt.setup x y o n s) fun _ ready =>
    match Program.grid M (params x y o n) ready.kernel with
    | none => .error "source interpreter failure" ready
    | some final => .ok () {ready with kernel:=final}

/-- Exact Python order: original empty_like before CUDA checks, then shape and
device validation, two resolve/contiguous preparations, fresh packed work,
setup even for N=0, full source grid, coordinate copy_, original output return. -/
def run {α σ : Type} [Inhabited α] (M : Algebra α) (rt : Runtime α σ)
    (x y : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (rt.emptyLike x s) fun output allocated =>
    if x.isCUDA && y.isCUDA && output.isCUDA then
      if x.shape=y.shape then
        if x.device=y.device then
          bind (prepare rt x allocated) fun px sx =>
          bind (prepare rt y sx) fun py sy =>
          bind (rt.emptyWork px sy) fun work sw =>
          bind (launch M rt px py work output.numel sw) fun _ computed =>
          bind (rt.copy output work computed) fun _ final => .ok output final
        else .error "same CUDA device required" allocated
      else .error "equal shapes required" allocated
    else .error "CUDA assertion" allocated

structure LaunchData {α σ : Type} [Inhabited α] (M : Algebra α) (rt : Runtime α σ)
    (x y o : Tensor) (n : Nat) (before after : World α σ) where
  ready : World α σ
  final : State α
  setup_call : rt.setup x y o n before=.ok () ready
  source_call : Program.grid M (params x y o n) ready.kernel=some final
  same_state : after={ready with kernel:=final}
  memory : after.kernel.mem = fun r i =>
    if r=o.region ∧ i<n then
      .mk .real (Program.scalar M ((before.kernel.mem x.region i).read .real)
        ((before.kernel.mem y.region i).read .real)) else before.kernel.mem r i
  live : after.live=before.live

theorem launch_complete {α σ : Type} [Inhabited α] (neg : α → α) (M : Algebra α)
    (rt : Runtime α σ) (lib : Library neg rt) (x y o : Tensor) (n : Nat) (s : World α σ) :
    Ensures (fun _ after => Nonempty (LaunchData M rt x y o n s after))
      (launch M rt x y o n s) := by
  unfold launch
  apply bind_ensures _ _ _ _ (self_ensures _)
  intro u ready hs
  cases u
  obtain ⟨hm,hl⟩ := lib.setup _ _ _ _ _ _ hs
  obtain ⟨final,he,hf⟩ := Program.grid_complete M (params x y o n) ready.kernel (by change 0<1024; decide)
  rw [he]
  refine ⟨⟨ready,final,hs,he,rfl,?_,hl⟩⟩
  simpa only [params,Program.resultCell,
    VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory,hm] using hf

/-- Source execution cannot manufacture a failure after successful setup;
this statement also covers the empty grid. Every reported launch error is the
actual setup error with the exact same error-state world. -/
theorem launch_error_is_setup {α σ : Type} [Inhabited α] (M : Algebra α)
    (rt : Runtime α σ) (x y o : Tensor) (n : Nat) (s failed : World α σ) (reason : String)
    (h : launch M rt x y o n s=.error reason failed) :
    rt.setup x y o n s=.error reason failed := by
  cases hs : rt.setup x y o n s with
  | error error state =>
    simpa [launch,bind,hs] using h
  | ok unit ready =>
    cases unit
    obtain ⟨last,he,hm⟩ := Program.grid_complete M (params x y o n) ready.kernel
      (by change 0<1024; decide)
    simp [launch,bind,hs,he] at h

theorem allocation_failure {α σ : Type} [Inhabited α] (M : Algebra α)
    (rt : Runtime α σ) (x y : Tensor) (s failed : World α σ) (reason : String)
    (h : rt.emptyLike x s=.error reason failed) :
    run M rt x y s=.error reason failed := by simp [run,bind,h]

structure ReturnedData {α σ : Type} [Inhabited α] (neg : α → α) (M : Algebra α)
    (rt : Runtime α σ) (x y : Tensor) (initial : World α σ) (output : Tensor) (final : World α σ) where
  (px py work : Tensor)
  (allocated sx sy sw computed : World α σ)
  allocation_call : rt.emptyLike x initial=.ok output allocated
  prepare_x_call : prepare rt x allocated=.ok px sx
  prepare_y_call : prepare rt y sx=.ok py sy
  work_call : rt.emptyWork px sy=.ok work sw
  launch_data : LaunchData M rt px py work output.numel sw computed
  copy_call : rt.copy output work computed=.ok output final
  input_shape : x.shape=y.shape
  metadata : SameMetadata x output
  fresh : ¬initial.live output.region
  output_not_neg : output.isNeg=Bool.false
  caller_live : ∀ r, initial.live r → final.live r
  output_live : final.live output.region
  contents : ∀ i, i<x.numel →
    value neg final.kernel output i=Program.scalar M
      (value neg initial.kernel x i) (value neg initial.kernel y i)
  caller_frame : ∀ r i, initial.live r → final.kernel.mem r i=initial.kernel.mem r i
  output_padding : ∀ i, (∀ j, j<output.numel → i≠output.index j) →
    final.kernel.mem output.region i=allocated.kernel.mem output.region i

/-- Every successful actual caller execution yields the same allocated
output descriptor and the values stored by that call's source grid/copyback.
No prepared-buffer, successful-source or whole-output premise is supplied. -/
theorem whole {α σ : Type} [Inhabited α] (neg : α → α) (M : Algebra α)
    (rt : Runtime α σ) (lib : Library neg rt) (x y : Tensor) (initial : World α σ)
    (hx : initial.live x.region) (hy : initial.live y.region) :
    Ensures (fun output final => Nonempty (ReturnedData neg M rt x y initial output final))
      (run M rt x y initial) := by
  unfold run
  apply bind_ensures _ _ _ _ (self_ensures _)
  intro output allocated ha
  have ca := lib.empty _ _ _ _ hx ha
  split
  next hcuda =>
    split
    next hshape =>
      split
      next hdevice =>
        apply bind_ensures _ _ _ _ (self_ensures _)
        intro px sx hpx
        obtain ⟨cx,dx⟩ := prepare_correct neg rt lib x px allocated sx (ca.frame.live _ hx) hpx
        apply bind_ensures _ _ _ _ (self_ensures _)
        intro py sy hpy
        obtain ⟨cy,dy⟩ := prepare_correct neg rt lib y py sx sy
          (cx.frame.live _ (ca.frame.live _ hy)) hpy
        apply bind_ensures _ _ _ _ (self_ensures _)
        intro work sw hw
        obtain ⟨cw,dw⟩ := lib.work _ _ _ _ (cy.frame.live _ cx.live) hw
        apply bind_ensures _ _ _ _ (launch_complete neg M rt lib px py work output.numel sw)
        intro ignored computed he
        obtain ⟨le⟩ := he
        have before : Preserves initial sw := ((ca.frame.trans cx.frame).trans cy.frame).trans cw.frame
        have output_live : sw.live output.region := cw.frame.live _ (cy.frame.live _ (cx.frame.live _ ca.live))
        have work_ne_output : work.region≠output.region := by
          intro eq
          exact cw.fresh (eq ▸ cy.frame.live _ (cx.frame.live _ ca.live))
        have shape_work : work.shape=x.shape := cw.metadata.shape.trans cx.metadata.shape
        have shape_copy : output.shape=work.shape := ca.metadata.shape.trans shape_work.symm
        apply bind_ensures _ _ _ _ (self_ensures _)
        intro copied final hc
        obtain ⟨hout,hl,hvals,hframe,hpadding⟩ := lib.copy output work computed copied final
          (le.live.symm ▸ output_live) (le.live.symm ▸ cw.live) shape_copy hc
        subst copied
        refine ⟨⟨px,py,work,allocated,sx,sy,sw,computed,ha,hpx,hpy,hw,le,hc,
          hshape,ca.metadata,ca.fresh,ca.notNeg,
          (fun r hr => hl.symm ▸ le.live.symm ▸ before.live r hr),
          (hl.symm ▸ le.live.symm ▸ output_live),?_,?_,?_⟩⟩
        · intro i hi
          have hio : i<output.numel := by simpa [ca.metadata.numel] using hi
          have hiw : i<work.numel := by simpa [Tensor.numel,shape_work] using hi
          have hix : i<px.numel := by simpa [cx.metadata.numel] using hi
          have hiy : i<y.numel := by simpa [Tensor.numel,hshape] using hi
          have hipy : i<py.numel := by simpa [cy.metadata.numel] using hiy
          have vx : (sw.kernel.mem px.region i).read .real=value neg initial.kernel x i := by
            have h := preserved_value neg (cy.frame.trans cw.frame) px cx.live i
            have c := (cx.values i hi).trans (preserved_value neg ca.frame x hx i)
            simpa [value,dx.1,dx.2 i hix] using h.trans c
          have vy : (sw.kernel.mem py.region i).read .real=value neg initial.kernel y i := by
            have h := preserved_value neg cw.frame py cy.live i
            have c := (cy.values i hiy).trans
              (preserved_value neg (ca.frame.trans cx.frame) y hy i)
            simpa [value,dy.1,dy.2 i hipy] using h.trans c
          rw [hvals i hio]
          simp only [value,dw.1,Bool.false_eq_true,↓reduceIte,dw.2 i hiw,le.memory,
            hio,and_self,↓reduceIte,Cell.read_same,vx,vy]
        · intro r i hr
          have no : r≠output.region := by intro eq; exact ca.fresh (eq ▸ hr)
          have nw : r≠work.region := by
            intro eq
            exact cw.fresh (eq ▸ (ca.frame.trans cx.frame |>.trans cy.frame).live _ hr)
          rw [hframe r i no,le.memory]
          simp only [nw, false_and,↓reduceIte]
          exact before.frame r i hr
        · intro i outside
          rw [hpadding i outside,le.memory]
          have ne : output.region≠work.region := Ne.symm work_ne_output
          simp only [ne,false_and,↓reduceIte]
          exact ((cx.frame.trans cy.frame).trans cw.frame).frame output.region i ca.live
      next => trivial
    next => trivial
  next => trivial

#axiomsClean launch_error_is_setup
#axiomsClean allocation_failure
#axiomsClean whole
#axiomsClean prepare_correct
#axiomsClean launch_complete
#axiomsClean failed_prefix
end
end VeriTile.Bench.Repairs.KldivCompute.Host

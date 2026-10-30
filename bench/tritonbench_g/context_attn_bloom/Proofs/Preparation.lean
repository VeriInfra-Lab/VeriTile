import bench.tritonbench_g.context_attn_bloom.Proofs.NativeStaging
import bench.tritonbench_g.context_attn_bloom.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Preparation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 3000000

/-- Each data function is the caller storage beginning at the actual passed
pointer, including its storage offset. The public domain uses dense floating
and page-table views. Metadata vectors retain their actual input stride. -/
structure FloatingView where
  negative : Bool
  words : Nat → BitVec 32
  reals : Nat → ℝ

def FloatingView.logicalWords (v : FloatingView) (i : Nat) : BitVec 32 :=
  if v.negative then v.words i ^^^ (BitVec.ofNat 32 0x80000000) else v.words i

def FloatingView.logicalReals (v : FloatingView) (i : Nat) : ℝ :=
  if v.negative then -v.reals i else v.reals i

structure IntegerView where
  negative : Bool
  stride : Nat
  values : Nat → Int

def IntegerView.logical (v : IntegerView) (i : Nat) : Int :=
  if v.negative then -v.values (i*v.stride) else v.values (i*v.stride)

structure Data where
  q : FloatingView
  k : FloatingView
  v : FloatingView
  start : IntegerView
  sequence : IntegerView
  prompt : IntegerView
  request : IntegerView
  /-- Dense page table, flattened in its actual rank-two logical order. -/
  pages : IntegerView

def Data.inputs (d : Data) : Inputs where
  q := d.q.logicalReals
  k := d.k.logicalReals
  v := d.v.logicalReals
  start := d.start.logical
  sequence := d.sequence.logical
  prompt := d.prompt.logical
  request := d.request.logical
  tokens := d.pages.logical

inductive Stage where
  | query | key | value | request | start | sequence | prompt | pages
  deriving DecidableEq

def stages : List Stage := [.query,.key,.value,.request,.start,.sequence,.prompt,.pages]

def region (r : Regions) : Stage → RegionName
  | .query => r.query | .key => r.key | .value => r.value | .request => r.request
  | .start => r.start | .sequence => r.sequence | .prompt => r.prompt | .pages => r.tokens

structure Runtime where
  real : BlockState
  native : State Native.Expr

def Bound (r : Regions) (d : Data) (stage : Stage) (s : Runtime) : Prop :=
  match stage with
  | .query => s.real.readMem r.query=d.q.logicalReals ∧
      ∀ i, (s.native.mem r.query i).read .real=Native.Expr.bits (d.q.logicalWords i)
  | .key => s.real.readMem r.key=d.k.logicalReals ∧
      ∀ i, (s.native.mem r.key i).read .real=Native.Expr.bits (d.k.logicalWords i)
  | .value => s.real.readMem r.value=d.v.logicalReals ∧
      ∀ i, (s.native.mem r.value i).read .real=Native.Expr.bits (d.v.logicalWords i)
  | .request => s.real.readMemValue .int r.request=d.request.logical ∧
      ∀ i, (s.native.mem r.request i).read .int=d.request.logical i
  | .start => s.real.readMemValue .int r.start=d.start.logical ∧
      ∀ i, (s.native.mem r.start i).read .int=d.start.logical i
  | .sequence => s.real.readMemValue .int r.sequence=d.sequence.logical ∧
      ∀ i, (s.native.mem r.sequence i).read .int=d.sequence.logical i
  | .prompt => s.real.readMemValue .int r.prompt=d.prompt.logical ∧
      ∀ i, (s.native.mem r.prompt i).read .int=d.prompt.logical i
  | .pages => s.real.readMemValue .int r.tokens=d.pages.logical ∧
      ∀ i, (s.native.mem r.tokens i).read .int=d.pages.logical i

/-- Ordinary local tensor contracts: one resolve_neg/contiguous call binds
one logical input, preserves other prepared inputs, and does not write the
caller-owned resolved output. No kernel success or output formula is assumed.
The runtime resource/exception boundary is the same ordinary successful API
boundary used for tensor allocation and compilation. -/
structure LocalAPI (r : Regions) (d : Data) where
  visible : RegionName → Prop
  outputVisible : visible r.out
  run : Stage → Runtime → Runtime
  prepares : ∀ stage s, Bound r d stage (run stage s)
  preserves : ∀ stage other s, Bound r d other s → Bound r d other (run stage s)
  outputFrame : ∀ stage s i,
    (run stage s).real.mem r.out i=s.real.mem r.out i ∧
    (run stage s).native.mem r.out i=s.native.mem r.out i
  callerFrame : ∀ stage s rg, visible rg → ∀ i,
    (run stage s).real.mem rg i=s.real.mem rg i ∧
    (run stage s).native.mem rg i=s.native.mem rg i
  undefFrame : ∀ stage s, (run stage s).real.undef=s.real.undef

def prepare (api : LocalAPI r d) (s : Runtime) : Runtime := stages.foldl (fun t stage => api.run stage t) s

theorem fold_bound (api : LocalAPI r d) (jobs : List Stage) (s : Runtime) :
    ∀ stage, stage∈jobs ∨ Bound r d stage s →
      Bound r d stage (jobs.foldl (fun t x => api.run x t) s) := by
  induction jobs generalizing s with
  | nil => intro stage h;exact h.resolve_left (by simp)
  | cons x rest ih =>
    intro stage h
    apply ih
    rcases h with hm|hb
    · rcases List.mem_cons.mp hm with he|hr
      · subst stage;exact Or.inr (api.prepares x s)
      · exact Or.inl hr
    · exact Or.inr (api.preserves x stage s hb)

theorem prepared_inputs (r : Regions) (d : Data) (api : LocalAPI r d) (s : Runtime) :
    inputs (prepare api s).real r=d.inputs ∧
    FPSuccess.Metadata ⟨a,r,d.inputs⟩ (prepare api s).native ∧
    (∀ i, ((prepare api s).native.mem r.query i).read .real=Native.Expr.bits (d.q.logicalWords i)) ∧
    (∀ i, ((prepare api s).native.mem r.key i).read .real=Native.Expr.bits (d.k.logicalWords i)) ∧
    (∀ i, ((prepare api s).native.mem r.value i).read .real=Native.Expr.bits (d.v.logicalWords i)) := by
  have bound : ∀ stage, Bound r d stage (prepare api s) := by
    intro stage
    apply fold_bound api stages s stage
    exact Or.inl (by cases stage <;> simp [stages])
  have q := bound .query;have k := bound .key;have v := bound .value
  have start := bound .start;have seq := bound .sequence;have prompt := bound .prompt
  have request := bound .request;have pages := bound .pages
  refine ⟨?_,⟨start.2,seq.2,prompt.2,request.2,pages.2⟩,q.2,k.2,v.2⟩
  unfold inputs Data.inputs
  rw [q.1,k.1,v.1,start.1,seq.1,prompt.1,request.1,pages.1]

theorem output_frame (api : LocalAPI r d) (s : Runtime) (i : Nat) :
    (prepare api s).real.mem r.out i=s.real.mem r.out i ∧
    (prepare api s).native.mem r.out i=s.native.mem r.out i := by
  unfold prepare
  generalize stages=jobs
  induction jobs generalizing s with
  | nil => exact ⟨rfl,rfl⟩
  | cons stage rest ih =>
    have h := ih (api.run stage s)
    exact ⟨h.1.trans (api.outputFrame stage s i).1,h.2.trans (api.outputFrame stage s i).2⟩

theorem caller_frame (api : LocalAPI r d) (s : Runtime) (rg : RegionName)
    (visible : api.visible rg) (i : Nat) :
    (prepare api s).real.mem rg i=s.real.mem rg i ∧
    (prepare api s).native.mem rg i=s.native.mem rg i := by
  unfold prepare
  generalize stages=jobs
  induction jobs generalizing s with
  | nil => exact ⟨rfl,rfl⟩
  | cons stage rest ih =>
    have h := ih (api.run stage s)
    exact ⟨h.1.trans (api.callerFrame stage s rg visible i).1,
      h.2.trans (api.callerFrame stage s rg visible i).2⟩

theorem undef_frame (api : LocalAPI r d) (s : Runtime) : (prepare api s).real.undef=s.real.undef := by
  unfold prepare
  generalize stages=jobs
  induction jobs generalizing s with
  | nil => rfl
  | cons stage rest ih => exact (ih _).trans (api.undefFrame stage s)

#axiomsClean prepared_inputs
#axiomsClean output_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Preparation

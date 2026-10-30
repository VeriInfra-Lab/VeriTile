import bench.tritonbench_g.iv_dependent_matmul.Proofs.API

/-! Source-ordered wrapper execution from pre-call memory. Successful ordinary
allocations return concrete tensor identities, regions, byte bases and memory;
the launch success branch invokes the same finite Execution.host directly.
Allocation contracts contain no kernel result or execution-success premise. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Wrapper
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section
set_option maxHeartbeats 5000000

structure Buffer where
  region : RegionName
  object : Nat
  base : Nat
  extent : Nat

structure Heap (α : Type) where
  state : State α
  buffers : List Buffer

/-- Actual object and pointer returned by the ordinary FP32 allocation API.
Shape, contiguous strides and device are part of its supplied contract. -/
structure Tensor (shape strides : List Nat) (device : String) where
  region : RegionName
  object : Nat
  base : Nat

def Tensor.buffer {shape strides : List Nat} {device : String}
    (t : Tensor shape strides device) : Buffer := ⟨t.region,t.object,t.base,shape.prod⟩

def Tensor.metadata {shape strides : List Nat} {device : String}
    (t : Tensor shape strides device) : TensorMetadata := ⟨t.region,shape,strides,.fp32,device⟩

/-- Ordinary fresh allocation contract. Failure leaves the supplied tensor heap
unchanged. Successful allocation may initialize only its newly allocated cells;
all older live regions, identities and physical byte intervals remain distinct. -/
structure Allocated (α : Type) (before : Heap α) (shape strides : List Nat) (device : String) where
  tensor : Tensor shape strides device
  after : Heap α
  registry : after.buffers=tensor.buffer::before.buffers
  freshRegion : ∀ old∈before.buffers,tensor.region≠old.region
  freshObject : ∀ old∈before.buffers,tensor.object≠old.object
  physical : ∀ old∈before.buffers,
    shape.prod=0 ∨ old.extent=0 ∨ tensor.base+4*shape.prod≤old.base ∨ old.base+4*old.extent≤tensor.base
  aligned : tensor.base%4=0
  addressable : tensor.base+4*shape.prod≤2^64
  typed : ∀ o<shape.prod,∃ v : α,after.state.mem tensor.region o=Cell.mk .fp32 v
  frame : ∀ region o,¬(region=tensor.region ∧ o<shape.prod) →
    after.state.mem region o=before.state.mem region o

structure Prepared (α : Type) (r : Request) (before : Heap α) where
  a : Allocated α before [r.rows,r.inner] [max r.inner 1,1] r.device
  b : Allocated α a.after [r.inner,r.columns] [max r.columns 1,1] r.device
  c : Allocated α b.after [r.rows,r.columns] [max r.columns 1,1] r.device

def Prepared.request {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) : Request :=
  {r with A:=p.a.tensor.region,B:=p.b.tensor.region,C:=p.c.tensor.region}

def Prepared.footprint {α : Type} {r : Request} {before : Heap α}
    (p : Prepared α r before) (region : RegionName) (o : Nat) : Prop :=
  (region=p.a.tensor.region ∧ o<r.rows*r.inner) ∨
  (region=p.b.tensor.region ∧ o<r.inner*r.columns) ∨
  (region=p.c.tensor.region ∧ o<r.rows*r.columns)

theorem Prepared.ba {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) :
    p.b.tensor.region≠p.a.tensor.region :=
  p.b.freshRegion p.a.tensor.buffer (by rw [p.a.registry]; exact List.mem_cons_self)

theorem Prepared.ca {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) :
    p.c.tensor.region≠p.a.tensor.region :=
  p.c.freshRegion p.a.tensor.buffer (by rw [p.b.registry,p.a.registry]; simp)

theorem Prepared.cb {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) :
    p.c.tensor.region≠p.b.tensor.region :=
  p.c.freshRegion p.b.tensor.buffer (by rw [p.b.registry]; exact List.mem_cons_self)

theorem Prepared.memoryA {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) (o : Nat) :
    p.c.after.state.mem p.a.tensor.region o=p.a.after.state.mem p.a.tensor.region o := by
  exact (p.c.frame _ _ (by intro h; exact p.ca h.1.symm)).trans
    (p.b.frame _ _ (by intro h; exact p.ba h.1.symm))

theorem Prepared.memoryB {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before) (o : Nat) :
    p.c.after.state.mem p.b.tensor.region o=p.b.after.state.mem p.b.tensor.region o :=
  p.c.frame _ _ (by intro h; exact p.cb h.1.symm)

theorem Prepared.frame {α : Type} {r : Request} {before : Heap α}
    (p : Prepared α r before) (region : RegionName) (o : Nat) (hn : ¬p.footprint region o) :
    p.c.after.state.mem region o=before.state.mem region o := by
  have ha : ¬(region=p.a.tensor.region ∧ o<[r.rows,r.inner].prod) := by
    simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using
      (show ¬(region=p.a.tensor.region ∧ o<r.rows*r.inner) from fun h => hn (Or.inl h))
  have hb : ¬(region=p.b.tensor.region ∧ o<[r.inner,r.columns].prod) := by
    simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using
      (show ¬(region=p.b.tensor.region ∧ o<r.inner*r.columns) from fun h => hn (Or.inr (Or.inl h)))
  have hc : ¬(region=p.c.tensor.region ∧ o<[r.rows,r.columns].prod) := by
    simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using
      (show ¬(region=p.c.tensor.region ∧ o<r.rows*r.columns) from fun h => hn (Or.inr (Or.inr h)))
  exact (p.c.frame region o hc).trans ((p.b.frame region o hb).trans (p.a.frame region o ha))

theorem Prepared.caller_outside {α : Type} {r : Request} {before : Heap α}
    (p : Prepared α r before) (old : Buffer) (ho : old∈before.buffers) (o : Nat) :
    ¬p.footprint old.region o := by
  have ha := p.a.freshRegion old ho
  have hb := p.b.freshRegion old (by rw [p.a.registry]; exact List.mem_cons_of_mem _ ho)
  have hc := p.c.freshRegion old (by rw [p.b.registry,p.a.registry]; simp [ho])
  intro h
  rcases h with h | h | h
  · exact ha h.1.symm
  · exact hb h.1.symm
  · exact hc h.1.symm

/-- Pure geometry from ordinary allocations. The memory premises are only the
input tensor-library dtype guarantees, not any kernel output property. -/
def Prepared.allocation {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before)
    (s : State Word)
    (ha : ∀ o<r.rows*r.inner,∃ v,s.mem p.a.tensor.region o=Cell.mk .fp32 v)
    (hb : ∀ o<r.inner*r.columns,∃ v,s.mem p.b.tensor.region o=Cell.mk .fp32 v) : Allocation p.request s where
  freshA := p.ca
  freshB := p.cb
  baseA := p.a.tensor.base
  baseB := p.b.tensor.base
  baseC := p.c.tensor.base
  physicalFreshA := by
    have h := p.c.physical p.a.tensor.buffer (by rw [p.b.registry,p.a.registry]; simp)
    simp only [Tensor.buffer,List.prod_cons,List.prod_nil,Nat.mul_one] at h
    change r.rows*r.inner=0 ∨ r.rows*r.columns=0 ∨
      p.a.tensor.base+4*(r.rows*r.inner)≤p.c.tensor.base ∨ p.c.tensor.base+4*(r.rows*r.columns)≤p.a.tensor.base
    tauto
  physicalFreshB := by
    have h := p.c.physical p.b.tensor.buffer (by rw [p.b.registry]; exact List.mem_cons_self)
    simp only [Tensor.buffer,List.prod_cons,List.prod_nil,Nat.mul_one] at h
    change r.inner*r.columns=0 ∨ r.rows*r.columns=0 ∨
      p.b.tensor.base+4*(r.inner*r.columns)≤p.c.tensor.base ∨ p.c.tensor.base+4*(r.rows*r.columns)≤p.b.tensor.base
    tauto
  alignA := p.a.aligned
  alignB := p.b.aligned
  alignC := p.c.aligned
  inputA := ha
  inputB := hb

theorem Prepared.addressable {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before)
    (s : State Word)
    (ha : ∀ o<r.rows*r.inner,∃ v,s.mem p.a.tensor.region o=Cell.mk .fp32 v)
    (hb : ∀ o<r.inner*r.columns,∃ v,s.mem p.b.tensor.region o=Cell.mk .fp32 v) :
    Index64.Addressable p.request s (p.allocation s ha hb) where
  input := by simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using p.a.addressable
  weight := by simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using p.b.addressable
  output := by simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using p.c.addressable

theorem Prepared.typedA {r : Request} {before : Heap Word} (p : Prepared Word r before) :
    ∀ o<r.rows*r.inner,∃ v,p.c.after.state.mem p.a.tensor.region o=Cell.mk .fp32 v := by
  intro o ho
  rw [p.memoryA o]
  exact p.a.typed o (by simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using ho)

theorem Prepared.typedB {r : Request} {before : Heap Word} (p : Prepared Word r before) :
    ∀ o<r.inner*r.columns,∃ v,p.c.after.state.mem p.b.tensor.region o=Cell.mk .fp32 v := by
  intro o ho
  rw [p.memoryB o]
  exact p.b.typed o (by simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using ho)

/-- Geometry-only Word state for the independent Real interpretation. Its
values never enter Real execution or the mathematical target. -/
def layout [Inhabited Word] : State Word where
  mem := fun _ _ => Cell.mk .fp32 (FP.Value.zero .fp32)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

def Prepared.realAllocation [Inhabited Word] {r : Request} {before : Heap ℝ}
    (p : Prepared ℝ r before) : Allocation p.request layout :=
  p.allocation layout (by intro o ho; exact ⟨_,rfl⟩) (by intro o ho; exact ⟨_,rfl⟩)

theorem Prepared.realAddressable [Inhabited Word] {r : Request} {before : Heap ℝ}
    (p : Prepared ℝ r before) : Index64.Addressable p.request layout p.realAllocation :=
  p.addressable layout _ _

/-- An ordinary launch failure may have occurred after submission. Its partial
memory is propagated without any kernel result or preservation assumption. -/
structure LaunchFailure (α Error : Type) where
  error : Error
  state : State α

structure Calls (α Device Error : Type) where
  device : String → Except Error Device
  random : Device → (shape strides : List Nat) → (device : String) → (before : Heap α) →
    Except Error (Allocated α before shape strides device)
  empty : Device → (shape strides : List Nat) → (device : String) → (before : Heap α) →
    Except Error (Allocated α before shape strides device)
  launch : Device → (r : Request) → Buffer → Buffer → Buffer → State α →
    Except (LaunchFailure α Error) (PLift (Index64.Launchable r))

structure Failure (α Error : Type) where
  trace : List API.Stage
  error : Error
  heap : Heap α

structure Returned (α : Type) (r : Request) where
  trace : List API.Stage
  tensor : Tensor [r.rows,r.columns] [max r.columns 1,1] r.device
  heap : Heap α

def prepare {α Device Error : Type} (api : Calls α Device Error) (r : Request) (before : Heap α) :
    Except (Failure α Error) (Device × Prepared α r before) :=
  match api.device r.device with
  | .error e => .error ⟨[.device],e,before⟩
  | .ok d => match api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before with
    | .error e => .error ⟨[.device,.randomA],e,before⟩
    | .ok a => match api.random d [r.inner,r.columns] [max r.columns 1,1] r.device a.after with
      | .error e => .error ⟨[.device,.randomA,.randomB],e,a.after⟩
      | .ok b => match api.empty d [r.rows,r.columns] [max r.columns 1,1] r.device b.after with
        | .error e => .error ⟨[.device,.randomA,.randomB,.emptyC],e,b.after⟩
        | .ok c => .ok (d,⟨a,b,c⟩)

def PreparedBy {α Device Error : Type} (api : Calls α Device Error) (r : Request) (before : Heap α)
    (d : Device) (p : Prepared α r before) : Prop :=
  api.device r.device=.ok d ∧
  api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before=.ok p.a ∧
  api.random d [r.inner,r.columns] [max r.columns 1,1] r.device p.a.after=.ok p.b ∧
  api.empty d [r.rows,r.columns] [max r.columns 1,1] r.device p.b.after=.ok p.c

def launch {α Device Error : Type} (api : Calls α Device Error) {r : Request} {before : Heap α}
    (d : Device) (p : Prepared α r before) : Except (LaunchFailure α Error) (PLift (Index64.Launchable p.request)) :=
  api.launch d p.request p.a.tensor.buffer p.b.tensor.buffer p.c.tensor.buffer p.c.after.state

def result {α : Type} [Inhabited α] (arithmetic : (r : Request) → Execution.Arithmetic α r)
    (g : Graph) {r : Request} {before : Heap α} (p : Prepared α r before) : Returned α r :=
  ⟨[.device,.randomA,.randomB,.emptyC,.launch,.returned],p.c.tensor,
    {p.c.after with state:=(Execution.host (arithmetic p.request) g
      p.a.tensor.base p.b.tensor.base p.c.tensor.base p.c.after.state).2}⟩

/-- The launch boundary supplies ordinary launch admissibility only. The
computed success state is the proved finite host. An error propagates the
ordinary API partial memory, without claiming all runtime errors preserve it. -/
def run {α Device Error : Type} [Inhabited α]
    (arithmetic : (r : Request) → Execution.Arithmetic α r) (g : Graph)
    (api : Calls α Device Error) (r : Request) (before : Heap α) : Except (Failure α Error) (Returned α r) :=
  match prepare api r before with
  | .error e => .error e
  | .ok (d,p) => match launch api d p with
    | .error e => .error ⟨[.device,.randomA,.randomB,.emptyC,.launch],e.error,{p.c.after with state:=e.state}⟩
    | .ok _ => .ok (result arithmetic g p)

inductive PreparationFailure {α Device Error : Type} (api : Calls α Device Error)
    (r : Request) (before : Heap α) : Failure α Error → Prop
  | device (e : Error) (h : api.device r.device=.error e) :
      PreparationFailure api r before ⟨[.device],e,before⟩
  | randomA (d : Device) (hd : api.device r.device=.ok d) (e : Error)
      (h : api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before=.error e) :
      PreparationFailure api r before ⟨[.device,.randomA],e,before⟩
  | randomB (d : Device) (hd : api.device r.device=.ok d)
      (a : Allocated α before [r.rows,r.inner] [max r.inner 1,1] r.device)
      (ha : api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before=.ok a) (e : Error)
      (h : api.random d [r.inner,r.columns] [max r.columns 1,1] r.device a.after=.error e) :
      PreparationFailure api r before ⟨[.device,.randomA,.randomB],e,a.after⟩
  | emptyC (d : Device) (hd : api.device r.device=.ok d)
      (a : Allocated α before [r.rows,r.inner] [max r.inner 1,1] r.device)
      (ha : api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before=.ok a)
      (b : Allocated α a.after [r.inner,r.columns] [max r.columns 1,1] r.device)
      (hb : api.random d [r.inner,r.columns] [max r.columns 1,1] r.device a.after=.ok b) (e : Error)
      (h : api.empty d [r.rows,r.columns] [max r.columns 1,1] r.device b.after=.error e) :
      PreparationFailure api r before ⟨[.device,.randomA,.randomB,.emptyC],e,b.after⟩

inductive FailedAt {α Device Error : Type} (api : Calls α Device Error)
    (r : Request) (before : Heap α) : Failure α Error → Prop
  | preparation {e} : PreparationFailure api r before e → FailedAt api r before e
  | launch (d : Device) (p : Prepared α r before) (hp : PreparedBy api r before d p)
      (e : LaunchFailure α Error) (hl : launch api d p=.error e) :
      FailedAt api r before ⟨[.device,.randomA,.randomB,.emptyC,.launch],e.error,{p.c.after with state:=e.state}⟩

theorem prepare_cases {α Device Error : Type} (api : Calls α Device Error) (r : Request) (before : Heap α) :
    (∃ e,prepare api r before=.error e ∧ PreparationFailure api r before e) ∨
    (∃ d p,prepare api r before=.ok (d,p) ∧ PreparedBy api r before d p) := by
  cases hd : api.device r.device with
  | error e => exact Or.inl ⟨_,by simp [prepare,hd],.device e hd⟩
  | ok d =>
    cases ha : api.random d [r.rows,r.inner] [max r.inner 1,1] r.device before with
    | error e => exact Or.inl ⟨_,by simp [prepare,hd,ha],.randomA d hd e ha⟩
    | ok a =>
      cases hb : api.random d [r.inner,r.columns] [max r.columns 1,1] r.device a.after with
      | error e => exact Or.inl ⟨_,by simp [prepare,hd,ha,hb],.randomB d hd a ha e hb⟩
      | ok b =>
        cases hc : api.empty d [r.rows,r.columns] [max r.columns 1,1] r.device b.after with
        | error e => exact Or.inl ⟨_,by simp [prepare,hd,ha,hb,hc],.emptyC d hd a ha b hb e hc⟩
        | ok c => exact Or.inr ⟨d,⟨a,b,c⟩,by simp [prepare,hd,ha,hb,hc],hd,ha,hb,hc⟩

theorem run_cases {α Device Error : Type} [Inhabited α]
    (arithmetic : (r : Request) → Execution.Arithmetic α r) (g : Graph)
    (api : Calls α Device Error) (r : Request) (before : Heap α) :
    (∃ e,run arithmetic g api r before=.error e ∧ FailedAt api r before e) ∨
    (∃ d : Device, ∃ p : Prepared α r before, ∃ support : Index64.Launchable p.request,
      run arithmetic g api r before=.ok (result arithmetic g p) ∧
      PreparedBy api r before d p ∧ launch api d p=.ok ⟨support⟩) := by
  rcases prepare_cases api r before with ⟨e,he,hf⟩ | ⟨d,p,hp,ht⟩
  · exact Or.inl ⟨e,by simp [run,he],.preparation hf⟩
  · cases hl : launch api d p with
    | error e => exact Or.inl ⟨_,by simp [run,hp,hl],.launch d p ht e hl⟩
    | ok support => exact Or.inr ⟨d,p,support.down,by simp [run,hp,hl],ht,hl⟩

/-- The final tensor is exactly the returned empty-allocation object, with the
metadata of the actual C region selected by the allocator. -/
theorem result_identity {α : Type} [Inhabited α]
    (arithmetic : (r : Request) → Execution.Arithmetic α r) (g : Graph)
    {r : Request} {before : Heap α} (p : Prepared α r before) :
    (result arithmetic g p).tensor=p.c.tensor ∧
    (result arithmetic g p).tensor.metadata=outputMetadata p.request := ⟨rfl,rfl⟩

/-- Compose a proved kernel frame with the three allocation frames. This is a
structural composition lemma; public entries supply the proved host frame. -/
theorem caller_frame {α : Type} {r : Request} {before : Heap α} (p : Prepared α r before)
    (final : State α) (frame : ∀ region o,¬Footprint p.request region o →
      final.mem region o=p.c.after.state.mem region o) :
    (∀ region o,¬p.footprint region o → final.mem region o=before.state.mem region o) ∧
    (∀ old∈before.buffers,∀ o,final.mem old.region o=before.state.mem old.region o) := by
  have hf : ∀ region o,¬p.footprint region o → final.mem region o=before.state.mem region o := by
    intro region o hn
    have hc : ¬Footprint p.request region o := by
      intro h
      exact hn (Or.inr (Or.inr h))
    exact (frame region o hc).trans (p.frame region o hn)
  exact ⟨hf,fun old ho o => hf old.region o (p.caller_outside old ho o)⟩

#axiomsClean Prepared.frame
#axiomsClean Prepared.addressable
#axiomsClean run_cases
#axiomsClean result_identity
#axiomsClean caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Wrapper

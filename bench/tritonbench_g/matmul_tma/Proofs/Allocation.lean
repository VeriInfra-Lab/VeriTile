import bench.tritonbench_g.matmul_tma.Proofs.Host

namespace VeriTile.Bench.TritonBenchG.MatmulTma.Host
open VeriTile.Triton FP.Structural FPValue
noncomputable section

inductive AllocationKind where | randn | empty
structure AllocationRequest where
  kind : AllocationKind
  dtype : FloatDType
  rows : Nat
  columns : Nat
  /-- Every allocation in this wrapper uses CUDA; the selected device/context
  is admitted by the ordinary runtime boundary. -/
  cuda : Bool := Bool.true

abbrev Allocator (α : Type) := AllocationRequest → State α → Except String (RegionName × State α)

def aRequest (h : Shape) (f : Flags) : AllocationRequest :=
  if f.transA then ⟨.randn,.fp16,h.inner,h.rows,Bool.true⟩
  else ⟨.randn,.fp16,h.rows,h.inner,Bool.true⟩
def bRequest (h : Shape) (f : Flags) : AllocationRequest :=
  if f.transB then ⟨.randn,.fp16,h.columns,h.inner,Bool.true⟩
  else ⟨.randn,.fp16,h.inner,h.columns,Bool.true⟩
def cRequest (h : Shape) (d : FloatDType) : AllocationRequest :=
  ⟨.empty,d,h.rows,h.columns,Bool.true⟩

/-- Calls randn twice, then always empty(fp32), then empty(fp16) only when
requested. Transpose is a view: its stride/shape changes are in `parameters`. -/
def prepare {α : Type} (alloc : Allocator α) (h : Shape) (f : Flags)
    (initial : State α) : Except String (Regions × State α) := do
  let (a,sa) ← alloc (aRequest h f) initial
  let (b,sb) ← alloc (bRequest h f) sa
  let (c,sc) ← alloc (cRequest h .fp32) sb
  if f.half then
    let (replacement,s) ← alloc (cRequest h .fp16) sc
    .ok (⟨a,b,c,replacement⟩,s)
  else .ok (⟨a,b,c,c⟩,sc)

/-- Successful ordinary allocation calls, with no program execution or output
value premise. The intermediate states retain the original allocation order. -/
def Prepared {α : Type} (alloc : Allocator α) (h : Shape) (f : Flags)
    (r : Regions) (initial s : State α) : Prop :=
  ∃ sa sb sc,
    alloc (aRequest h f) initial = .ok (r.a,sa) ∧
    alloc (bRequest h f) sa = .ok (r.b,sb) ∧
    alloc (cRequest h .fp32) sb = .ok (r.first,sc) ∧
    (if f.half then alloc (cRequest h .fp16) sc = .ok (r.replacement,s)
     else r.replacement = r.first ∧ s = sc)

theorem prepare_correct {α : Type} (alloc : Allocator α) (h : Shape) (f : Flags)
    (r : Regions) (initial s : State α) (prepared : Prepared alloc h f r initial s) :
    prepare alloc h f initial = .ok (r,s) := by
  rcases prepared with ⟨sa,sb,sc,ha,hb,hc,hd⟩
  rcases r with ⟨a,b,c,d⟩
  cases hf : f.half
  · simp only [hf, Bool.false_eq_true, ↓reduceIte] at hd
    rcases hd with ⟨rfl,rfl⟩
    simp [prepare,hf,ha,hb,hc,Except.bind,bind]
  · simp only [hf, ↓reduceIte] at hd
    simp [prepare,hf,ha,hb,hc,hd,Except.bind,bind]

/-- Entire source wrapper after the Torch API boundary is supplied. Allocation
and native-setup failures propagate; only source execution produces a result. -/
def wrapper {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (alloc : Allocator α) (admit : Admission) (h : Shape) (f : Flags) (cfg : Config)
    (initial : State α) : Except String (Tensor × State α) := do
  let (r,s) ← prepare alloc h f initial
  run M D admit h f r cfg s

theorem wrapper_correct {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (alloc : Allocator α) (admit : Admission) (h : Shape) (f : Flags) (cfg : Config)
    (r : Regions) (initial s : State α)
    (prepared : Prepared alloc h f r initial s)
    (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, wrapper M D alloc admit h f cfg initial = .ok (returnedTensor h f r,t) ∧
      Returned M D h f r s t := by
  obtain ⟨t,he,hp⟩ := run_correct M D admit h f r cfg s ready
  exact ⟨t, by simp [wrapper,prepare_correct alloc h f r initial s prepared,
    Except.bind,bind,he], hp⟩

theorem allocation_error {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (alloc : Allocator α) (admit : Admission) (h : Shape) (f : Flags) (cfg : Config)
    (initial : State α) (message : String)
    (failed : prepare alloc h f initial = .error message) :
    wrapper M D alloc admit h f cfg initial = .error message := by
  simp [wrapper,failed,Except.bind,bind]

/-- Ordinary allocator API frame: randn/empty may initialize their returned
fresh allocation, but they cannot modify another tensor's cells. -/
def AllocatorFrame {α : Type} (alloc : Allocator α) : Prop :=
  ∀ request before region after, alloc request before = .ok (region,after) →
    ∀ R off, R ≠ region → after.mem R off = before.mem R off

def OutsideAllocated (f : Flags) (r : Regions) (R : RegionName) : Prop :=
  R ≠ r.a ∧ R ≠ r.b ∧ R ≠ r.first ∧ (f.half = Bool.true → R ≠ r.replacement)

theorem prepared_frame {α : Type} (alloc : Allocator α) (h : Shape) (f : Flags)
    (r : Regions) (initial s : State α) (prepared : Prepared alloc h f r initial s)
    (frame : AllocatorFrame alloc) (R : RegionName) (off : Nat)
    (outside : OutsideAllocated f r R) : s.mem R off = initial.mem R off := by
  rcases prepared with ⟨sa,sb,sc,ha,hb,hc,hd⟩
  have h1 := frame _ _ _ _ ha R off outside.1
  have h2 := frame _ _ _ _ hb R off outside.2.1
  have h3 := frame _ _ _ _ hc R off outside.2.2.1
  cases hh : f.half
  · simp only [hh, Bool.false_eq_true, ↓reduceIte] at hd
    rw [hd.2,h3,h2,h1]
  · simp only [hh, ↓reduceIte] at hd
    have h4 := frame _ _ _ _ hd R off (outside.2.2.2 hh)
    exact h4.trans (h3.trans (h2.trans h1))

theorem returned_frame_initial {α : Type} [Inhabited α]
    (M : Algebra α) (D : Dot α) (alloc : Allocator α) (h : Shape) (f : Flags)
    (r : Regions) (initial s t : State α) (prepared : Prepared alloc h f r initial s)
    (frame : AllocatorFrame alloc) (post : Returned M D h f r s t)
    (R : RegionName) (off : Nat) (outside : OutsideAllocated f r R) :
    t.mem R off = initial.mem R off := by
  have hp : t.mem R off = s.mem R off := by
    apply post.2 R off
    left
    change R ≠ (returnedTensor h f r).region
    rw [returned_region]
    cases hh : f.half
    · simpa [outputRegion,hh] using outside.2.2.1
    · simpa [outputRegion,hh] using outside.2.2.2 hh
  exact hp.trans (prepared_frame alloc h f r initial s prepared frame R off outside)

#axiomsClean prepared_frame
#axiomsClean returned_frame_initial
#axiomsClean prepare_correct
#axiomsClean wrapper_correct
#axiomsClean allocation_error
#specNonCircular Prepared avoiding [prepare,run,wrapper]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Host

import bench.tritonbench_g.seeded_dropout.Proofs.WholeGrid

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

abbrev PhysicalMemory (α : Type) := Nat → Nat → α

/-- The ordinary Torch tensor metadata and logical-index contract. Inactive
padding is not an allocated element. Scalars have shape=[] and numel=1. -/
structure Tensor where
  shape : List Nat
  allocation : Nat
  base : Nat
  offset : Nat → Nat
  contiguous : Bool
  dtype : Storage
  cudaDevice : Nat
  contiguousOffsets : contiguous = Bool.true → ∀ i, i < shape.prod → offset i = base+i

def Tensor.numel (x : Tensor) : Nat := x.shape.prod
def Tensor.read (x : Tensor) (mem : PhysicalMemory α) (i : Nat) : α :=
  mem x.allocation (x.offset i)

/-- Only ordinary empty_like facts: fresh arbitrary storage, metadata, and
preservation of every preexisting allocation. This runs before the assertion. -/
structure Allocated (α : Type) (input : Tensor) (entry : PhysicalMemory α)
    (preexisting : Nat → Prop) where
  out : Tensor
  memory : PhysicalMemory α
  inputPreexisting : preexisting input.allocation
  outputFresh : ¬ preexisting out.allocation
  shape : out.shape = input.shape
  dtype : out.dtype = input.dtype
  device : out.cudaDevice = input.cudaDevice
  contiguous : input.contiguous = Bool.true → out.contiguous = Bool.true
  frame : ∀ allocation a, preexisting allocation → memory allocation a = entry allocation a

def Allocated.layout (a : Allocated α input entry preexisting) : Layout :=
  ⟨"X","OUT",input.numel,1024,SourceDomain.wrapper_block_legal⟩

def Allocated.state [Inhabited α] (a : Allocated α input entry preexisting) : State α where
  mem := fun region i => Cell.mk .real
    (if region = ("X" : RegionName) then a.memory input.allocation (input.base+i)
     else a.memory a.out.allocation (a.out.base+i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax = 0 then SourceDomain.grid input.numel 1024 else 1
  undef := fun d _ _ => defaultValue d

structure Domain (a : Allocated α input entry preexisting) where
  dtype : input.dtype = .fp32
  size : input.numel < 2^31
  capacity : Nat → Nat
  inputBase : input.base ≤ capacity input.allocation
  outputBase : a.out.base ≤ capacity a.out.allocation
  inputBounds : ∀ i, i < input.numel → input.base+i < capacity input.allocation
  outputBounds : ∀ i, i < input.numel → a.out.base+i < capacity a.out.allocation

/-- The complete launched order; it supplies no kernel result or value. -/
structure Schedule (a : Allocated α input entry preexisting) where
  order : List Nat
  complete : order.Perm (grid a.layout)

def Allocated.commit [Inhabited α] (a : Allocated α input entry preexisting)
    (final : State α) : PhysicalMemory α :=
  fun allocation offset => if allocation = a.out.allocation ∧ a.out.base ≤ offset then
    (final.mem "OUT" (offset-a.out.base)).read .real else a.memory allocation offset

def PhysicalOutside (a : Allocated α input entry preexisting) (allocation offset : Nat) : Prop :=
  ∀ i, i < input.numel → (allocation,offset) ≠ (a.out.allocation,a.out.base+i)

theorem commit_frame [Inhabited α] (a : Allocated α input entry preexisting) (final : State α)
    (frame : ∀ region i, (region ≠ a.layout.OUT ∨ a.layout.n ≤ i) →
      final.mem region i = a.state.mem region i)
    (allocation offset : Nat) (outside : PhysicalOutside a allocation offset) :
    a.commit final allocation offset = a.memory allocation offset := by
  unfold Allocated.commit
  split
  · rename_i h
    rcases h with ⟨same,base⟩
    subst allocation
    have bound : a.layout.n ≤ offset-a.out.base := by
      by_contra h
      have hi : offset-a.out.base < input.numel := Nat.lt_of_not_ge h
      apply outside (offset-a.out.base) hi
      congr 1
      omega
    rw [frame "OUT" (offset-a.out.base) (Or.inr bound)]
    change a.memory a.out.allocation (a.out.base+(offset-a.out.base)) = a.memory a.out.allocation offset
    congr 1
    omega
  · rfl

theorem preexisting_frame [Inhabited α] (a : Allocated α input entry preexisting) (final : State α)
    (frame : ∀ region i, (region ≠ a.layout.OUT ∨ a.layout.n ≤ i) →
      final.mem region i = a.state.mem region i)
    (allocation offset : Nat) (existed : preexisting allocation) :
    a.commit final allocation offset = entry allocation offset := by
  have outside : PhysicalOutside a allocation offset := by
    intro i hi equal
    apply a.outputFresh
    have same : allocation = a.out.allocation := congrArg Prod.fst equal
    simpa only [← same] using existed
  exact (commit_frame a final frame allocation offset outside).trans (a.frame allocation offset existed)

theorem input_read [Inhabited α] (a : Allocated α input entry preexisting)
    (contiguous : input.contiguous = Bool.true) (i : Nat) (hi : i < input.numel) :
    ((a.state.mem a.layout.X i).read .real) = input.read entry i := by
  simp only [Allocated.state,Allocated.layout,Cell.read_same,if_true,Tensor.read,
    input.contiguousOffsets contiguous i hi,a.frame input.allocation _ a.inputPreexisting]

def entryRun [Inhabited α] (M : Algebra α) (a : Allocated α input entry preexisting)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (schedule : Schedule a) :
    Option (State α) :=
  if input.contiguous then runTiles M a.layout kind p seed schedule.order a.state else none

theorem entry_result [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (a : Allocated α input entry preexisting)
    (d : Domain a) (contiguous : input.contiguous = Bool.true)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind)
    (validSeed : kind.inRange (seedInt kind seed) = Bool.true) (schedule : Schedule a) :
    ∃ final, entryRun M a kind p seed schedule = some final ∧
      (∀ i, i < input.numel → a.out.read (a.commit final) i =
        scalar M lt p (seedInt kind seed) i (input.read entry i)) ∧
      (∀ allocation offset, PhysicalOutside a allocation offset →
        a.commit final allocation offset = a.memory allocation offset) ∧
      (∀ allocation offset, preexisting allocation →
        a.commit final allocation offset = entry allocation offset) := by
  obtain ⟨final,ran,values,frame⟩ := grid_result M lt compare a.layout kind p seed d.size validSeed
    (by change ("X" : RegionName) ≠ "OUT"; decide) schedule.order schedule.complete a.state
  refine ⟨final,?_,?_,commit_frame a final frame,preexisting_frame a final frame⟩
  · simp only [entryRun,contiguous,if_true,ran]
  · intro i hi
    have ho : i < a.out.shape.prod := by simpa [a.shape] using hi
    have offset := a.out.contiguousOffsets (a.contiguous contiguous) i ho
    simp only [Tensor.read,offset,Allocated.commit,le_add_iff_nonneg_right,Nat.zero_le,
      and_self,if_true,Nat.add_sub_cancel_left]
    have v := values i hi
    rw [input_read a contiguous i hi] at v
    change (final.mem a.layout.OUT i).read .real = _
    rw [v]
    rfl

/-- Source order, including allocation before the only assertion. -/
def hostOperations : List String :=
  ["output=torch.empty_like(x)","assert x.is_contiguous()","n_elements=x.numel()",
   "grid=(triton.cdiv(n_elements,BLOCK_SIZE),)","_seeded_dropout[grid](BLOCK_SIZE=1024)","return output"]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole

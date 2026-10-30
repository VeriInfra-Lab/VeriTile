import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholeGrid

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

inductive Config where
  | b32 | b64 | b128 | b256 | b512 | b1024
  deriving DecidableEq, Repr

def Config.block : Config → Nat
  | .b32 => 32 | .b64 => 64 | .b128 => 128
  | .b256 => 256 | .b512 => 512 | .b1024 => 1024

theorem Config.positive (c : Config) : 0 < c.block := by cases c <;> decide

/-- The runtime chooses configurations and a legal order only. It cannot
provide any final value, memory, or scalar formula. -/
structure Launch (c : Layout) where
  config : Config
  order : List (Nat × Nat)
  schedule : order.Perm (grid c config.block)

def runLaunch {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (launch : Launch c) (s : State α) : Option (State α) :=
  runTiles M c launch.config.block launch.order s

def runTrials {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) :
    List (Launch c) → State α → Option (State α)
  | [], s => some s
  | launch :: rest, s => (runLaunch M c launch s).bind (runTrials M c rest)

def WholeOutside (c : Layout) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.OUT ∨ ∀ r q, r < c.rows → q < c.cols → a ≠ r*c.so+q

theorem trials_frame {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (own : Ownership c) (trials : List (Launch c)) (s : State α) :
    ∃ final, runTrials M c trials s = some final ∧
      ∀ region a, WholeOutside c region a → final.mem region a = s.mem region a := by
  induction trials generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons launch rest ih =>
    obtain ⟨middle,ran,_,frame⟩ := grid_result M c launch.config.block
      launch.config.positive own launch.order launch.schedule s
    obtain ⟨final,tail,tailFrame⟩ := ih middle
    refine ⟨final,?_,?_⟩
    · simp only [runTrials,runLaunch,ran,Option.bind_some,tail]
    · intro region a outside
      exact (tailFrame region a outside).trans (frame region a outside)

def tunedLaunch {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (trials : List (Launch c)) (selected : Launch c) (s : State α) : Option (State α) :=
  (runTrials M c trials s).bind (runLaunch M c selected)

theorem tuned_result {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (own : Ownership c) (trials : List (Launch c)) (selected : Launch c) (s : State α) :
    ∃ final, tunedLaunch M c trials selected s = some final ∧
      (∀ r q, r < c.rows → q < c.cols →
        final.mem c.OUT (r*c.so+q) = Cell.mk .real (scalar M
          ((s.mem c.X (r*c.sx+q)).read .real) ((s.mem c.Y (r*c.sy+q)).read .real))) ∧
      (∀ region a, WholeOutside c region a → final.mem region a = s.mem region a) := by
  obtain ⟨middle,tried,trialFrame⟩ := trials_frame M c own trials s
  obtain ⟨final,ran,values,frame⟩ := grid_result M c selected.config.block
    selected.config.positive own selected.order selected.schedule middle
  refine ⟨final,?_,?_,?_⟩
  · simp only [tunedLaunch,runLaunch,tried,Option.bind_some,ran]
  · intro r q hr hq
    rw [values r q hr hq,trialFrame c.X _ (Or.inl own.xSeparate),
      trialFrame c.Y _ (Or.inl own.ySeparate)]
  · intro region a outside
    exact (frame region a outside).trans (trialFrame region a outside)

abbrev PhysicalMemory (α : Type) := Nat → Nat → α

/-- Tensor's logical flat indexing is supplied by the ordinary Torch view
contract; allocation and offsets are physical, not separate copies per view. -/
structure Tensor where
  shape : List Nat
  allocation : Nat
  offset : Nat → Nat
  lastStride : Nat
  cudaDevice : Nat

def Tensor.read (t : Tensor) (mem : PhysicalMemory α) (k : Nat) : α :=
  mem t.allocation (t.offset k)

structure Matrix where
  allocation : Nat
  base : Nat
  rowStride : Nat
  lastStride : Nat
  rows : Nat
  cols : Nat
  cudaDevice : Nat

def Matrix.address (v : Matrix) (r q : Nat) : Nat := v.base+r*v.rowStride+q*v.lastStride

def Matrix.read (v : Matrix) (mem : PhysicalMemory α) (r q : Nat) : α :=
  mem v.allocation (v.address r q)

/-- Metadata/copy/allocator receipt, before the source assertions and launch.
The equalities mention only input values; no result callback is admitted.
A supplied out may reshape by copying, exactly as Python torch.reshape may.
The caller supplies the preexisting-allocation set; preparation cannot redefine
it, and may modify only fresh allocations while preserving all preexisting cells. -/
structure Prepared (α : Type) (input : Tensor) (supplied : Option Tensor)
    (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  batchShape : List Nat
  ncols : Nat
  normalized : Tensor
  flat : Matrix
  out : Matrix
  memory : PhysicalMemory α
  originalPreexisting : preexisting input.allocation
  suppliedPreexisting : ∀ t, supplied = some t → preexisting t.allocation
  preparationFrame : ∀ allocation a, preexisting allocation → memory allocation a = entry allocation a
  sourceShape : input.shape = batchShape ++ [2*ncols]
  normalizedShape : normalized.shape = input.shape
  normalizedDevice : normalized.cudaDevice = input.cudaDevice
  flatDevice : flat.cudaDevice = input.cudaDevice
  noCopyBranch : input.lastStride = 1 → normalized = input
  contiguousBranch : input.lastStride ≠ 1 → normalized.lastStride = 1
  normalizedValues : ∀ k, k < input.shape.prod →
    normalized.read memory k = input.read entry k
  flatRows : flat.rows = batchShape.prod
  flatCols : flat.cols = 2*ncols
  flatLastStride : flat.lastStride = 1
  reshapeValues : ∀ r q, r < batchShape.prod → q < 2*ncols →
    flat.read memory r q = normalized.read memory (r*(2*ncols)+q)
  emptyLike : supplied = none → out.rows = flat.rows ∧ out.cols = ncols ∧ out.lastStride = 1
  emptyLikeFresh : supplied = none → out.allocation ≠ flat.allocation
  emptyLikeNew : supplied = none → ¬ preexisting out.allocation
  suppliedReshape : ∀ t, supplied = some t → out.rows*out.cols = t.shape.prod ∧
    out.cols = t.shape.getLastD 0
  suppliedReshapeValues : ∀ t, supplied = some t → ∀ r q, r < out.rows → q < out.cols →
    out.read memory r q = t.read entry (r*out.cols+q)

/-- Actual source assertions. Shape is tested only on the supplied-output
branch, then last stride is tested on both branches. -/
def checks (p : Prepared α input supplied entry preexisting) : Bool :=
  (if supplied.isSome then decide (p.out.rows = p.flat.rows ∧ p.out.cols = p.ncols)
    else Bool.true) && decide (p.out.lastStride = 1)

theorem checks_metadata (p : Prepared α input supplied entry preexisting) (h : checks p = Bool.true) :
    p.out.rows = p.flat.rows ∧ p.out.cols = p.ncols ∧ p.out.lastStride = 1 := by
  cases supplied with
  | none => exact p.emptyLike rfl
  | some t =>
    have v : (p.out.rows = p.flat.rows ∧ p.out.cols = p.ncols) ∧ p.out.lastStride = 1 := by
      simpa [checks] using h
    exact ⟨v.1.1,v.1.2,v.2⟩

def Prepared.layout (p : Prepared α input supplied entry preexisting) : Layout :=
  ⟨"X","Y","OUT",p.batchShape.prod,p.ncols,p.flat.rowStride,p.flat.rowStride,p.out.rowStride⟩

/-- The first chunk and second chunk share flat.allocation; Y begins ncols
cells later in that same allocation. There is no artificial X/Y separation. -/
def Prepared.state [Inhabited α] (p : Prepared α input supplied entry preexisting) : State α where
  mem := fun region a => Cell.mk .real
    (if region = ("X" : RegionName) then p.memory p.flat.allocation (p.flat.base+a)
     else if region = ("Y" : RegionName) then p.memory p.flat.allocation (p.flat.base+p.ncols+a)
     else p.memory p.out.allocation (p.out.base+a))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

/-- Public semantic domain, not added Python assertions. Source pairing,
CUDA binding, memory bounds and writable ownership are explicit. Address/ABI
lowering to the target must implement these Nat addresses within its limits. -/
structure Domain (p : Prepared α input supplied entry preexisting) where
  positiveColumns : 0 < p.ncols
  sourceChecks : checks p = Bool.true
  sameDevice : p.out.cudaDevice = input.cudaDevice
  outputUnique : ∀ r q r' q', r < p.batchShape.prod → q < p.ncols →
    r' < p.batchShape.prod → q' < p.ncols →
    r*p.out.rowStride+q = r'*p.out.rowStride+q' → r = r' ∧ q = q'
  capacity : Nat → Nat
  originalBounds : ∀ k, k < input.shape.prod → input.offset k < capacity input.allocation
  flatBounds : ∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.flat.address r q < capacity p.flat.allocation
  outBounds : ∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.out.base+r*p.out.rowStride+q < capacity p.out.allocation
  outputOwnership : ∀ r q r' q', r < p.batchShape.prod → q < p.ncols →
    r' < p.batchShape.prod → q' < 2*p.ncols →
    (p.out.allocation,p.out.base+r*p.out.rowStride+q) ≠
      (p.flat.allocation,p.flat.address r' q')

theorem Domain.ownership (p : Prepared α input supplied entry preexisting) (d : Domain p) :
    Ownership p.layout := by
  refine ⟨by change ("X" : RegionName) ≠ "OUT"; decide,
    by change ("Y" : RegionName) ≠ "OUT"; decide,?_⟩
  exact d.outputUnique

theorem prepared_reads [Inhabited α] (p : Prepared α input supplied entry preexisting)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    ((p.state.mem p.layout.X (r*p.layout.sx+q)).read .real =
      input.read entry (r*(2*p.ncols)+q)) ∧
    ((p.state.mem p.layout.Y (r*p.layout.sy+q)).read .real =
      input.read entry (r*(2*p.ncols)+(p.ncols+q))) := by
  have hx := p.reshapeValues r q hr (by omega)
  have hy := p.reshapeValues r (p.ncols+q) hr (by omega)
  have rowBound : (r+1)*(2*p.ncols) ≤ p.batchShape.prod*(2*p.ncols) :=
    Nat.mul_le_mul_right _ (by omega)
  have boundX : r*(2*p.ncols)+q < input.shape.prod := by
    rw [p.sourceShape,List.prod_append]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    nlinarith
  have boundY : r*(2*p.ncols)+(p.ncols+q) < input.shape.prod := by
    rw [p.sourceShape,List.prod_append]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    nlinarith
  rw [p.normalizedValues _ boundX] at hx
  rw [p.normalizedValues _ boundY] at hy
  constructor
  · simpa [Prepared.state,Prepared.layout,Matrix.read,Matrix.address,p.flatLastStride,Nat.add_assoc] using hx
  · simpa [Prepared.state,Prepared.layout,Matrix.read,Matrix.address,p.flatLastStride,
      Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using hy

/-- Reinterpret final pointer-view memory in the actual output allocation.
Unwritten cells are transported too, and are later proved unchanged. -/
def Prepared.commit [Inhabited α] (p : Prepared α input supplied entry preexisting)
    (final : State α) : PhysicalMemory α :=
  fun allocation a => if allocation = p.out.allocation ∧ p.out.base ≤ a then
    (final.mem "OUT" (a-p.out.base)).read .real else p.memory allocation a

def Prepared.returnShape (p : Prepared α input supplied entry preexisting) : List Nat :=
  p.batchShape ++ [p.out.cols]

def entryRun [Inhabited α] (M : Algebra α) (p : Prepared α input supplied entry preexisting)
    (trials : List (Launch p.layout)) (selected : Launch p.layout) : Option (State α) :=
  if checks p then tunedLaunch M p.layout trials selected p.state else none

/-- Complete successful public path: optional input copy, flatten/chunk,
optional output reshape/allocate, both source assertions, all tuning trials,
selected 2D launch, and return reshape's complete logical tensor. -/
theorem entry_result [Inhabited α] (M : Algebra α) (p : Prepared α input supplied entry preexisting)
    (d : Domain p) (trials : List (Launch p.layout)) (selected : Launch p.layout) :
    ∃ final, entryRun M p trials selected = some final ∧
      p.returnShape = p.batchShape ++ [p.ncols] ∧
      (∀ r q, r < p.batchShape.prod → q < p.ncols →
        p.commit final p.out.allocation (p.out.base+r*p.out.rowStride+q) =
          scalar M (input.read entry (r*(2*p.ncols)+q))
            (input.read entry (r*(2*p.ncols)+(p.ncols+q)))) ∧
      (∀ region a, WholeOutside p.layout region a →
        final.mem region a = p.state.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := tuned_result M p.layout (Domain.ownership p d) trials selected p.state
  refine ⟨final,?_,?_,?_,frame⟩
  · simp only [entryRun,d.sourceChecks,if_true,ran]
  · simp only [Prepared.returnShape,(checks_metadata p d.sourceChecks).2.1]
  · intro r q hr hq
    have v := values r q hr hq
    have reads := prepared_reads p r q hr hq
    rw [reads.1,reads.2] at v
    have address : p.out.base+r*p.out.rowStride+q-p.out.base = r*p.out.rowStride+q := by omega
    unfold Prepared.commit
    rw [if_pos ⟨rfl, by omega⟩,address]
    change (final.mem p.layout.OUT (r*p.layout.so+q)).read .real = _
    rw [v]
    rfl

theorem failed_checks [Inhabited α] (M : Algebra α) (p : Prepared α input supplied entry preexisting)
    (trials : List (Launch p.layout)) (selected : Launch p.layout)
    (failed : checks p = Bool.false) : entryRun M p trials selected = none := by
  simp [entryRun,failed]

def PhysicalOutside (p : Prepared α input supplied entry preexisting) (allocation a : Nat) : Prop :=
  ∀ r q, r < p.batchShape.prod → q < p.ncols →
    (allocation,a) ≠ (p.out.allocation,p.out.base+r*p.out.rowStride+q)

theorem commit_frame [Inhabited α] (p : Prepared α input supplied entry preexisting) (final : State α)
    (frame : ∀ region a, WholeOutside p.layout region a → final.mem region a = p.state.mem region a)
    (allocation a : Nat) (outside : PhysicalOutside p allocation a) :
    p.commit final allocation a = p.memory allocation a := by
  unfold Prepared.commit
  split
  · rename_i h
    rcases h with ⟨same,base⟩
    subst allocation
    have miss : WholeOutside p.layout "OUT" (a-p.out.base) := by
      right
      intro r q hr hq eq
      apply outside r q hr hq
      congr 1
      dsimp [Prepared.layout] at eq
      omega
    rw [frame "OUT" (a-p.out.base) miss]
    change p.memory p.out.allocation (p.out.base+(a-p.out.base)) = p.memory p.out.allocation a
    congr 1
    omega
  · rfl

/-- Compose the kernel frame with the ordinary preparation frame. Fresh
allocations may contain copied input or uninitialized output; every cell in a
caller-declared preexisting allocation outside actual output remains at entry. -/
theorem entry_memory_frame [Inhabited α] (p : Prepared α input supplied entry preexisting)
    (final : State α)
    (frame : ∀ region a, WholeOutside p.layout region a → final.mem region a = p.state.mem region a)
    (allocation a : Nat) (existed : preexisting allocation)
    (outside : PhysicalOutside p allocation a) :
    p.commit final allocation a = entry allocation a :=
  (commit_frame p final frame allocation a outside).trans (p.preparationFrame allocation a existed)

/-- The original input is preexisting. If a copied input allows output to
alias the original allocation, only actual output cells may change. -/
theorem original_input_frame [Inhabited α] (p : Prepared α input supplied entry preexisting)
    (final : State α)
    (frame : ∀ region a, WholeOutside p.layout region a → final.mem region a = p.state.mem region a)
    (k : Nat) (outside : PhysicalOutside p input.allocation (input.offset k)) :
    input.read (p.commit final) k = input.read entry k :=
  entry_memory_frame p final frame _ _ p.originalPreexisting outside

theorem live_input_preserved [Inhabited α] (p : Prepared α input supplied entry preexisting)
    (d : Domain p) (final : State α)
    (frame : ∀ region a, WholeOutside p.layout region a → final.mem region a = p.state.mem region a)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < 2*p.ncols) :
    p.flat.read (p.commit final) r q = p.flat.read p.memory r q := by
  apply commit_frame p final frame
  intro ro qo hro hqo same
  exact d.outputOwnership ro qo r q hro hqo hr hq same.symm

theorem active_addresses (p : Prepared α input supplied entry preexisting) (d : Domain p)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    p.flat.base+r*p.flat.rowStride+q < d.capacity p.flat.allocation ∧
    p.flat.base+p.ncols+(r*p.flat.rowStride+q) < d.capacity p.flat.allocation ∧
    p.out.base+r*p.out.rowStride+q < d.capacity p.out.allocation := by
  refine ⟨?_,?_,d.outBounds r q hr hq⟩
  · simpa [Matrix.address,p.flatLastStride] using d.flatBounds r q hr (by omega)
  · simpa [Matrix.address,p.flatLastStride,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using
      d.flatBounds r (p.ncols+q) hr (by omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole

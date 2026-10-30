import bench.tritonbench_g.kcache_copy_triton.SignedCorrect
import bench.tritonbench_g.kcache_copy_triton.DiscreteCorrect

/-! Actual sequential execution of every program of the three-dimensional
logical launch. Every program observes the preceding program's complete
memory, including aliases with K, block tables and sequence lengths. This is
not a native concurrency or scheduling theorem. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.HostGrid
open VeriTile.Triton FP.Structural
open SignedCopy (Parameters)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

inductive Profile where
  | floating (storage : TypedCopy.Storage)
  | discrete (storage : SignedDiscreteCopy.Storage)
  deriving DecidableEq, Repr

def Profile.tag : Profile → TileDType
  | .floating d => d.tag
  | .discrete d => d.tag

def source (d : Profile) (p : Parameters) : ComputeKernel :=
  match d with
  | .floating d => SignedCopy.source d p
  | .discrete d => SignedDiscreteCopy.source d p

structure Grid where
  tokens : Nat
  heads : Nat
  splits : Nat
  deriving DecidableEq, Repr

abbrev Program := Nat × Nat × Nat

def ids (i : Program) : Nat → Nat :=
  fun dim => if dim = 0 then i.1 else if dim = 1 then i.2.1 else if dim = 2 then i.2.2 else 0

def dimensions (g : Grid) : Nat → Nat :=
  fun dim => if dim = 0 then g.tokens else if dim = 1 then g.heads else if dim = 2 then g.splits else 1

/-- All three grid coordinates, in one explicit sequential order. -/
def programs (g : Grid) : List Program :=
  (List.range g.tokens).flatMap fun t =>
    (List.range g.heads).flatMap fun h =>
      (List.range g.splits).map fun x => (t,h,x)

theorem mem_programs (g : Grid) (t h x : Nat) :
    (t,h,x) ∈ programs g ↔ t < g.tokens ∧ h < g.heads ∧ x < g.splits := by
  simp [programs, List.mem_flatMap, List.mem_map]

variable {α : Type} [Inhabited α]
abbrev Memory (α : Type) := RegionName → Nat → Cell α

def start (g : Grid) (i : Program) (previous : State α) : State α :=
  { previous with regs := fun _ _ _ => none, pids := ids i, numPids := dimensions g }

/-- Independent signed metadata and address reference; no execution witness
or interpreter is used by this memory specification. -/
def past (p : Parameters) (m : Memory α) (i : Program) : Int :=
  (m p.lengths.cast (i.1 / p.nTokens)).read .int +
    ((i.1 : Int) - (p.nTokens : Int) * (((i.1 / p.nTokens : Nat) : Int) + 1))

def blockId (p : Parameters) (m : Memory α) (i : Program) : Int :=
  (m p.blockTables.cast (i.1 / p.nTokens * p.strideBts +
    SignedCopy.unsigned64 ((past p m i).tdiv (p.blockSize : Int)) * p.strideBtb)).read .int

def inputOffset (p : Parameters) (i : Program) (lane : TileIndex [p.width]) : Nat :=
  i.1 * p.strideKt + i.2.1 * p.strideKh + (i.2.2 * p.width + lane.1.val) * p.strideKd

def outputOffset (p : Parameters) (m : Memory α) (i : Program)
    (lane : TileIndex [p.width]) : Nat :=
  SignedCopy.unsigned64 (blockId p m i) * p.strideKcb + i.2.1 * p.strideKch +
    i.2.2 * p.strideKcsplit + SignedCopy.unsigned64 ((past p m i).tmod (p.blockSize : Int)) * p.strideKcs + lane.1.val

def loaded (d : Profile) (p : Parameters) (m : Memory α) (i : Program)
    (lane : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((m p.K (inputOffset p i lane)).read d.tag)

def programMemory (d : Profile) (p : Parameters) (m : Memory α) (i : Program) : Memory α :=
  (TileShape.allIndices [p.width]).foldl (fun out lane => fun r o =>
    if r = p.KCache ∧ o = outputOffset p m i lane then loaded d p m i lane else out r o) m

def reference (d : Profile) (p : Parameters) : List Program → Memory α → Memory α
  | [], m => m
  | i :: rest, m => reference d p rest (programMemory d p m i)

def runPrograms (rest : Algebra α) (d : Profile) (p : Parameters) (g : Grid) :
    List Program → State α → Option (State α)
  | [], initial => some initial
  | i :: jobs, initial =>
      (FP.Structural.exec (TypedCopy.copyBackend rest) (source d p) (start g i initial)).bind
        (runPrograms rest d p g jobs)

def run (rest : Algebra α) (d : Profile) (p : Parameters) (g : Grid)
    (initial : State α) : Option (State α) :=
  runPrograms rest d p g (programs g) initial

private theorem input_correspondence (p : Parameters) (g : Grid) (i : Program)
    (s : State α) (lane : TileIndex [p.width]) :
    SignedCopy.inputOffset p (start g i s) lane = inputOffset p i lane := by
  simp [SignedCopy.inputOffset, start, ids, inputOffset]

private theorem output_correspondence (p : Parameters) (g : Grid) (i : Program)
    (s : State α) (lane : TileIndex [p.width]) :
    SignedCopy.outputOffset p (start g i s) lane = outputOffset p s.mem i lane := by
  simp [SignedCopy.outputOffset, SignedCopy.blockId, SignedCopy.past,
    SignedCopy.sequence, SignedCopy.tokenShift, start, ids, outputOffset, blockId, past]

theorem program_correct (rest : Algebra α) (d : Profile) (p : Parameters)
    (g : Grid) (i : Program) (initial : State α) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend rest) (source d p) (start g i initial) = some final ∧
      final.mem = programMemory d p initial.mem i := by
  cases d with
  | floating d =>
    refine ⟨(SignedCopy.execution rest d p (start g i initial)).val,
      (SignedCopy.execution rest d p (start g i initial)).property, ?_⟩
    rw [SignedCopy.execution_memory]
    simp only [SignedCopy.copied, TypedCopy.scatter_memory, programMemory,
      SignedCopy.loaded, input_correspondence, output_correspondence]
    rfl
  | discrete d =>
    refine ⟨(SignedDiscreteCopy.execution rest d p (start g i initial)).val,
      (SignedDiscreteCopy.execution rest d p (start g i initial)).property, ?_⟩
    rw [SignedDiscreteCopy.execution_memory]
    simp only [SignedDiscreteCopy.copied, TypedCopy.scatter_memory, programMemory,
      SignedDiscreteCopy.loaded, input_correspondence, output_correspondence]
    rfl

theorem programs_correct (rest : Algebra α) (d : Profile) (p : Parameters)
    (g : Grid) (jobs : List Program) (initial : State α) :
    ∃ final, runPrograms rest d p g jobs initial = some final ∧
      final.mem = reference d p jobs initial.mem := by
  induction jobs generalizing initial with
  | nil => exact ⟨initial,rfl,rfl⟩
  | cons i jobs ih =>
    obtain ⟨next, he, hm⟩ := program_correct rest d p g i initial
    obtain ⟨final, hr, hf⟩ := ih next
    refine ⟨final, ?_, ?_⟩
    · simp only [runPrograms, he, Option.bind_some]
      exact hr
    · simpa only [reference, hm] using hf

/-- Complete cell frame, evaluated against each preceding reference memory.
No disjointness or stable-metadata assumption is silently imposed. -/
def Unwritten (d : Profile) (p : Parameters) : List Program → Memory α → RegionName → Nat → Prop
  | [], _, _, _ => True
  | i :: jobs, m, r, o =>
      (r ≠ p.KCache ∨ ∀ lane : TileIndex [p.width], o ≠ outputOffset p m i lane) ∧
      Unwritten d p jobs (programMemory d p m i) r o

private theorem scatter_frame {I : Type} (xs : List I) (m : Memory α)
    (region : RegionName) (offset : I → Nat) (value : I → Cell α)
    (r : RegionName) (o : Nat) (miss : r ≠ region ∨ ∀ i, o ≠ offset i) :
    (xs.foldl (fun out i => fun r' o' =>
      if r' = region ∧ o' = offset i then value i else out r' o') m) r o = m r o := by
  induction xs generalizing m with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons, ih]
    have no : ¬(r = region ∧ o = offset i) := by
      intro both
      rcases miss with hr | ho
      · exact hr both.1
      · exact ho i both.2
    simp only [no, ↓reduceIte]

theorem reference_frame (d : Profile) (p : Parameters) (jobs : List Program)
    (m : Memory α) (r : RegionName) (o : Nat) (miss : Unwritten d p jobs m r o) :
    reference d p jobs m r o = m r o := by
  induction jobs generalizing m with
  | nil => rfl
  | cons i jobs ih =>
    change reference d p jobs (programMemory d p m i) r o = m r o
    rw [ih _ miss.2]
    exact scatter_frame _ _ _ _ _ _ _ miss.1

theorem grid_correct (rest : Algebra α) (d : Profile) (p : Parameters)
    (g : Grid) (initial : State α) :
    ∃ final, run rest d p g initial = some final ∧
      final.mem = reference d p (programs g) initial.mem ∧
      (∀ r o, Unwritten d p (programs g) initial.mem r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final,hr,hm⟩ := programs_correct rest d p g (programs g) initial
  refine ⟨final,hr,hm,?_⟩
  intro r o miss
  rw [hm]
  exact reference_frame d p _ initial.mem r o miss

#specNonCircular reference avoiding [source, run, runPrograms, FP.Structural.exec,
  SignedCopy.execution, SignedDiscreteCopy.execution]
#axiomsClean mem_programs
#axiomsClean program_correct
#axiomsClean programs_correct
#axiomsClean reference_frame
#axiomsClean grid_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.HostGrid

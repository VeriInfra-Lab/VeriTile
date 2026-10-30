import VeriTile.Triton.Memory.AlgorithmProgramIO

/-! Sequential Algorithm IO with explicit private launch initialization.

Some established source models clear private registers or select zero undef
for every CTA. These are closed data choices, not user-supplied state updates.
Execution and access safety follow the same actual source and predecessor.
Existing AlgorithmLaunch and its preserving semantics remain unchanged.
-/
namespace VeriTile.Triton
noncomputable section

structure PreparedAlgorithmInvocation where
  source : AlgorithmSource
  launch : AlgorithmLaunch
  clearRegisters : Bool := false
  zeroUndef : Bool := false

namespace PreparedAlgorithmInvocation
def initial (job : PreparedAlgorithmInvocation) (s : BlockState) : BlockState :=
  job.launch.initial {s with
    regs := if job.clearRegisters then fun _ _ _=>none else s.regs
    undef := if job.zeroUndef then fun _ _=>0 else s.undef}

@[simp] theorem initial_mem (job : PreparedAlgorithmInvocation) (s : BlockState) :
    (job.initial s).mem=s.mem := rfl

def run (job : PreparedAlgorithmInvocation) (s : BlockState) : Option BlockState :=
  exec job.source.kernel.toAlgKernel (job.initial s)

def Safe (job : PreparedAlgorithmInvocation) (bounds : RegionBounds) (s : BlockState) : Prop :=
  Kernel.TraceSafe bounds job.source.kernel.toAlgKernel (job.initial s)
end PreparedAlgorithmInvocation

def runPreparedAlgorithmSchedule : List PreparedAlgorithmInvocation → BlockState → Option BlockState
  | [],s=>some s
  | job::rest,s=>(job.run s).bind (runPreparedAlgorithmSchedule rest)

def PreparedAlgorithmScheduleSafe (bounds : RegionBounds) :
    List PreparedAlgorithmInvocation → BlockState → Prop
  | [],_=>True
  | job::rest,s=>job.Safe bounds s ∧
      ∀ t,job.run s=some t → PreparedAlgorithmScheduleSafe bounds rest t

/-- Every stage is a named source, fixed executor, and data-only launch.
Windows, typed logical binding and exact exterior frame are shared unchanged
with the existing AlgorithmProgramIO interface. -/
structure PreparedAlgorithmProgramIO extends AlgorithmMemoryIO where
  schedule : (∀ k,Fin (iarity k) → (ity k).carrier) → List PreparedAlgorithmInvocation

namespace PreparedAlgorithmProgramIO
abbrev Ctx (io : PreparedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.Ctx
abbrev InputBinding (io : PreparedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.InputBinding
abbrev InBounds (io : PreparedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.InBounds
abbrev OutputBinding (io : PreparedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.OutputBinding
abbrev Frame (io : PreparedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.Frame

/-- Source execution, typed logical outputs and exact initial-memory frame are
unconditional on capacities. Declared window bounds imply actual access safety
along the same deterministic sequential source execution. -/
def ImplementsOn (io : PreparedAlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ vals,domain vals → ∀ initial,io.InputBinding initial vals →
    (∀ bounds,io.InBounds bounds initial vals →
      PreparedAlgorithmScheduleSafe bounds (io.schedule vals) initial) ∧
    ∃ final,runPreparedAlgorithmSchedule (io.schedule vals) initial=some final ∧
      io.OutputBinding initial final vals (f vals) ∧ io.Frame initial final vals

def Implements (io : PreparedAlgorithmProgramIO)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn (fun _=>True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

theorem Implements.intro (io : PreparedAlgorithmProgramIO)
    {f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier}
    (safe : ∀ vals initial,io.InputBinding initial vals → ∀ bounds,
      io.InBounds bounds initial vals → PreparedAlgorithmScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial,io.InputBinding initial vals →
      ∃ final,runPreparedAlgorithmSchedule (io.schedule vals) initial=some final ∧
        io.OutputBinding initial final vals (f vals) ∧ io.Frame initial final vals) : io.Implements f := by
  intro vals _ initial inputs
  exact ⟨safe vals initial inputs,correct vals initial inputs⟩

end PreparedAlgorithmProgramIO
end
end VeriTile.Triton

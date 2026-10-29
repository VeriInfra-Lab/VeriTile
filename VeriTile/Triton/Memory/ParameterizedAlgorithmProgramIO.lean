import VeriTile.Triton.Memory.AlgorithmProgramIO

/-! Ordinary-grid Algorithm IO with explicit runtime register arguments.
Bindings are typed data, installed once by the existing `BlockState.setReg`.
The original source, same-predecessor ordinary grids, stage composition and
trace safety are unchanged. No state transformer or executor is an IO field.
-/
namespace VeriTile.Triton
noncomputable section

/-- One complete typed runtime argument. Duplicate names follow the existing
assignment semantics: the later binding replaces all earlier typed views. -/
structure AlgorithmRegisterBinding where
  name : RegName
  dtype : TileDType
  shape : TileShape
  value : Tile dtype shape

def bindAlgorithmArguments (args : List AlgorithmRegisterBinding) (s : BlockState) : BlockState :=
  args.foldl (fun state arg=>state.setReg arg.name arg.dtype arg.shape arg.value) s

@[simp] theorem bindAlgorithmArguments_nil (s : BlockState) : bindAlgorithmArguments [] s=s := rfl

@[simp] theorem bindAlgorithmArguments_singleton (arg : AlgorithmRegisterBinding) (s : BlockState) :
    bindAlgorithmArguments [arg] s=s.setReg arg.name arg.dtype arg.shape arg.value := rfl

theorem bindAlgorithmArguments_append (left right : List AlgorithmRegisterBinding) (s : BlockState) :
    bindAlgorithmArguments (left++right) s=bindAlgorithmArguments right (bindAlgorithmArguments left s) := by
  exact List.foldl_append

@[simp] theorem bindAlgorithmArguments_mem (args : List AlgorithmRegisterBinding) (s : BlockState) :
    (bindAlgorithmArguments args s).mem=s.mem := by
  induction args generalizing s with
  | nil => rfl
  | cons arg rest ih => exact ih (s.setReg arg.name arg.dtype arg.shape arg.value)

@[simp] theorem bindAlgorithmArguments_pids (args : List AlgorithmRegisterBinding) (s : BlockState) :
    (bindAlgorithmArguments args s).pids=s.pids := by
  induction args generalizing s with
  | nil => rfl
  | cons arg rest ih => exact ih (s.setReg arg.name arg.dtype arg.shape arg.value)

@[simp] theorem bindAlgorithmArguments_numPids (args : List AlgorithmRegisterBinding) (s : BlockState) :
    (bindAlgorithmArguments args s).numPids=s.numPids := by
  induction args generalizing s with
  | nil => rfl
  | cons arg rest ih => exact ih (s.setReg arg.name arg.dtype arg.shape arg.value)

@[simp] theorem bindAlgorithmArguments_undef (args : List AlgorithmRegisterBinding) (s : BlockState) :
    (bindAlgorithmArguments args s).undef=s.undef := by
  induction args generalizing s with
  | nil => rfl
  | cons arg rest ih => exact ih (s.setReg arg.name arg.dtype arg.shape arg.value)

/-- Installing program coordinates does not change runtime arguments. Each
ordinary CTA therefore receives the same prepared predecessor memory/regs. -/
theorem bindAlgorithmArguments_withGridIndex (args : List AlgorithmRegisterBinding)
    (s : BlockState) {g : Grid} (idx : GridIndex g) :
    bindAlgorithmArguments args (s.withGridIndex idx)=(bindAlgorithmArguments args s).withGridIndex idx := by
  induction args generalizing s with
  | nil => rfl
  | cons arg rest ih => exact ih (s.setReg arg.name arg.dtype arg.shape arg.value)

/-- Extends the existing ordinary/single-stage interface with data-only
register arguments. Sources still read their original runtime references. -/
structure ParameterizedAlgorithmProgramIO extends AlgorithmProgramIO where
  arguments : (∀ k,Fin (iarity k) → (ity k).carrier) → List AlgorithmRegisterBinding

namespace ParameterizedAlgorithmProgramIO
abbrev Ctx (io : ParameterizedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.Ctx
abbrev InputBinding (io : ParameterizedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.InputBinding
abbrev InBounds (io : ParameterizedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.InBounds
abbrev OutputBinding (io : ParameterizedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.OutputBinding
abbrev Frame (io : ParameterizedAlgorithmProgramIO) := io.toAlgorithmMemoryIO.Frame

/-- Both closed preparations happen once, before the first stage. Subsequent
stages use their actual predecessors, without reinstalling argument values. -/
def initial (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) : BlockState :=
  bindAlgorithmArguments (io.arguments vals) (io.initialUndef.initial s)

@[simp] theorem initial_mem (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) :
    (io.initial vals s).mem=s.mem := by simp [initial]

@[simp] theorem initial_pids (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) :
    (io.initial vals s).pids=s.pids := by simp [initial]

@[simp] theorem initial_numPids (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) :
    (io.initial vals s).numPids=s.numPids := by simp [initial]

@[simp] theorem initial_undef (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) :
    (io.initial vals s).undef=(io.initialUndef.initial s).undef := by simp [initial]

theorem initial_withGridIndex (io : ParameterizedAlgorithmProgramIO) (vals : io.Ctx)
    (s : BlockState) {g : Grid} (idx : GridIndex g) :
    io.initial vals (s.withGridIndex idx)=(io.initial vals s).withGridIndex idx := by
  have undef : io.initialUndef.initial (s.withGridIndex idx)=
      (io.initialUndef.initial s).withGridIndex idx := by cases io.initialUndef <;> rfl
  simp only [initial,undef,bindAlgorithmArguments_withGridIndex]

/-- Success, logical outputs and exact original-memory frame are independent
of buffer capacities. Bounds imply safety of this same source execution trace,
including each ordinary CTA on the common prepared stage predecessor. -/
def ImplementsOn3D (io : ParameterizedAlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ vals,domain vals → ∀ original,io.InputBinding original vals →
    ∃ final, ∃ trace : AlgorithmExecution (io.stages vals) (io.initial vals original) final,
      (∀ bounds,io.InBounds bounds original vals → trace.Safe bounds) ∧
      io.OutputBinding original final vals (f (original.pids 0) (original.pids 1) (original.pids 2) vals) ∧
      io.Frame original final vals

def ImplementsOn (io : ParameterizedAlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn3D domain (fun _ _ _=>f)

def Implements3D (io : ParameterizedAlgorithmProgramIO)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn3D (fun _=>True) f

def Implements (io : ParameterizedAlgorithmProgramIO)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn (fun _=>True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

theorem ImplementsOn3D.intro (io : ParameterizedAlgorithmProgramIO) {domain : io.Ctx → Prop}
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier}
    (proof : ∀ vals,domain vals → ∀ original,io.InputBinding original vals →
      ∃ final, ∃ trace : AlgorithmExecution (io.stages vals) (io.initial vals original) final,
        (∀ bounds,io.InBounds bounds original vals → trace.Safe bounds) ∧
        io.OutputBinding original final vals (f (original.pids 0) (original.pids 1) (original.pids 2) vals) ∧
        io.Frame original final vals) : io.ImplementsOn3D domain f := proof

/-- Existing AlgorithmProgramIO remains the empty-argument special case. -/
def ofProgram (io : AlgorithmProgramIO) : ParameterizedAlgorithmProgramIO :=
  {io with arguments:=fun _=>[]}

@[simp] theorem ofProgram_initial (io : AlgorithmProgramIO) (vals : io.Ctx) (s : BlockState) :
    (ofProgram io).initial vals s=io.initialUndef.initial s := rfl

@[simp] theorem ofProgram_implementsOn3D (io : AlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) :
    (ofProgram io).ImplementsOn3D domain f ↔ io.ImplementsOn3D domain f := Iff.rfl

@[simp] theorem ofProgram_implements (io : AlgorithmProgramIO)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) :
    (ofProgram io).Implements f ↔ io.Implements f := Iff.rfl

end ParameterizedAlgorithmProgramIO
end
end VeriTile.Triton

import VeriTile.Triton.Memory.RegionalScheduledKernelIO
import VeriTile.Triton.Launch.Composition

/-! Source-connected IO for lists of actual Algorithm stages. A stage is either
one explicit source launch, or the existing ordinary independent-grid relation.
The latter retains same-initial-state program reads and mergeFrames; it is not
silently replaced by serial program execution. Every stage receives the actual
previous stage result. No executor, state-preparation or safety callback occurs
in the public data. Safety is proved for the same execution witness as outputs.
-/
namespace VeriTile.Triton
noncomputable section

/-- An actual source with a checked Algorithm projection. Callers explicitly
name dtype erasure where Real interpretation requires it. -/
structure AlgorithmSource where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]

/-- Exactly two existing execution forms; no generic concurrency model. -/
inductive AlgorithmStage where
  | single (source : AlgorithmSource) (launch : AlgorithmLaunch)
  | ordinary (source : AlgorithmSource) (grid : Grid)

namespace AlgorithmStage
/-- Ordinary stages use the already established source-derived disjoint frames
and merge relation. Single stages use the fixed identity-rounding executor. -/
def Runs : AlgorithmStage → BlockState → BlockState → Prop
  | .single source launch,s,t => execR .triv source.kernel.toAlgKernel (launch.initial s)=some t
  | .ordinary source grid,s,t => Nonempty (Kernel.GridLaunchedOrdinary source.kernel.toAlgKernel grid s t)

/-- Source access safety of the existing nonrounding Algorithm executor.
This deliberately uses `TraceSafe`, as in `RegionalKernelIO`: it is not a claim
that `TraceSafe` and `TraceSafeR .triv` are equivalent (their pointer-subtraction
and global pointer-memory obligations differ). The single-stage execution is
connected to this executor by `execR_triv`. -/
def Safe (bounds : RegionBounds) : AlgorithmStage → BlockState → Prop
  | .single source launch,s => Kernel.TraceSafe bounds source.kernel.toAlgKernel (launch.initial s)
  | .ordinary source grid,s => ∀ idx : GridIndex grid,
      Kernel.TraceSafe bounds source.kernel.toAlgKernel (s.withGridIndex idx)

@[simp] theorem single_runs (source : AlgorithmSource) (launch : AlgorithmLaunch) (s t : BlockState) :
    (.single source launch : AlgorithmStage).Runs s t ↔
      exec source.kernel.toAlgKernel (launch.initial s)=some t := by
  simp only [Runs,execR_triv]

@[simp] theorem ordinary_runs (source : AlgorithmSource) (grid : Grid) (s t : BlockState) :
    (.ordinary source grid : AlgorithmStage).Runs s t ↔
      Nonempty (Kernel.GridLaunchedOrdinary source.kernel.toAlgKernel grid s t) := Iff.rfl
end AlgorithmStage

/-- A concrete finite execution witness retains all actual intermediate
states, so the safety proof cannot choose different stage predecessors. -/
inductive AlgorithmExecution : List AlgorithmStage → BlockState → BlockState → Type where
  | nil (s : BlockState) : AlgorithmExecution [] s s
  | cons {stage : AlgorithmStage} {rest : List AlgorithmStage} {s middle final : BlockState}
      (head : stage.Runs s middle) (tail : AlgorithmExecution rest middle final) :
      AlgorithmExecution (stage::rest) s final

namespace AlgorithmExecution

def Safe {stages : List AlgorithmStage} {s t : BlockState}
    (trace : AlgorithmExecution stages s t) (bounds : RegionBounds) : Prop :=
  match trace with
  | .nil _ => True
  | @cons stage _ initial _ _ _ tail => stage.Safe bounds initial ∧ tail.Safe bounds

def append {left right : List AlgorithmStage} {s middle final : BlockState}
    (first : AlgorithmExecution left s middle) (next : AlgorithmExecution right middle final) :
    AlgorithmExecution (left++right) s final := by
  induction first with
  | nil => exact next
  | cons head tail ih => exact .cons head (ih next)

theorem safe_append {left right : List AlgorithmStage} {s middle final : BlockState}
    (first : AlgorithmExecution left s middle) (next : AlgorithmExecution right middle final)
    (bounds : RegionBounds) : (first.append next).Safe bounds ↔ first.Safe bounds ∧ next.Safe bounds := by
  induction first with
  | nil => simp [append,Safe]
  | @cons stage rest initial middle finish head tail ih =>
    change (stage.Safe bounds initial ∧ (tail.append next).Safe bounds) ↔
      (stage.Safe bounds initial ∧ tail.Safe bounds) ∧ next.Safe bounds
    rw [ih next]
    exact and_assoc.symm

@[simp] theorem nil_final {s t : BlockState} (trace : AlgorithmExecution [] s t) : t=s := by
  cases trace
  rfl
end AlgorithmExecution

/-- Existing fixed undef conventions, applied once before stages. Memory,
registers, program IDs and all other state fields are preserved. -/
inductive AlgorithmInitialUndef where
  | preserve
  | zero
  deriving DecidableEq, Repr

namespace AlgorithmInitialUndef
def initial : AlgorithmInitialUndef → BlockState → BlockState
  | .preserve,s => s
  | .zero,s => {s with undef:=fun _ _=>0}
@[simp] theorem initial_mem (mode : AlgorithmInitialUndef) (s : BlockState) :
    (mode.initial s).mem=s.mem := by cases mode <;> rfl
@[simp] theorem initial_regs (mode : AlgorithmInitialUndef) (s : BlockState) :
    (mode.initial s).regs=s.regs := by cases mode <;> rfl
@[simp] theorem initial_pids (mode : AlgorithmInitialUndef) (s : BlockState) :
    (mode.initial s).pids=s.pids := by cases mode <;> rfl
@[simp] theorem initial_numPids (mode : AlgorithmInitialUndef) (s : BlockState) :
    (mode.initial s).numPids=s.numPids := by cases mode <;> rfl
end AlgorithmInitialUndef

/-- Source-free typed regional windows. ChanTy and its operational readers are
shared with UKernelIO, but no unrelated single-source kernel is inherited. -/
structure AlgorithmMemoryIO where
  /-- Number of input channels. -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- Number of scratch (private working) channels. -/
  nScr : Nat
  /-- Declared buffers, separate from channel roles so duplicate-region
  (in-place) wiring remains expressible. No allocator execution is modeled. -/
  bufs : List RegionName
  /-- Each input channel's element type. -/
  ity : Fin nIn → ChanTy
  /-- Each input channel's lane count (scalar slots are 1-lane). -/
  iarity : Fin nIn → Nat
  /-- Each input channel's buffer. -/
  ibuf : Fin nIn → RegionName
  /-- Each output channel's element type. Symmetric to `ity`: the
  postcondition reads output channel `o` back through `(oty o).read`, so a
  kernel whose store is `.to(tl.int8)`-style typed can state a *true* value
  contract. `.float` reads are definitionally `readMem`, so every skin that
  sets `oty := fun _ => .float` keeps its `Implements` statement unchanged. -/
  oty : Fin nOut → ChanTy
  /-- Each output channel's lane count. -/
  oarity : Fin nOut → Nat
  /-- Each output channel's buffer. -/
  obuf : Fin nOut → RegionName
  /-- Every output channel names a declared buffer. -/
  obuf_mem : ∀ o, obuf o ∈ bufs
  /-- Each scratch channel's lane count. -/
  sarity : Fin nScr → Nat
  /-- Each scratch channel's buffer. -/
  sbuf : Fin nScr → RegionName
  /-- Input channel `i`'s per-lane read address — may depend on the pinned
  values of every channel (the data-dependent genre). -/
  iwin : (i : Fin nIn) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (iarity i) → Nat
  /-- Input channel `i`'s read-active lanes. -/
  imask : (i : Fin nIn) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (iarity i) → Prop
  /-- Output channel `o`'s per-lane write address. -/
  owin : (o : Fin nOut) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (oarity o) → Nat
  /-- Output channel `o`'s write-active lanes. -/
  omask : (o : Fin nOut) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (oarity o) → Prop
  /-- Scratch channel `t`'s per-lane write address. -/
  swin : (t : Fin nScr) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (sarity t) → Nat
  /-- Scratch channel `t`'s write-active lanes. -/
  smask : (t : Fin nScr) → (∀ k, Fin (iarity k) → (ity k).carrier) →
    Nat → Nat → Nat → Fin (sarity t) → Prop

namespace AlgorithmMemoryIO
abbrev Ctx (io : AlgorithmMemoryIO) : Type :=
  ∀ k : Fin io.nIn, Fin (io.iarity k) → (io.ity k).carrier
/-- Initial typed values at every active declared input address. -/
def InputBinding (io : AlgorithmMemoryIO) (s : BlockState) (vals : io.Ctx) : Prop :=
  ∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
    io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    (io.ity i).read s (io.ibuf i)
      (io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals i j

/-- Every active declared input/output/private-scratch window is in bounds. -/
def InBounds (io : AlgorithmMemoryIO) (bounds : RegionBounds)
    (s : BlockState) (vals : io.Ctx) : Prop :=
  (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
    io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.ibuf i)) ∧
  (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
    io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.obuf o)) ∧
  (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
    io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.sbuf t))

/-- The exact raw-cell frame excludes only active output and scratch cells. -/
def Frame (io : AlgorithmMemoryIO) (s final : BlockState) (vals : io.Ctx) : Prop :=
  ∀ r address,
    (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
      io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      r ≠ io.obuf o ∨ address ≠ io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) →
    (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
      io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      r ≠ io.sbuf t ∨ address ≠ io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j) →
    final.mem r address = s.mem r address


/-- Typed active outputs; the raw exterior frame is a separate exact claim. -/
def OutputBinding (io : AlgorithmMemoryIO) (initial final : BlockState) (vals : io.Ctx)
    (expected : (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ o j, io.omask o vals (initial.pids 0) (initial.pids 1) (initial.pids 2) j →
    (io.oty o).read final (io.obuf o)
      (io.owin o vals (initial.pids 0) (initial.pids 1) (initial.pids 2) j)=expected o j
end AlgorithmMemoryIO

/-- All executed sources are named in stages; initializer is a closed data
choice and cannot edit input/output memory or arbitrary local registers. -/
structure AlgorithmProgramIO extends AlgorithmMemoryIO where
  initialUndef : AlgorithmInitialUndef := .preserve
  stages : (∀ k, Fin (iarity k) → (ity k).carrier) → List AlgorithmStage

namespace AlgorithmProgramIO
abbrev Ctx (io : AlgorithmProgramIO) := io.toAlgorithmMemoryIO.Ctx
abbrev InputBinding (io : AlgorithmProgramIO) := io.toAlgorithmMemoryIO.InputBinding
abbrev InBounds (io : AlgorithmProgramIO) := io.toAlgorithmMemoryIO.InBounds
abbrev OutputBinding (io : AlgorithmProgramIO) := io.toAlgorithmMemoryIO.OutputBinding
abbrev Frame (io : AlgorithmProgramIO) := io.toAlgorithmMemoryIO.Frame

/-- Total source execution, independent typed outputs and exact original-memory
frame. Bounds condition only safety of the same execution witness. Logical
input-domain conditions never replace a missing source-safety proof. -/
def ImplementsOn3D (io : AlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ vals,domain vals → ∀ initial,io.InputBinding initial vals →
    ∃ final, ∃ trace : AlgorithmExecution (io.stages vals) (io.initialUndef.initial initial) final,
      (∀ bounds,io.InBounds bounds initial vals → trace.Safe bounds) ∧
      io.OutputBinding initial final vals (f (initial.pids 0) (initial.pids 1) (initial.pids 2) vals) ∧
      io.Frame initial final vals

def ImplementsOn (io : AlgorithmProgramIO) (domain : io.Ctx → Prop)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn3D domain (fun _ _ _=>f)

def Implements3D (io : AlgorithmProgramIO)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn3D (fun _=>True) f

/-- The actual stage schedule safely produces the specified logical outputs
for every bound input, preserving the exact exterior memory frame. -/
def Implements (io : AlgorithmProgramIO)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.ImplementsOn (fun _=>True) f

@[inherit_doc] scoped infix:25 " ⊨ " => AlgorithmProgramIO.Implements
scoped notation:25 io " ⊨{" domain "} " f => AlgorithmProgramIO.ImplementsOn io domain f

theorem ImplementsOn3D.intro (io : AlgorithmProgramIO) {domain : io.Ctx → Prop}
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier}
    (proof : ∀ vals,domain vals → ∀ initial,io.InputBinding initial vals →
      ∃ final, ∃ trace : AlgorithmExecution (io.stages vals) (io.initialUndef.initial initial) final,
        (∀ bounds,io.InBounds bounds initial vals → trace.Safe bounds) ∧
        io.OutputBinding initial final vals (f (initial.pids 0) (initial.pids 1) (initial.pids 2) vals) ∧
        io.Frame initial final vals) : io.ImplementsOn3D domain f := proof

end AlgorithmProgramIO
end
end VeriTile.Triton

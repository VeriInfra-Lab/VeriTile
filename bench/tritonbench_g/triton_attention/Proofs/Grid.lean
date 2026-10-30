import bench.tritonbench_g.triton_attention.Kernels
import bench.tritonbench_g.triton_attention.Proofs.FPBackwardFrame
import bench.tritonbench_g.triton_attention.Proofs.FPForwardReadback

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

abbrev Program := Nat × Nat
abbrev Memory (α : Type) := RegionName → Nat → Cell α

def ids (pid : Program) (ax : Nat) : Nat :=
  if ax = 0 then pid.1 else if ax = 1 then pid.2 else 0
def entry {α : Type} [Inhabited α] (grid pid : Program) (mem : Memory α) : State α where
  mem := mem
  regs := fun _ _ _ => none
  pids := ids pid
  numPids := fun ax => if ax = 0 then grid.1 else if ax = 1 then grid.2 else 1
  undef := fun d _ _ => defaultValue d

def execute {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (code : ComputeKernel) (grid pid : Program) (mem : Memory α) : Option (Memory α) :=
  (FP.Structural.exec (backend M P) code (entry grid pid mem)).map State.mem

def forwardGrid (c : Shape) : Program := (c.blocks,c.groups)
def preprocessGrid (c : Shape) : Program := (c.blocks*c.groups,1)
def backwardGrid (c : Shape) : Program := (c.groups,1)

def programs (grid : Program) : List Program :=
  (List.range grid.1).flatMap (fun x => (List.range grid.2).map (fun y => (x,y)))

/-- Input-derived operation recurrences with explicit stores, independent of
the source interpreter, control flow, and its successful-execution result. -/
def forwardStep {α : Type} [Inhabited α] (P : Primitives α) (c : Shape)
    (scale : ℝ) (pid : Program) (mem : Memory α) : Memory α :=
  let s := entry (forwardGrid c) pid mem
  (forwardWrite P (forwardParameters c scale) s s).mem
def preprocessStep {α : Type} [Inhabited α] (P : Primitives α) (c : Shape)
    (pid : Program) (mem : Memory α) : Memory α :=
  let s := entry (preprocessGrid c) pid mem
  (preWrite P (preParameters c) s s).mem
def backwardStep {α : Type} [Inhabited α] (P : Primitives α) (c : Shape)
    (scale : ℝ) (pid : Program) (mem : Memory α) : Memory α :=
  backwardFinalMemory P (backwardParameters c scale) (entry (backwardGrid c) pid mem)

theorem forward_success {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α) :
    execute M P (forwardKernel c scale) (forwardGrid c) pid mem =
      some (forwardStep P c scale pid mem) := by
  obtain ⟨final,run,values⟩ := forwardFullExecution M P (forwardParameters c scale)
    (entry (forwardGrid c) pid mem) (by change 0 < 128; decide)
    (by dsimp [forwardStop,forwardParameters]; exact dvd_mul_left 128 _)
  simp only [execute,forwardKernel,run,Option.map_some,values,forwardStep]

theorem preprocess_success {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (c : Shape) (pid : Program) (mem : Memory α) :
    execute M P (preprocessKernel c) (preprocessGrid c) pid mem =
      some (preprocessStep P c pid mem) := by
  unfold execute preprocessKernel
  rw [preprocessExecution]
  simp only [Option.map_some]
  congr 1
  apply preWrite_memory
  simp [preRegisters]

theorem backward_success {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α) :
    execute M P (backwardKernel c scale) (backwardGrid c) pid mem =
      some (backwardStep P c scale pid mem) := by
  obtain ⟨final,run,values⟩ := backwardFullExecution M P (backwardParameters c scale)
    (entry (backwardGrid c) pid mem) (by change 0 < 128; decide)
  simp only [execute,backwardKernel,run,Option.map_some,values,backwardStep]

def run {α : Type} (step : Program → Memory α → Option (Memory α)) :
    List Program → Memory α → Option (Memory α)
  | [],mem => some mem
  | p::ps,mem => (step p mem).bind (run step ps)

theorem run_eq_fold {α : Type} (step : Program → Memory α → Option (Memory α))
    (reference : Program → Memory α → Memory α)
    (step_eq : ∀ p mem, step p mem = some (reference p mem))
    (work : List Program) (mem : Memory α) :
    run step work mem = some (work.foldl (fun m p => reference p m) mem) := by
  induction work generalizing mem with
  | nil => rfl
  | cons p ps ih => simp only [run,step_eq,Option.bind_some,List.foldl_cons,ih]

/-- All three shared source grids execute against the memory produced by the
preceding CTA/stage. Scratch allocation and initial zero-DQ are wrapper facts. -/
def wholeRun {α : Type} [Inhabited α] (M : Algebra α)
    (F P B : Primitives α) (c : Shape) (scale : ℝ)
    (fw pw bw : List Program) (mem : Memory α) : Option (Memory α) := do
  let afterForward ← run (execute M F (forwardKernel c scale) (forwardGrid c)) fw mem
  let afterPreprocess ← run (execute M P (preprocessKernel c) (preprocessGrid c)) pw afterForward
  run (execute M B (backwardKernel c scale) (backwardGrid c)) bw afterPreprocess

def wholeReference {α : Type} [Inhabited α] (F P B : Primitives α)
    (c : Shape) (scale : ℝ) (fw pw bw : List Program) (mem : Memory α) : Memory α :=
  let f := fw.foldl (fun m pid => forwardStep F c scale pid m) mem
  let p := pw.foldl (fun m pid => preprocessStep P c pid m) f
  bw.foldl (fun m pid => backwardStep B c scale pid m) p

theorem whole_success {α : Type} [Inhabited α] (M : Algebra α)
    (F P B : Primitives α) (c : Shape) (scale : ℝ)
    (fw pw bw : List Program) (mem : Memory α) :
    wholeRun M F P B c scale fw pw bw mem =
      some (wholeReference F P B c scale fw pw bw mem) := by
  simp only [wholeRun,run_eq_fold _ _ (forward_success M F c scale),Option.bind_some,
    run_eq_fold _ _ (preprocess_success M P c),run_eq_fold _ _ (backward_success M B c scale),
    wholeReference]
  rfl

#axiomsClean forward_success
#axiomsClean preprocess_success
#axiomsClean backward_success
#axiomsClean whole_success
#specNonCircular wholeReference avoiding [wholeRun,execute,FP.Structural.exec,run]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid

import bench.tritonbench_g.mean_reduction.Kernels
import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_Source
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.RegionalScheduledKernelIO
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import Mathlib.Logic.Equiv.Fin.Basic

namespace VeriTile.Bench.Repairs.MeanReduction.MeanIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000

/-- Finite compressed row-major input and one output per row. Structural
cells use the erased real tag; every source load, cast, accumulator operation,
reduction and division retains its explicit FP32 precision in `kernel`. -/
def inputChannels (M N : Nat) : List IOChannel := [⟨.real,[M*N],"X"⟩]
def outputChannels (M : Nat) : List IOChannel := [⟨.real,[M],"Mean"⟩]
abbrev Inputs (α : Type) (M N : Nat) := ChannelValues α (inputChannels M N)
abbrev Outputs (α : Type) (M : Nat) := ChannelValues α (outputChannels M)

def element (values : Inputs α M N) (i : Fin (M*N)) : α :=
  values ⟨0,by simp [inputChannels]⟩ (i,PUnit.unit)

theorem rowMajor_bound {row col M N : Nat} (hr : row<M) (hc : col<N) : row*N+col<M*N := by
  calc
    row*N+col < row*N+N := Nat.add_lt_add_left hc _
    _ = (row+1)*N := by simp [Nat.add_mul]
    _ ≤ M*N := Nat.mul_le_mul_right N hr

abbrev Accumulator (α : Type) := Values α .real [8,8]

/-- Independent logical FP32 chunk: inactive rows/columns use the source zero
fallback, then undergo the same explicit FP32 conversion as loaded values. -/
def chunk (A : Algebra α) (M N pid off : Nat) (values : Inputs α M N) : Accumulator α :=
  fun i => A.cast (some .fp32) .real .real
    (if live : pid*8+i.1.val<M ∧ off+i.2.1.val<N then
      element values ⟨(pid*8+i.1.val)*N+(off+i.2.1.val), rowMajor_bound live.1 live.2⟩
    else A.literal (some .fp32) .real 0)

/-- Source-order streaming additions, expressed on logical arrays. The
backend retains the chosen reduction tree; no reassociation law is assumed. -/
def accumulated (A : Algebra α) (M N pid : Nat) (values : Inputs α M N)
    (off : Nat) (acc : Accumulator α) : Accumulator α :=
  if off<N then accumulated A M N pid values (off+8)
    (fun i => A.binary (some .fp32) .real .add (acc i) (chunk A M N pid off values i))
  else acc
termination_by N-off

def nativeRow (family : Nat → Algebra α) (M N : Nat) (values : Inputs α M N)
    (row : Fin M) : α :=
  let A := family (row.val/8)
  let acc := accumulated A M N (row.val/8) values 0
    (fun _ => A.literal (some .fp32) .real 0)
  A.binary (some .fp32) .real .div
    (A.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false acc (⟨row.val%8,by omega⟩,PUnit.unit))
    (A.literal (some .fp32) .real (N:ℝ))

def nativeTarget (family : Nat → Algebra α) (M N : Nat)
    (values : Inputs α M N) : Outputs α M :=
  Fin.cases (fun i => nativeRow family M N values i.1) (fun o => Fin.elim0 o)

def memory (M N : Nat) : MemoryIO α where
  inputs := inputChannels M N
  outputs := outputChannels M
  buffers := ["X","Mean"]
  read := Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun o => Fin.elim0 o)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Original per-program backend and source; only axis zero is overwritten,
matching the existing full-grid launch semantics. -/
def invocation (family : Nat → Algebra α) (variant : Bool) (M N pid : Nat) : Invocation α where
  kernel := if variant then indexCommutedKernel M N else kernel M N
  backend := family pid
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

def nativeIO (family : Nat → Algebra α) (variant : Bool) (M N : Nat)
    (jobs : List Nat) : ProgramIO α α where
  toMemoryIO := memory M N
  schedule := fun _ => jobs.map (invocation family variant M N)
  encode := id
  observe := fun _ _ => id

def candidate (family : Nat → Algebra α) (M N : Nat) (jobs : List Nat) :
    ProgramIO.Variant (nativeIO family Bool.false M N jobs) where
  schedule := fun _ => jobs.map (invocation family Bool.true M N)
  observe := fun _ _ => id

/-- Public program views retain both complete actual invocation schedules. -/
def original (family : Nat → Algebra α) (M N : Nat) (jobs : List Nat) :=
  ProgramIO.Implementation.original (nativeIO family Bool.false M N jobs) (fun _ => True)

def indexCommuted (family : Nat → Algebra α) (M N : Nat) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (candidate family M N jobs) (fun _ => True)

/-- The Real interpretation of the same actual FP32 source. Its finite
channels cover the entire compressed input and output, independently of PID;
each launched program retains the original row/column masks internally. -/
def realIO (M N : Nat) : UKernelIO where
  kernel := (kernel M N).eraseDType
  projection := VeriTile.Bench.TritonBenchG.MeanReduction.typed_supported
    Bool.false "X" "Mean" M N 8 8
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := ["X","Mean"]
  ity := fun _ => .float
  iarity := fun _ => M*N
  ibuf := fun _ => "X"
  oty := fun _ => .float
  oarity := fun _ => M
  obuf := fun _ => "Mean"
  obuf_mem := fun _ => by simp
  sarity := fun k => Fin.elim0 k
  sbuf := fun k => Fin.elim0 k
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun k => Fin.elim0 k
  smask := fun k => Fin.elim0 k

/-- Independent finite row mean, retaining Real division at N=0. -/
def mathematical (M N : Nat) (values : Fin 1 → Fin (M*N) → ℝ)
    (_ : Fin 1) (row : Fin M) : ℝ :=
  (∑ j : Fin N,values 0
    ⟨row.val*N+j.val,rowMajor_bound row.isLt j.isLt⟩)/(N:ℝ)

def realLaunch (pid : Nat) : AlgorithmLaunch where
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

/-- Full explicit grid order; only program-ID axis zero is overwritten. -/
def realSchedule (M N : Nat) (jobs : List Nat) (_ : (realIO M N).Ctx) : List AlgorithmLaunch :=
  jobs.map realLaunch

/-- The single flat input channel viewed as a logical row/column matrix.
The inverse also covers empty dimensions; no positivity assumption is added. -/
def matrixInputs (M N : Nat) : (realIO M N).Ctx ≃ (Fin M → Fin N → ℝ) := by
  change (Fin 1 → Fin (M*N) → ℝ) ≃ (Fin M → Fin N → ℝ)
  exact {
    toFun := fun values row col => values 0 (finProdFinEquiv (row, col))
    invFun := fun xs _ i => xs (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
    left_inv := by
      intro values
      funext channel i
      have channel_zero : channel = 0 := Fin.eq_zero channel
      subst channel
      exact congrArg (values 0) (finProdFinEquiv.apply_symm_apply i)
    right_inv := by
      intro xs
      funext row col
      change xs (finProdFinEquiv.symm (finProdFinEquiv (row, col))).1
        (finProdFinEquiv.symm (finProdFinEquiv (row, col))).2 = xs row col
      rw [finProdFinEquiv.symm_apply_apply] }

/-- Every output cell of the sole output channel is exactly one logical row. -/
def rowOutputs (M N : Nat) :
    ((o : Fin (realIO M N).nOut) × Fin ((realIO M N).oarity o)) ≃ Fin M := by
  change ((o : Fin 1) × Fin M) ≃ Fin M
  exact {
    toFun := fun cell => cell.2
    invFun := fun row => ⟨0, row⟩
    left_inv := by
      rintro ⟨channel, row⟩
      have channel_zero : channel = 0 := Fin.eq_zero channel
      subst channel
      rfl
    right_inv := fun _ => rfl }

/-- Logical matrix input, one result per row, and the complete original grid.
The underlying memory windows and schedule are reused without duplication. -/
def meanIO (M N : Nat) (jobs : List Nat) :
    LogicalScheduledKernelIO (Fin M → Fin N → ℝ) (Fin M) where
  core := realIO M N
  schedule := realSchedule M N jobs
  inputs := matrixInputs M N
  indices := rowOutputs M N
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.MeanReduction.MeanIO

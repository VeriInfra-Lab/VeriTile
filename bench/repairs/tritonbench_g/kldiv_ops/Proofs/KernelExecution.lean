import bench.repairs.tritonbench_g.kldiv_ops.Contract
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Direct

/-! Complete typed source rows, executed against the current shared physical
memory. The row interpreter may fail; its success is proved, not assumed. -/
namespace VeriTile.Bench.Repairs.KldivOps.Kernel
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Grid (runRows rowJobs stores_congr)
open Contiguous
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 8000000

/-- Execute a whole source row, then realize exactly that row's stores in the
one physical memory. Every next row reads the resulting memory. -/
def executeRows {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (output : View) (cols : Nat)
    (source : Nat → M → Option (State κ)) : List Nat → M → Option M
  | [],memory => some memory
  | r::rest,memory => do
      let final ← source r memory
      let next := stores ops (fun rc => output.key (rc.1*cols+rc.2))
        (fun rc => a.project ((final.mem "O" (rc.1*cols+rc.2)).read .real))
        (rowJobs cols r) memory
      executeRows ops a output cols source rest next

def forwardRow {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (r : Nat) (memory : M) :=
  run (engine a.algebra) (KernelDSL.forward (parameters rows cols) flag red.code).surfaceBody
    (Tile.bindEpsilon (a.embed eps)
      (Program.atRow (physicalState ops a memory prediction target output) r))

def backwardRow {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (target output : View) (rows cols : Nat)
    (flag : Bool) (r : Nat) (memory : M) :=
  run (engine a.algebra) (KernelDSL.backward (parameters rows cols) flag).surfaceBody
    (Program.atRow (physicalState ops a memory target target output) r)

def executeForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (memory : M) : Option M :=
  executeRows ops a output (width red cols)
    (forwardRow ops a prediction target output rows cols flag red eps) (List.range rows) memory

def executeBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (target output : View) (rows cols : Nat)
    (flag : Bool) (memory : M) : Option M :=
  executeRows ops a output cols
    (backwardRow ops a target output rows cols flag) (List.range rows) memory

theorem executeRows_complete {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (output : View) (cols : Nat)
    (source : Nat → M → Option (State κ)) (value : M → Nat → Nat → α)
    (complete : ∀ r memory, ∃ final, source r memory = some final ∧
      ∀ c, a.project ((final.mem "O" (r*cols+c)).read .real) = value memory r c)
    (schedule : List Nat) (memory : M) :
    executeRows ops a output cols source schedule memory =
      some (runRows ops (fun rc => output.key (rc.1*cols+rc.2)) value cols schedule memory) := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons r rest ih =>
    obtain ⟨final,executed,values⟩ := complete r memory
    simp only [executeRows,executed,bind,Option.bind,runRows]
    have stored : stores ops (fun rc => output.key (rc.1*cols+rc.2))
        (fun rc => a.project ((final.mem "O" (rc.1*cols+rc.2)).read .real))
        (rowJobs cols r) memory =
      stores ops (fun rc => output.key (rc.1*cols+rc.2))
        (fun rc => value memory rc.1 rc.2) (rowJobs cols r) memory := by
      apply stores_congr
      intro rc member
      obtain ⟨c,_,same⟩ := List.mem_map.mp member
      cases same
      exact values c
    rw [stored]
    exact ih _

theorem forward_runs {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α) (memory : M) :
    executeForward ops a prediction target output rows cols flag red eps memory =
      some (runRows ops (fun rc => output.key (rc.1*width red cols+rc.2))
        (fun s r c => forwardRowValue ops a s prediction target output rows cols flag red eps r c)
        (width red cols) (List.range rows) memory) :=
  executeRows_complete ops a output (width red cols) _ _
    (fun r s => by
      simpa only [forwardRow,KernelDSL.forward_eq] using
        forward_row_executes ops a s prediction target output rows cols flag red eps r) _ memory

theorem backward_runs {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (a : Adapter α κ) (target output : View) (rows cols : Nat)
    (flag : Bool) (memory : M) :
    executeBackward ops a target output rows cols flag memory =
      some (runRows ops (fun rc => output.key (rc.1*cols+rc.2))
        (fun s r c => backwardRowValue ops a s target output rows cols flag r c)
        cols (List.range rows) memory) :=
  executeRows_complete ops a output cols _ _
    (fun r s => by
      simpa only [backwardRow,KernelDSL.backward_eq] using
        backward_row_executes ops a s target output rows cols flag r) _ memory

/-- Complete raw-cell and byte complement, including allocation padding,
unrelated allocations, and partial-byte tails. -/
def MemoryFrame {M α β : Type} (ops : Ops M α β) (output : View) (before after : M) : Prop :=
  (∀ key, (∀ i, i < output.numel → key.disjoint (output.key i)) →
    ops.read after key = ops.read before key) ∧
  (∀ owner address, (∀ i, i < output.numel → ¬ (output.key i).covers owner address) →
    ops.observe after owner address = ops.observe before owner address)

theorem writeView_frame {M α β : Type} (ops : Ops M α β) (output : View)
    (values : Nat → α) (memory : M) :
    MemoryFrame ops output memory (writeView ops output values memory) := by
  constructor
  · intro key outside
    apply read_frame
    intro i hi
    exact outside i (List.mem_range.mp hi)
  · exact writeView_bytes ops output values memory

/-- Machine PID, signed tail masks and terminal loop increments. -/
def Control (rows cols : Nat) : Prop :=
  ∀ r, r < rows → (BitVec.ofNat 32 r).toInt = (r : Int) ∧
    ∀ k lane,
      k < VeriTile.Bench.TritonBenchG.KldivOps.FullMath.rounds cols (Host.blockSize cols) →
      lane < Host.blockSize cols →
      (BitVec.ofNat 64 (k*Host.blockSize cols)+BitVec.ofNat 64 lane).slt
        (BitVec.ofNat 64 cols) = decide (k*Host.blockSize cols+lane < cols) ∧
      (BitVec.ofNat 64 (k*Host.blockSize cols)+BitVec.ofNat 64 (Host.blockSize cols)).toInt =
        (((k+1)*Host.blockSize cols : Nat) : Int)

theorem control {t : Tensor} {v : View} (binding : Binding t v)
    (admitted : TensorContract t) (rows cols : Nat) (shape : v.shape = [rows,cols])
    (dispatch : Dispatch rows) : Control rows cols := by
  intro r hr
  exact ⟨pid_exact rows r dispatch hr,
    fun k lane hk hl => checked_indices binding admitted rows cols shape (by omega) k lane hk hl⟩

structure ForwardPost {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (neg : Negation α) (a : Adapter α κ) (p t : Tensor)
    (prediction target output : View) (rows cols : Nat) (flag : Bool)
    (red : Host.Reduction) (eps : α) (expected : Nat → α) (before after : M) : Prop where
  executes : executeForward ops a prediction target output rows cols flag red eps before = some after
  values : ∀ i, i < output.numel → logical ops neg after output i = expected i
  frame : MemoryFrame ops output before after
  predictionAddresses : TensorAddresses p prediction
  targetAddresses : TensorAddresses t target
  outputAddresses : OutputAddresses output
  control : Control rows cols

structure BackwardPost {M α β κ : Type} [Inhabited κ] (ops : Ops M α β)
    (neg : Negation α) (a : Adapter α κ) (d t : Tensor)
    (target output : View) (rows cols : Nat) (flag : Bool)
    (expected : Nat → α) (before after : M) : Prop where
  executes : executeBackward ops a target output rows cols flag before = some after
  values : ∀ i, i < output.numel → logical ops neg after output i = expected i
  frame : MemoryFrame ops output before after
  targetAddresses : TensorAddresses t target
  outputAddresses : TensorAddresses d output
  control : Control rows cols

theorem completeForward {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (initial : M)
    (p t : Tensor) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : α)
    (domain : ForwardDomain p t prediction target output rows cols red) (dispatch : Dispatch rows)
    (expected : Nat → α)
    (equal : ∀ i, forwardExpected ops neg a initial prediction target cols flag red eps i = expected i) :
    ∃ after, ForwardPost ops neg a p t prediction target output rows cols flag red eps expected initial after := by
  refine ⟨forwardKernel ops a initial prediction target output rows cols flag red eps,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [forward_runs,Contiguous.forward_complete ops neg a initial p t prediction target output rows cols flag red eps
      domain.admitted.toPairContract domain.predictionBinding domain.targetBinding domain.shape domain.outputShape
      domain.outputDense domain.outputPositive domain.outputBacked domain.predictionSeparate domain.targetSeparate]
  · intro i hi
    exact (forward_values ops neg a initial p t prediction target output rows cols flag red eps
      domain.admitted.toPairContract domain.predictionBinding domain.targetBinding domain.shape domain.outputShape
      domain.outputDense domain.outputPositive i hi).trans (equal i)
  · exact writeView_frame ops output _ initial
  · exact source_pointer domain.predictionBinding domain.admitted.left
  · exact source_pointer domain.targetBinding domain.admitted.right
  · exact allocated_pointer output domain.outputDense domain.outputBacked domain.outputAligned
  · exact control domain.predictionBinding domain.admitted.left rows cols domain.shape dispatch

theorem completeBackward {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (a : Adapter α κ) (initial : M)
    (d t g : Tensor) (target output : View) (rows cols : Nat) (flag : Bool)
    (domain : BackwardDomain d t g target output rows cols) (dispatch : Dispatch rows)
    (expected : Nat → α)
    (equal : ∀ i, Contiguous.derivative a flag (logical ops neg initial target i) = expected i) :
    ∃ after, BackwardPost ops neg a d t target output rows cols flag expected initial after := by
  refine ⟨backwardKernel ops a initial target output rows cols flag,?_,?_,?_,?_,?_,?_⟩
  · rw [backward_runs,Contiguous.backward_complete ops neg a initial d t g target output rows cols flag
      domain.admitted domain.destinationBinding domain.targetBinding domain.shape]
  · intro i hi
    exact (backward_values ops neg a initial d t target output rows cols flag domain.admitted.toPairContract
      domain.destinationBinding domain.targetBinding domain.shape i hi).trans (equal i)
  · exact writeView_frame ops output _ initial
  · exact source_pointer domain.targetBinding domain.admitted.right
  · exact source_pointer domain.destinationBinding domain.admitted.left
  · exact control domain.destinationBinding domain.admitted.left rows cols domain.shape dispatch

#axiomsClean forward_runs
#axiomsClean backward_runs
#axiomsClean writeView_frame
#axiomsClean completeForward
#axiomsClean completeBackward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel

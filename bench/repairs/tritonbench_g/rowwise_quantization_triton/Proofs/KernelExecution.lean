import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- The shared interpreter rejects llrint and the DSL marks its subsequent
integer register read as opaque. This task-local extension supplies only the
scalar signed64 primitive and that exact named, typed register read. Missing
registers still fail. Every other expression uses the shared interpreter.
This is an explicit semantic boundary, not a numerical experiment or a
premise about any row/output result. -/
def eval {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64) :
    {d : TileDType} → {sh : TileShape} → ComputeExpr d sh → State α →
      Option (Values α d sh)
  | _, _, .llrint e, s => do
      let v ← FP.Structural.evalExpr M e s
      return fun i => (round (v i)).toInt
  | .int, sh, .opaque label, s =>
      if label = "runtime compute value `output`" then s.regs .int sh "output" else none
  | _, _, e, s => FP.Structural.evalExpr M e s

/-- Signed value stored after the source INT8 narrowing of llrint. -/
def signedByte (word : BitVec 64) : Int := (BitVec.ofInt 8 word.toInt).toInt

/-- Narrowing changes the signed representative, preserving its observed byte. -/
theorem narrow_signedByte (word : BitVec 64) :
    narrow8 (signedByte word) = narrow8 word.toInt := by
  have same : (BitVec.ofInt 8 word.toInt).toInt % 256 = word.toInt % 256 := by
    simpa only [BitVec.toInt_ofInt] using (Int.bmod_emod (x := word.toInt) (m := 256))
  simp only [signedByte, narrow8, same]


def step {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64) :
    ComputeStmt → State α → Option (State α)
  | .assign d sh n e, s => do
      let v ← eval M round e s
      return s.setReg n d sh v
  | .store _ _ mem e mask, s => do
      store M mem mask (← eval M round e s) s
  | st, s => FP.Structural.step M st s

def run {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64) :
    List ComputeStmt → State α → Option (State α)
  | [], s => some s
  | st :: body, s => (step M round st s).bind (run M round body)

def exec {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (kernel : ComputeKernel) (s : State α) : Option (State α) :=
  run M round kernel.surfaceBody s

/-- Equality is primitive floating equality. It is not structural equality of
payloads, and it is not required to equal Real equality. -/
def backend {α : Type} (M : Algebra α) (equal : α → α → Bool) : Algebra α :=
  { M with compareEq := fun _ _ => some equal }

structure Parameters where
  x : RegionName
  out : RegionName
  maxOut : RegionName
  columns : Nat
  width : Nat

def kernel (p : Parameters) : ComputeKernel :=
  RowwiseQuantizationTriton.source p.x p.out p.maxOut p.columns p.width

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl


/-- Actual source execution succeeds, with a signed64 result for llrint and
both original stores retained. No final-state equation is an assumption. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (hw : 0 < p.width) :
    {t : State α // exec (backend M equal) round (kernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := { maxSteps := 1000000 }) [kernel, RowwiseQuantizationTriton.source,
    exec, ComputeKernel.surfaceBody, run, step, eval, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.natLt, FP.Structural.bop, backend, State.setReg,
    ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
    TileShape.axisDim, hw, fold_regs, fold_pids]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelExecution

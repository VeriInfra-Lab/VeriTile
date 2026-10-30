import bench.tritonbench_g.int4_matmul.Proofs.Kernel.Execution
import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPContract

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

def prelude (p : Parameters) := (kernel p).surfaceBody.take 17
def body (p : Parameters) : List ComputeStmt :=
  match (kernel p).surfaceBody[17]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (kernel p).surfaceBody.drop 18

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]

def blocksM (p : Parameters) : Nat := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) : Nat := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : RegisterState α) :=
  (s.pids 0 / (p.groupRows*blocksN p))*p.groupRows
def rowsInGroup (p : Parameters) (s : RegisterState α) :=
  min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : RegisterState α) :=
  firstRow p s+s.pids 0%rowsInGroup p s
def columnBlock (p : Parameters) (s : RegisterState α) :=
  (s.pids 0%(p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : RegisterState α) (i : Fin p.blockRows) :=
  (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : RegisterState α) (i : Fin p.blockColumns) :=
  (columnBlock p s*p.blockColumns+i.val)%p.columns
def inputInner (p : Parameters) (s : RegisterState α) (i : Fin p.blockInner) :=
  s.pids 1*p.blockInner+i.val
def innerIndex (p : Parameters) (s : RegisterState α) (step : Nat) (i : Fin p.blockInner) :=
  inputInner p s i+step*p.blockInner*p.splits

/-- Actual source increments, including multiplication before packed division. -/
def aOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+
    step*(p.blockInner*p.splits*p.inputInnerStride)
def bOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : BIndex p) :=
  inputInner p s i.1/8*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+
    step*(p.blockInner*p.splits*p.weightInnerStride/8)
def scaleOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : BIndex p) :=
  innerIndex p s step i.1/p.groupSize*p.scaleGroupStride+
    inputColumn p s i.2.1*p.scaleColumnStride
def zeroOffset (p : Parameters) (s : RegisterState α) (step : Nat) (i : BIndex p) :=
  innerIndex p s step i.1/p.groupSize*p.zeroGroupStride+
    inputColumn p s i.2.1/8*p.zeroColumnStride
def outputOffset (p : Parameters) (s : RegisterState α) (i : CIndex p) :=
  p.outputRowStride*(rowBlock p s*p.blockRows+i.1.val)+
    p.outputColumnStride*(columnBlock p s*p.blockColumns+i.2.1.val)
def active (p : Parameters) (s : RegisterState α) (i : CIndex p) : Prop :=
  rowBlock p s*p.blockRows+i.1.val<p.rows ∧
    columnBlock p s*p.blockColumns+i.2.1.val<p.columns
instance (p : Parameters) (s : RegisterState α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

abbrev signedNibble := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble

def difference [Inhabited α] (p : Parameters) (s : RegisterState α) (step : Nat) (i : BIndex p) : Int :=
  signedNibble (if innerIndex p s step i.1<p.inner then
    (s.mem p.weight (bOffset p s step i)).read .int else 0) (inputInner p s i.1)-
  signedNibble (if innerIndex p s step i.1<p.inner then
    (s.mem p.zero (zeroOffset p s step i)).read .int else 0) (inputColumn p s i.2.1)

/-- The native INT32 nibble subtraction cannot overflow, including masked
loads. This is an integer theorem, not a floating numerical assumption. -/
theorem difference_int32_exact [Inhabited α] (p : Parameters) (s : RegisterState α)
    (step : Nat) (i : BIndex p) :
    (BitVec.ofInt 32 (difference p s step i)).toInt=difference p s step i := by
  exact VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits.difference_no_overflow _ _

def aValue [Inhabited α] (o : Ops α) (p : Parameters) (s : RegisterState α)
    (step : Nat) (i : AIndex p) : α :=
  if innerIndex p s step i.2.1<p.inner then (s.mem p.input (aOffset p s step i)).read .real else o.zero
def bValue [Inhabited α] (o : Ops α) (p : Parameters) (s : RegisterState α)
    (step : Nat) (i : BIndex p) : α :=
  o.mul (o.fromInt (difference p s step i))
    (if innerIndex p s step i.1<p.inner then (s.mem p.scale (scaleOffset p s step i)).read .real else o.zero)

/-- Explicit bounded-instruction recurrence over the four source loads.
This definition has no callback for a complete dot, block or output. -/
def accumulation [Inhabited α] (o : Ops α) (p : Parameters) (s : RegisterState α) :
    Nat → CIndex p → α
  | 0 => fun _ => o.zero
  | step+1 => dotTile o (aValue o p s step) (bValue o p s step) (accumulation o p s step)

theorem source_decomposition (p : Parameters) :
    (kernel p).surfaceBody=prelude p++[.forRange "k" 0 (steps p) 1 (body p)]++epilogue p := rfl

#axiomsClean source_decomposition
#axiomsClean difference_int32_exact
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution

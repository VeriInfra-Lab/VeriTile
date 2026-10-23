import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelExecution
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
abbrev Parameters := KernelSource.GemmParameters

def source : Precision → Precision → Parameters → ComputeKernel
  | .half, .half => KernelSource.gemmHalfHalf
  | .half, .single => KernelSource.gemmHalfSingle
  | .single, .half => KernelSource.gemmSingleHalf
  | .single, .single => KernelSource.gemmSingleSingle

def prelude (rowScale output : Precision) (p : Parameters) :=
  (source rowScale output p).surfaceBody.take 20
def body (p : Parameters) : List ComputeStmt :=
  match (source .half .half p).surfaceBody[20]? with
  | some (ComputeStmt.forRange _ _ _ _ b) => b
  | _ => []
def epilogue (rowScale output : Precision) (p : Parameters) :=
  (source rowScale output p).surfaceBody.drop 21

theorem source_decomposition (rowScale output : Precision) (p : Parameters) :
    (source rowScale output p).surfaceBody =
      prelude rowScale output p ++ [.forRange "k" 0 p.steps 1 (body p)] ++
        epilogue rowScale output p := by
  cases rowScale <;> cases output <;> rfl

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State F) := s.pids 0 / (p.groupRows*blocksN p) * p.groupRows
def rowsInGroup (p : Parameters) (s : State F) := min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State F) := firstRow p s + s.pids 0 % rowsInGroup p s
def columnBlock (p : Parameters) (s : State F) := (s.pids 0 % (p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State F) (i : Fin p.blockRows) :=
  (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State F) (j : Fin p.blockColumns) :=
  (columnBlock p s*p.blockColumns+j.val)%p.columns
def inputInner (p : Parameters) (s : State F) (j : Fin p.blockInner) := s.pids 1*p.blockInner+j.val
def aOffset (p : Parameters) (s : State F) (k : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+
    k*(p.blockInner*p.split*p.inputInnerStride)
def bOffset (p : Parameters) (s : State F) (k : Nat) (i : BIndex p) :=
  inputInner p s i.1*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+
    k*(p.blockInner*p.split*p.weightInnerStride)

def aValue {F : Type} [Inhabited F] (p : Parameters) (s : State F)
    (k : Nat) (i : AIndex p) : Int :=
  if inputInner p s i.2.1 < p.inner-k*(p.blockInner*p.split) then
    (s.mem p.input (aOffset p s k i)).read .int else 0
def bValue {F : Type} [Inhabited F] (p : Parameters) (s : State F)
    (k : Nat) (i : BIndex p) : Int :=
  if inputInner p s i.1 < p.inner-k*(p.blockInner*p.split) then
    (s.mem p.weight (bOffset p s k i)).read .int else 0

/-- Fixed instruction recurrence, including the source's incoming accumulator
at every block. Coordinates and masked operands come from the source; no
zero-seed/modular substitution is made for carry-fed saturating MMA. -/
def accumulation {I F H : Type} [Inhabited F] (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (p : Parameters) (s : State F) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i =>
      carrier.decode <| (plans p.blockInner (rowBlock p s*p.blockRows+i.1.val)
        (columnBlock p s*p.blockColumns+i.2.1.val) (s.pids 1) k).eval ops
        (fun j => BitVec.ofInt 8 (aValue p s k (i.1,j,PUnit.unit)))
        (fun j => BitVec.ofInt 8 (bValue p s k (j,i.2.1,PUnit.unit)))
        (carrier.encode (accumulation carrier ops plans p s k i))

def aScale {F : Type} [Inhabited F] (M : Algebra F) (kind : Precision)
    (p : Parameters) (s : State F) (i : Fin p.blockRows) : F :=
  match kind with
  | .half => if inputRow p s i < p.rows then
      (s.mem p.inputScale (inputRow p s i*p.inputScaleStride)).read .fp16
      else M.cast none .real .fp16 (M.literal none .real 0)
  | .single => M.fp32Load (if inputRow p s i < p.rows then
      (s.mem p.inputScale (inputRow p s i*p.inputScaleStride)).read .real
      else M.literal (some .fp32) .real 0)
def bScale {F : Type} [Inhabited F] (M : Algebra F)
    (p : Parameters) (s : State F) (i : Fin p.blockColumns) : F :=
  M.fp32Load (if inputColumn p s i < p.columns then
    (s.mem p.weightScale (inputColumn p s i*p.weightScaleStride)).read .real
    else M.literal (some .fp32) .real 0)

structure Invariant {I F H : Type} [Inhabited F] (M : Algebra F)
    (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale : Precision) (p : Parameters) (initial : State F)
    (k : Nat) (current : State F) : Prop where
  memory : current.mem = initial.mem
  pids : current.pids = initial.pids
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "offs_k" = some (fun i => inputInner p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial k i))
  ascale : match rowScale with
    | .half => current.regs .fp16 [p.blockRows] "a_scale" = some (fun i => aScale M .half p initial i.1)
    | .single => current.regs .real [p.blockRows] "a_scale" = some (fun i => aScale M .single p initial i.1)
  bscale : current.regs .real [p.blockColumns] "b_scale" = some (fun i => bScale M p initial i.1)
  acc : current.regs .int [p.blockRows,p.blockColumns] "accumulator" =
    some (accumulation carrier ops plans p initial k)

#axiomsClean source_decomposition
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm

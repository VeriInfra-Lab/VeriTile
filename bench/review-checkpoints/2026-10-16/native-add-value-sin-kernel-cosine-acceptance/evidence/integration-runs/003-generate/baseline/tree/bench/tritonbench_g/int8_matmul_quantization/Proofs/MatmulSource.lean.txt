import bench.tritonbench_g.int8_matmul_quantization.Int8MatmulQuantization
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Float.SignedCast
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance matmulSourceDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
structure Parameters where
  (input weight inputScale weightScale output : RegionName)
  (rows columns inner inputRowStride inputInnerStride inputScaleStride
    weightInnerStride weightColumnStride weightScaleStride outputRowStride outputColumnStride
    blockRows blockColumns blockInner groupRows steps : Nat)

/-- Original SPLIT_K=1 source. The original accumulator.to(float32)
is represented by the frontend's intToReal node under fp32 precision;
explicit widening of the half scale selects that precision before promotion.
The source below has no extra arithmetic event. Its erased signed storage
and update boundaries are restored by widths in source. The final half
cast, every K iteration, and both remainder masks remain. -/
def typed (p : Parameters) : ComputeKernel :=
  let A : Region .int := p.input
  let B : Region .int := p.weight
  let as_ptr := p.inputScale
  let bs_ptr := p.weightScale
  let C := p.output
  triton {
  tl.region A = tl.int8, B = tl.int8, as_ptr = tl.float16, bs_ptr = tl.float32, C = tl.float16
  pid = tl.program_id(0)
  pid_sp_k = tl.program_id(1)
  num_pid_m = tl.cdiv($(p.rows), $(p.blockRows))
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  num_pid_in_group = $(p.groupRows) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(p.groupRows)
  group_size_m = min(num_pid_m - first_pid_m, $(p.groupRows))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))) % $(p.rows)
  offs_bn = (pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))) % $(p.columns)
  offs_k = pid_sp_k * $(p.blockInner) + tl.arange(0, $(p.blockInner))
  a_ptrs = A + (offs_am[:, None] * $(p.inputRowStride) + offs_k[None, :] * $(p.inputInnerStride))
  b_ptrs = B + (offs_k[:, None] * $(p.weightInnerStride) + offs_bn[None, :] * $(p.weightColumnStride))
  as_ptrs = as_ptr + offs_am * $(p.inputScaleStride)
  bs_ptrs = bs_ptr + offs_bn * $(p.weightScaleStride)
  a_scale = tl.load(as_ptrs, mask=offs_am < $(p.rows), other=0.0)
  b_scale = tl.load(bs_ptrs, mask=offs_bn < $(p.columns), other=0.0)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=tl.int32)
  for k in range($(0), $(p.steps), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(p.inner) - k * $(p.blockInner), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(p.inner) - k * $(p.blockInner), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(p.blockInner) * $(p.inputInnerStride)
    b_ptrs += $(p.blockInner) * $(p.weightInnerStride)
  }
  c = (accumulator * (a_scale[:, None]).to(tl.float32) * b_scale[None, :]).to(tl.float16)
  offs_cm = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_cn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  c_ptrs = C + $(p.outputRowStride) * offs_cm[:, None] + $(p.outputColumnStride) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(p.rows)) & (offs_cn[None, :] < $(p.columns))
  tl.store(c_ptrs, c, mask=c_mask)
}
def widths : ComputeStmt → ComputeStmt
  | .assign .int sh name (.alg e) =>
      if name = "a" ∨ name = "b" then .assign .int sh name (.alg (.intToSigned 8 e))
      else if name = "accumulator" then .assign .int sh name (.alg (.intToSigned 32 e))
      else .assign .int sh name (.alg e)
  | stmt => stmt
def source (p : Parameters) : ComputeKernel := .mk
  [p.input,p.weight,p.inputScale,p.weightScale] [p.output]
  ((typed p).surfaceBody.map fun stmt => match stmt with
    | .forRange n lo hi step b => .forRange n lo hi step (b.map widths)
    | other => other)
def prelude (p : Parameters) := (source p).surfaceBody.take 20
def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[20]? with | some (ComputeStmt.forRange _ _ _ _ b) => b | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 21

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [.forRange "k" 0 p.steps 1 (body p)] ++ epilogue p := by rfl

/-- Every integer dot receives explicitly narrowed int8 operands. Addition
is narrowed to int32 after every source block. This total software dot has
no floating associativity premise; native lowering is a separate binding. -/
def backend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := QuantizeKvTransform.MixedBits.backend rest cfg (fun _ _ => 0)
  { base with
    fromInt := fun precision n => if precision = some .fp32 then FP.round .fp32 cfg (n : Rat)
      else base.fromInt precision n
    integerDot := fun _ {batch} {_} {_} {_} a b =>
      some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt) }

def wrap (w : Nat) (n : Int) : Int := (BitVec.ofInt w n).toInt
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State Word) := s.pids 0 / (p.groupRows*blocksN p) * p.groupRows
def rowsInGroup (p : Parameters) (s : State Word) := min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State Word) := firstRow p s + s.pids 0 % rowsInGroup p s
def columnBlock (p : Parameters) (s : State Word) := (s.pids 0 % (p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State Word) (i : Fin p.blockRows) := (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := (columnBlock p s*p.blockColumns+j.val)%p.columns
def inputInner (p : Parameters) (s : State Word) (j : Fin p.blockInner) := s.pids 1*p.blockInner+j.val
def aOffset (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) :=
  inputRow p s i.1*p.inputRowStride+inputInner p s i.2.1*p.inputInnerStride+step*(p.blockInner*p.inputInnerStride)
def bOffset (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) :=
  inputInner p s i.1*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+step*(p.blockInner*p.weightInnerStride)
def aValue (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) : Int :=
  wrap 8 (if inputInner p s i.2.1 < p.inner-step*p.blockInner then
    (s.mem p.input (aOffset p s step i)).read .int else 0)
def bValue (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Int :=
  wrap 8 (if inputInner p s i.1 < p.inner-step*p.blockInner then
    (s.mem p.weight (bOffset p s step i)).read .int else 0)
def accumulation (p : Parameters) (s : State Word) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => wrap 32 (accumulation p s k i + (FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i).toInt)
def aScale (p : Parameters) (s : State Word) (i : Fin p.blockRows) : Word :=
  if inputRow p s i < p.rows then (s.mem p.inputScale (inputRow p s i*p.inputScaleStride)).read .fp16 else FP.Value.zero .fp32
def bScale (p : Parameters) (s : State Word) (j : Fin p.blockColumns) : Word :=
  if inputColumn p s j < p.columns then (s.mem p.weightScale (inputColumn p s j*p.weightScaleStride)).read .real else FP.Value.zero .fp32
/-- Independent source-ordered word recurrence, integer-to-fp32 rounding,
two separately rounded fp32 products, and actual binary16 output encoding. -/
def outputValue (cfg : FP.Config) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow cfg
    (FP.mul cfg (FP.mul cfg (FP.round .fp32 cfg (accumulation p s p.steps i : Rat))
      (QuantizeKvTransform.MixedBits.widen (aScale p s i.1))) (bScale p s i.2.1))
def outputOffset (p : Parameters) (s : State Word) (i : CIndex p) :=
  p.outputRowStride*(rowBlock p s*p.blockRows+i.1.val)+p.outputColumnStride*(columnBlock p s*p.blockColumns+i.2.1.val)
def active (p : Parameters) (s : State Word) (i : CIndex p) : Prop :=
  rowBlock p s*p.blockRows+i.1.val < p.rows ∧ columnBlock p s*p.blockColumns+i.2.1.val < p.columns
instance (p : Parameters) (s : State Word) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul

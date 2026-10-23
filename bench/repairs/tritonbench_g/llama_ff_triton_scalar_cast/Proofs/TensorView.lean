import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Checked
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Words
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RealArithmetic

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.TensorView
open VeriTile.Triton FP.Structural Execution
open LlamaFF.Contract
noncomputable section
set_option maxHeartbeats 4000000

/-- Task-local observation of the actual returned contiguous half Tensor.
Only metadata and coordinate laws occur here; no allocation, numerical value,
execution result or source-to-program binding is a field. The storage origin
is the observed data_ptr, as in the source's zero-offset dense contract. -/
structure Output (x w : TensorInfo) where
  info : TensorInfo
  shape : info.shape=[batch x,sequence x,outputs w]
  dtype : info.dtype=.other "float16"
  device : info.device=x.device
  physicalIndex : Nat → Nat → Nat → Nat
  coordinates : ∀ b m n,b<batch x → m<sequence x → n<outputs w →
    physicalIndex b m n=(b*sequence x+m)*outputs w+n

abbrev Logical (x w : TensorInfo) := Fin (batch x) × Fin (sequence x) × Fin (outputs w)
def index (x w : TensorInfo) (c : Logical x w) : Grid.Index (Checked.parameters x w) :=
  (⟨c.1.val*sequence x+c.2.1.val,by
      have hb := c.1.isLt
      have hm := c.2.1.isLt
      change c.1.val*sequence x+c.2.1.val<batch x*sequence x
      nlinarith⟩,c.2.2)
theorem coordinates (x w : TensorInfo) (o : Output x w) (c : Logical x w) :
    o.physicalIndex c.1.val c.2.1.val c.2.2.val=Grid.address (Checked.parameters x w) (index x w c) := by
  rw [o.coordinates _ _ _ c.1.isLt c.2.1.isLt c.2.2.isLt]
  simp only [Grid.address,Checked.parameters,index]
  ring

structure NativeTensor (x w : TensorInfo) where
  view : Output x w
  bytes : Nat → Words.Byte
def NativeTensor.get (t : NativeTensor x w) (c : Logical x w) : Words.H :=
  Words.decode16 (t.bytes (2*t.view.physicalIndex c.1.val c.2.1.val c.2.2.val))
    (t.bytes (2*t.view.physicalIndex c.1.val c.2.1.val c.2.2.val+1))
def nativeResult (o : Output x w) (initial : Words.Memory) (final : State Words.F) : NativeTensor x w :=
  ⟨o,Words.commit initial final "OUT"⟩

/-- Explicit cast-erased half decoding. Nonfinite half payloads are sent to
zero only in this mathematical model; the native representation keeps them. -/
def decodeHalf (h : Words.H) : ℝ :=
  let exponent := h.toNat/1024%32
  let fraction := h.toNat%1024
  if exponent=31 then 0 else
    let magnitude : ℚ := if exponent=0 then (fraction:ℚ)/16777216
      else if exponent<25 then (1024+fraction:ℚ)/(2^(25-exponent))
      else (1024+fraction:ℚ)*2^(exponent-25)
    ((if h.toNat≥32768 then -magnitude else magnitude):ℚ)
def realEntry (p : Parameters) (mem : Words.Memory) : State ℝ where
  mem := fun r o => if Words.isByteRegion p r then .mk .int (mem r o).toInt
    else .mk .fp16 (decodeHalf (Words.read16 mem r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d
theorem realEntry_bytes (p : Parameters) (mem : Words.Memory) : ByteValues (realEntry p mem) := by
  intro r hr o
  change Float8E5Bits.isInt8 ((if Words.isByteRegion p r then Cell.mk .int (mem r o).toInt
    else Cell.mk .fp16 (decodeHalf (Words.read16 mem r o))).read .int)=Bool.true
  split
  · simp only [Cell.read_same,Float8E5Bits.isInt8,decide_eq_true_eq]
    exact ⟨BitVec.le_toInt _,BitVec.toInt_lt⟩
  · simp [Cell.read,defaultValue,Float8E5Bits.isInt8]
structure RealTensor (x w : TensorInfo) where
  view : Output x w
  values : Nat → ℝ
def RealTensor.get (t : RealTensor x w) (c : Logical x w) : ℝ :=
  t.values (t.view.physicalIndex c.1.val c.2.1.val c.2.2.val)
def realResult (o : Output x w) (final : State ℝ) : RealTensor x w :=
  ⟨o,fun offset => (final.mem "OUT" offset).read .fp16⟩

#axiomsClean coordinates
#axiomsClean realEntry_bytes
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.TensorView

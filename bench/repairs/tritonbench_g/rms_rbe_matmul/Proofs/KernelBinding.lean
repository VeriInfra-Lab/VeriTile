import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNative
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelSourceGraph
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelGraphCongruence
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.CheckedIndex

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelBinding
open VeriTile.Triton FP.Structural
open RmsMatmulRbe.Contract (width int8 fp16 fp32)
open Contract (Inputs InputContract)
open KernelNative (Term Word)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
abbrev Bytes := Nat → BitVec 8
abbrev OriginalTensor := RmsMatmulRbe.Contract.Tensor
abbrev Cast := VeriTile.Torch.DType → VeriTile.Torch.DType → Word → Word

def parameters (i : Inputs) : KernelSource.Parameters :=
  ⟨(Checked.shape i).M,(Checked.shape i).N,(Checked.shape i).K,(Checked.shape i).headDim,
    i.scalars.start,i.rotary⟩
def fp8 (i : Inputs) : Bool := decide (i.weight.metadata.dtype=int8)
def address (t : OriginalTensor) (j : Nat) := t.base+width t.metadata.dtype*(t.offset+j)
def readWord (heap : Bytes) (t : OriginalTensor) (j : Nat) : Word :=
  BitVec.ofNat 32 (∑ k∈Finset.range (width t.metadata.dtype),
    (heap (address t j+k)).toNat*256^k)
def signedByte (word : Word) : Int := (BitVec.ofNat 8 word.toNat).toInt
def inputTerm (region : RegionName) (t : OriginalTensor) (j : Nat) (word : Word) : Term :=
  .leaf region (address t j) word

/-- The original wrapper condition: same dtype keeps the original word;
different dtype invokes precisely one ordinary numeric Tensor.to scalar cast.
For int8, this numeric conversion precedes later E5 payload reinterpretation. -/
def rmsWord (i : Inputs) (cast : Cast) (before : Bytes) (k : Nat) : Word :=
  let original := readWord before i.rms k
  if i.rms.metadata.dtype=i.weight.metadata.dtype then original
  else cast i.rms.metadata.dtype i.weight.metadata.dtype original

/-- Boundary of successful ordinary input/storage APIs and the original scalar
Tensor.to operation. It mentions only original input bytes/words and one RMS
element conversion, never a kernel, reduction, projection or output value. -/
structure Preparation (i : Inputs) (a : Checked.Allocation i) (cast : Cast)
    (before prepared : Bytes) : Prop where
  originalWords : ∀ t, t∈i.reads → ∀ j, j<t.metadata.numel →
    readWord prepared t j=readWord before t j
  originalBytes : ∀ t, t∈i.reads → ∀ j, j<t.bytes → prepared (t.base+j)=before (t.base+j)
  castOne : ∀ k, k<(Checked.shape i).K → readWord prepared a.rms k=rmsWord i cast before k

def inputWords (i : Inputs) (a : Checked.Allocation i)
    (x w r : Nat → Word) : KernelGraph.Input Term where
  x := fun b m k => inputTerm "X" i.x (b*((parameters i).M*(parameters i).K)+m*(parameters i).K+k)
    (x (b*((parameters i).M*(parameters i).K)+m*(parameters i).K+k))
  weightHalf := fun n k => if fp8 i then default else inputTerm "W" i.weight (k+n*(parameters i).K) (w (k+n*(parameters i).K))
  weightByte := fun n k => if fp8 i then signedByte (w (k+n*(parameters i).K)) else 0
  rmsHalf := fun k => if fp8 i then default else inputTerm "RMS" a.rms k (r k)
  rmsByte := fun k => if fp8 i then signedByte (r k) else 0
def preparedInputs (i : Inputs) (a : Checked.Allocation i) (heap : Bytes) : KernelGraph.Input Term :=
  inputWords i a (readWord heap i.x) (readWord heap i.weight) (readWord heap a.rms)
def originalInputs (i : Inputs) (a : Checked.Allocation i) (cast : Cast) (before : Bytes) : KernelGraph.Input Term :=
  inputWords i a (readWord before i.x) (readWord before i.weight) (rmsWord i cast before)

/-- Actual argument pointers read a single shared physical byte heap. Same-base
input aliases and distinct storage offsets therefore read the same physical
bytes whenever their addresses coincide. -/
def state (i : Inputs) (a : Checked.Allocation i) (heap : Bytes) (scratch : State Term) : State Term :=
  {scratch with mem:=fun r off =>
    if r="X" then .mk .real (inputTerm "X" i.x off (readWord heap i.x off))
    else if r="W" then
      (if fp8 i then Cell.mk TileDType.int (signedByte (readWord heap i.weight off))
       else Cell.mk TileDType.fp16 (inputTerm "W" i.weight off (readWord heap i.weight off)))
    else if r="RMS" then
      (if fp8 i then Cell.mk TileDType.int (signedByte (readWord heap a.rms off))
       else Cell.mk TileDType.fp16 (inputTerm "RMS" a.rms off (readWord heap a.rms off)))
    else scratch.mem r off}

theorem state_inputs (i : Inputs) (a : Checked.Allocation i) (heap : Bytes) (scratch : State Term) :
    KernelSource.inputs (parameters i) (state i a heap scratch)=preparedInputs i a heap := by
  unfold KernelSource.inputs preparedInputs inputWords
  congr 1
  all_goals funext
  all_goals cases hf : fp8 i <;> simp only [state,hf,Bool.false_eq_true,if_true,if_false]
  all_goals simp [Cell.read,defaultValue]

theorem signedByte_valid (word : Word) : Float8E5Bits.isInt8 (signedByte word)=Bool.true := by
  have lo := BitVec.le_toInt (BitVec.ofNat 8 word.toNat)
  have hi := BitVec.toInt_lt (x:=BitVec.ofNat 8 word.toNat)
  simp only [signedByte,Float8E5Bits.isInt8,decide_eq_true_eq]
  norm_num at lo hi ⊢
  exact ⟨lo,hi⟩
theorem state_bytes (i : Inputs) (a : Checked.Allocation i) (heap : Bytes) (scratch : State Term) :
    KernelSource.ByteMemory (state i a heap scratch) := by
  intro r hr off
  rcases hr with rfl|rfl <;> cases hf : fp8 i <;>
    simp only [state,hf,Bool.false_eq_true,if_true,if_false]
  all_goals simp [Cell.read,defaultValue,signedByte_valid]
  all_goals decide

theorem original_agreement (i : Inputs) (h : InputContract i) (a : Checked.Allocation i)
    (cast : Cast) (before prepared : Bytes) (prep : Preparation i a cast before prepared)
    (batch : Nat) (hb : batch<(Checked.shape i).B) :
    KernelGraph.Agreement (parameters i) batch (preparedInputs i a prepared) (originalInputs i a cast before) := by
  have f := Checked.facts i h
  have hx (m k : Nat) (hm : m<(parameters i).M) (hk : k<(parameters i).K) :
      readWord prepared i.x (batch*((parameters i).M*(parameters i).K)+m*(parameters i).K+k)=
        readWord before i.x (batch*((parameters i).M*(parameters i).K)+m*(parameters i).K+k) := by
    apply prep.originalWords i.x (by simp [Contract.Inputs.reads])
    rw [f.xCount]
    simpa [RmsMatmulRbe.Checked.flat3,parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using
      RmsMatmulRbe.Checked.flat3_lt _ _ _ _ _ _ hb hm hk
  have hw (n k : Nat) (hn : n<(parameters i).N) (hk : k<(parameters i).K) :
      readWord prepared i.weight (k+n*(parameters i).K)=readWord before i.weight (k+n*(parameters i).K) := by
    apply prep.originalWords i.weight (by simp [Contract.Inputs.reads])
    rw [f.wCount]
    change k+n*(Checked.shape i).K<(Checked.shape i).N*(Checked.shape i).K
    dsimp [parameters] at hn hk
    nlinarith
  constructor
  · intro m k hm hk
    simp only [preparedInputs,originalInputs,inputWords,hx m k hm hk]
  · intro k hk
    simp only [preparedInputs,originalInputs,inputWords,prep.castOne k hk]
  · intro k hk
    simp only [preparedInputs,originalInputs,inputWords,prep.castOne k hk]
  · intro n k hn hk
    simp only [preparedInputs,originalInputs,inputWords,hw n k hn hk]
  · intro n k hn hk
    simp only [preparedInputs,originalInputs,inputWords,hw n k hn hk]

theorem pointer_exact (t : OriginalTensor) (ordinary : RmsMatmulRbe.Checked.Ordinary t)
    (j : Nat) (hj : j<t.metadata.numel) : (RmsMatmulRbe.Checked.pointer64 t j).toNat=address t j :=
  (RmsMatmulRbe.Checked.pointer_exact t ordinary j hj).1

#axiomsClean state_inputs
#axiomsClean state_bytes
#axiomsClean original_agreement
#specNonCircular originalInputs avoiding [FP.Structural.exec,KernelGraph.output,KernelSource.scatter]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelBinding

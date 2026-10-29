import bench.repairs.tritonbench_g.matmul_triton_autotune.Kernels
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Tensor

/-! Kernel admission and ordinary Tensor/launch observations. Generic Matrix
and FP16 output bindings are reused read-only from the accepted matmul task;
its kernel, guard bounds and SM90 floating graph are not this specification. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
noncomputable section

def metadataChecks (a b : Matrix) : Bool := decide (
  a.tensor.metadata.layout = .strided ∧ b.tensor.metadata.layout = .strided ∧
  a.tensor.metadata.shape = [a.rows,a.columns] ∧
  b.tensor.metadata.shape = [b.rows,b.columns] ∧
  a.tensor.metadata.dtype = .floating .fp32 ∧ b.tensor.metadata.dtype = .floating .fp32 ∧
  a.tensor.metadata.device.kind = "cuda" ∧ b.tensor.metadata.device.kind = "cuda" ∧
  a.tensor.metadata.device = b.tensor.metadata.device ∧
  a.columns = b.rows ∧ a.tensor.metadata.isContiguous = Bool.true ∧
  a.tensor.metadata.isNeg = Bool.false ∧ b.tensor.metadata.isNeg = Bool.false ∧
  (a.tensor.metadata.numel = 0 ∨ inputBase a % 4 = 0) ∧
  (b.tensor.metadata.numel = 0 ∨ inputBase b % 4 = 0))

abbrev InputContract (a b : Matrix) := metadataChecks a b = Bool.true

/-- Dynamic classification rejects non-Tensors, non-rank-two metadata,
negative signed strides, unresolved conjugate views and non-builtin strings
before producing Matrix/activation values. Nat strides represent exactly the
nonnegative branch; their address meaning is supplied by MatrixLaws. -/
inductive Arguments where
  | unsupported
  | matrices (a b : Matrix) (activation : String)

def checkArguments : Arguments → Bool
  | .unsupported => Bool.false
  | .matrices a b _ => metadataChecks a b

def Admitted : Arguments → Prop
  | .unsupported => False
  | .matrices a b _ => InputContract a b

theorem checkArguments_iff (args : Arguments) : checkArguments args = Bool.true ↔ Admitted args := by
  cases args <;> simp [checkArguments, Admitted, InputContract]

/-- Actual selected source configuration, independent of payloads. -/
structure Selection where
  target : String
  config : Whole.Config
  supported : target = "cuda" ∨ target = "hip"
  selected : config ∈ Whole.configurations target

def request (a b : Matrix) (activation : String) (selection : Selection) : Whole.Request where
  A := "a"; B := "b"; C := "c"
  rows := a.rows; inner := a.columns; innerB := b.rows; columns := b.columns
  strideAM := a.strideRow; strideAK := a.strideColumn
  strideBK := b.strideRow; strideBN := b.strideColumn
  aContiguous := a.tensor.metadata.isContiguous
  activation := activation
  target := selection.target; config := selection.config

/-- Ordinary signed metadata and successfully allocated backing-storage sizes.
No payload, padded source address or kernel result is constrained here. -/
structure TensorBinding (x : Matrix) : Prop extends MatrixLaws x where
  allocationBytes : x.storageBytes < 2^63
  strideRow : x.strideRow < 2^63
  strideColumn : x.strideColumn < 2^63

structure OutputBinding (c : OutputTensor) (a b : Matrix) : Prop extends OutputLaws c a b where
  allocationBytes : 2*c.storageElements < 2^63

/-- Actual complete launch order and ordinary program-id width. Runtime
 grid-product compatibility remains the separately deferred project item. -/
structure Schedule (r : Whole.Request) where
  order : List Nat
  complete : order.Perm (List.range (Whole.programs r))
  programIdWidth : Whole.programs r < 2^31

theorem admitted (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract a b) :
    a.columns = b.rows ∧ a.tensor.metadata.isContiguous = Bool.true ∧
    a.tensor.metadata.isNeg = Bool.false ∧ b.tensor.metadata.isNeg = Bool.false ∧
    inputBase a % 4 = 0 ∧ inputBase b % 4 = 0 := by
  simp only [InputContract, metadataChecks, decide_eq_true_eq] at h
  obtain ⟨_,_,_,_,_,_,_,_,_,dims,cont,na,nb,aa,ab⟩ := h
  refine ⟨dims,cont,na,nb,?_,?_⟩
  · rcases aa with hz | hp
    · simp [inputBase, al.emptyPointer hz]
    · exact hp
  · rcases ab with hz | hp
    · simp [inputBase, bl.emptyPointer hz]
    · exact hp

#axiomsClean checkArguments_iff
#axiomsClean admitted
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel

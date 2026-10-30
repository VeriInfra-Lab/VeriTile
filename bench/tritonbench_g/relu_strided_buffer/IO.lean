import bench.tritonbench_g.relu_strided_buffer.Proofs.RawKernel
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WholeReal
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WholeFP32
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.ReluIO
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer SignedView Whole
noncomputable section

/-- Actual source storage and logical channels are FP32. Signed strides and
storage offsets remain explicit, with no contiguity restriction or copying. -/
def inputChannels (x : SignedView.View) (n : Nat) : List IOChannel := [⟨.fp32,[n],x.region⟩]
def outputChannels (y : SignedView.View) (n : Nat) : List IOChannel := [⟨.fp32,[n],y.region⟩]
abbrev Inputs (α : Type) (x : SignedView.View) (n : Nat) := ChannelValues α (inputChannels x n)
abbrev Outputs (α : Type) (y : SignedView.View) (n : Nat) := ChannelValues α (outputChannels y n)

def element (values : Inputs α x n) (i : Fin n) : α :=
  values ⟨0,by simp [inputChannels]⟩ (i,PUnit.unit)

def pointwise (f : α → α) (x y : SignedView.View) (n : Nat)
    (values : Inputs α x n) : Outputs α y n :=
  Fin.cases (fun j => f (element values j.1)) (fun o => Fin.elim0 o)

/-- Independent mathematical ReLU. -/
def mathematical (x y : SignedView.View) (n : Nat) : Inputs ℝ x n → Outputs ℝ y n :=
  pointwise (fun z => max 0 z) x y n

/-- Independent ordered FP32 selection: NaNs, either zero sign and negative
words return positive zero; positive nonzero words remain bit-for-bit intact. -/
def nativeRelu (word : FP.Value .fp32) : FP.Value .fp32 :=
  if word.isNaN || word.isZero || word.sign then FP.Value.zero .fp32 else word

def nativeTarget (x y : SignedView.View) (n : Nat) :
    Inputs (FP.Value .fp32) x n → Outputs (FP.Value .fp32) y n := pointwise nativeRelu x y n

def memory (x y : SignedView.View) (n : Nat) : MemoryIO α where
  inputs := inputChannels x n
  outputs := outputChannels y n
  buffers := [x.region,y.region]
  read := Fin.cases (fun _ j => x.address j.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => y.address j.1.val) (fun o => Fin.elim0 o)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Original capped grid metadata and original one-tile/grid-stride source
selection. The candidate is only the existing unused integer-count fold. -/
def invocation (M : Algebra α) (folded : Bool) (x y : SignedView.View) (n pid : Nat) :
    Invocation α where
  kernel := if folded then foldedKernel (parameters (Raw.arguments x y n))
    else kernel (parameters (Raw.arguments x y n))
  backend := M
  pids := fun ax => if ax=0 then pid else 0
  numPids := some (fun ax => if ax=0 then (Raw.arguments x y n).C else 1)

def kernelIO (M : Algebra α) (folded : Bool) (x y : SignedView.View) (n : Nat) : ProgramIO α α where
  toMemoryIO := memory x y n
  schedule := fun _ => (List.range (Raw.arguments x y n).C).map (invocation M folded x y n)
  encode := id
  observe := fun _ _ => id

def candidate (M : Algebra α) (x y : SignedView.View) (n : Nat) :
    ProgramIO.Variant (kernelIO M Bool.false x y n) where
  schedule := fun _ => (List.range (Raw.arguments x y n).C).map (invocation M Bool.true x y n)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.ReluStridedBuffer.ReluIO

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.LogicalView
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer SignedView Whole
noncomputable section

def vector (v : SignedView.View) (n : Nat) : ReluIO.Inputs ℝ v n ≃ (Fin n → ℝ) where
  toFun := ReluIO.element
  invFun := fun xs => Fin.cases (fun j => xs j.1) (fun o => o.elim0)
  left_inv := by
    intro xs; funext o j
    have ho : o=0 := Fin.eq_zero o
    subst o
    rcases j with ⟨i,u⟩
    cases u
    rfl
  right_inv := fun _ => rfl

/-- Signed storage windows and both source branches remain in the unchanged core. -/
def reluIO (rest : Algebra ℝ) (x y : SignedView.View) (n : Nat) :
    LogicalProgramIO ℝ ℝ (Fin n → ℝ) (Fin n → ℝ) where
  core := ReluIO.kernelIO (realBackend rest) Bool.false x y n
  inputs := vector x n
  outputs := vector y n
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.LogicalView

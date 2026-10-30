import bench.tritonbench_g.softmax_triton3.Proofs.NativeGraph
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Memory.KernelSpec.Masked

/-! Public source and memory descriptors shared by the reader entries and their
proofs. Every descriptor names the actual source, buffers, windows, masks and
native observation. No execution or correctness proposition is stored here. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton3
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section
namespace Correct
/-- The FP32 row source without the optional additive-mask input. -/
def softmaxIO (out inp : RegionName) (n B : Nat) : MaskedKernelIO₁ where
  kernel := softmax_kernel out inp inp n n B Bool.false
  projection := by rfl
  inp := inp
  out := out
  B := B
  read := fun row => row*n
  write := fun row => row*n
  mask := fun _ j => j.val<n
  scratch := []

/-- The same FP32 row source with its active additive-mask input channel. -/
def softmaxMaskedIO (out inp mask : RegionName) (n B : Nat) : MaskedKernelIO₂ where
  kernel := softmax_kernel out inp mask n n B Bool.true
  projection := by rfl
  in1 := inp
  in2 := mask
  out := out
  B := B
  read1 := fun row => row*n
  read2 := fun row => row*n
  write := fun row => row*n
  mask := fun _ j => j.val<n
  scratch := []

end Correct
namespace FPEquiv
/-- Selected FP32 source and its row channels. Structural `.real` is this
source's erased cell tag; the quotation retains every explicit FP32 operation.
The optional second channel is inactive when no additive mask is provided.
Observation uses the actual selected native owner of each output column. -/
def softmaxIO (inst : Instructions) (profile : Profile) (out inp : RegionName)
    (mask : Option RegionName) (rows n B p : Nat) : ProgramIO Word Node where
  inputs := [⟨.real,[B],inp⟩,⟨.real,[B],mask.getD inp⟩]
  outputs := [⟨.real,[B],out⟩]
  buffers := [inp,mask.getD inp,out]
  read := Fin.cases (fun _ j => p*n+j.1.val)
    (Fin.cases (fun _ j => p*n+j.1.val) (fun i => Fin.elim0 i))
  inputMask := Fin.cases (fun _ j => j.1.val<n)
    (Fin.cases (fun _ j => j.1.val<n ∧ mask.isSome=Bool.true) (fun i => Fin.elim0 i))
  write := Fin.cases (fun _ j => p*n+j.1.val) (fun o => Fin.elim0 o)
  outputMask := Fin.cases (fun _ j => j.1.val<n) (fun o => Fin.elim0 o)
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [{
    kernel := softmax_kernel out inp (mask.getD inp) n n B mask.isSome
    backend := algebra inst profile.vector
    pids := fun ax => if ax=0 then p else 0
    numPids := some (fun ax => if ax=0 then rows else 1) }]
  encode := atom
  observe := Fin.cases (fun j value => value (owner profile.vector B j.1.val))
    (fun o => Fin.elim0 o)

end FPEquiv
end
end VeriTile.Bench.Repairs.SoftmaxTriton3

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Logical
/-- Logical active-column access into the complete B-lane row. The underlying
input/output domains and all inactive entries remain unchanged. -/
def active {n B : Nat} (cover : n ≤ B) (xs : Fin B → ℝ) (j : Fin n) : ℝ :=
  xs (Fin.castLE cover j)
end VeriTile.Bench.Repairs.SoftmaxTriton3.Logical

import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Interface
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelProfile
import VeriTile.Triton.Float.AddressedInputsIO

/-! Native words retain the old entry's source-address-labelled input leaves,
runtime scalar leaves, reduction topology and selected FMA/operand schedule.
No equality of native floating arithmetic with Real arithmetic is asserted. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native
open VeriTile.Triton FP.Structural KernelProfile KernelContract
noncomputable section
set_option maxHeartbeats 4000000

/-- Match the existing native entry's signed-label region interpretation,
including its original typed-read default when a floating view aliases it. -/
def encode (p : Parameters) (r : RegionName) (i : Nat) (x : Word) : Compiler.Term :=
  if r = Region.cast p.labels then defaultValue .real else .leaf r i x

def forwardIO (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat) : AddressedProgramIO Word Compiler.Term :=
  { Forward.memory Word p row block with
    schedule := fun _ => [invocation (Compiler.algebra R) p (KernelProfile.arguments v) Bool.false row block]
    encodeAt := encode p
    observe := fun _ _ => Compiler.eval I S }
def backwardIO (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat) : AddressedProgramIO Word Compiler.Term :=
  { Backward.memory Word p row block with
    schedule := fun _ => [invocation (Compiler.algebra R) p (KernelProfile.arguments v) Bool.true row block]
    encodeAt := encode p
    observe := fun _ _ => Compiler.eval I S }

def forwardInputs (p : Parameters) (row : Nat) (values : Forward.Inputs Word p) : Forward.Inputs Compiler.Term p :=
  fun i => match i with
  | ⟨0, _⟩ => fun j => encode p p.logits (p.logitsBase + row * p.logitsStride + j.1.val) (Forward.logit values j.1)
  | ⟨1, _⟩ => fun _ => Forward.label values
  | ⟨n + 2, h⟩ => False.elim (by simp [Forward.inputs] at h)
def backwardInputs (p : Parameters) (row : Nat) (values : Backward.Inputs Word p) : Backward.Inputs Compiler.Term p :=
  fun i => match i with
  | ⟨0, _⟩ => fun j => encode p p.logits (p.logitsBase + row * p.logitsStride + j.1.val) (Backward.logit values j.1)
  | ⟨1, _⟩ => fun _ => Backward.label values
  | ⟨2, _⟩ => fun _ => encode p p.lse (p.lseBase + row) (Backward.lse values)
  | ⟨3, _⟩ => fun _ => encode p p.upstream (p.upstreamBase + row * p.upstreamStride) (Backward.upstream values)
  | ⟨n + 4, h⟩ => False.elim (by simp [Backward.inputs] at h)

def forwardResult (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat)
    (values : Forward.Inputs Word p) : Forward.Outputs Word p :=
  let x := Forward.extend (forwardInputs p row values)
  let v := KernelProfile.arguments v
  Forward.result p (Compiler.eval I S (Ordered.lse (Compiler.algebra R) p v block x))
    (Compiler.eval I S (Ordered.loss (Compiler.algebra R) p v block x (Forward.label values)))
    (Compiler.eval I S (Ordered.zloss (Compiler.algebra R) p v block x (Forward.label values)))
def backwardResult (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat)
    (values : Backward.Inputs Word p) : Backward.Outputs Word p :=
  let data := backwardInputs p row values
  Backward.result p (fun lane => Compiler.eval I S (Ordered.gradient (Compiler.algebra R)
    p (KernelProfile.arguments v) block (Backward.extend data) (Backward.label values)
    (Backward.lse data) (Backward.upstream data) lane))

end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native

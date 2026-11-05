import VeriTile.Triton.Float.ProfiledRewrite
import bench.tritonbench_g.softmax_triton3.IO
import VeriTile.Triton.Float.ProgramEquivalence

/-! Pure strided source-pair IO on the original unrestricted scalar domain. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
open VeriTile.Triton FP.Structural
noncomputable section

/-- The original kernel's public typed input/output windows. All raw scalar
payloads are admitted; this is metadata, not a numerical input predicate. -/
structure Layout where
  out : RegionName
  inp : RegionName
  mask : RegionName
  stride : Nat
  n : Nat
  B : Nat
  hasMask : Bool
  p : Nat

structure IOProgram where
  kernel : ComputeKernel
  layout : Layout

def invocation {α : Type} (M : Algebra α) (program : IOProgram) : Invocation α where
  kernel := program.kernel
  backend := FP.ProfiledRewrite.engine M
  clearRegisters := Bool.true
  pids := fun ax => if ax = 0 then program.layout.p else 0

/-- Exactly the original active strided input/output cells. FP32 source
precision stays in the concrete kernel; `.real` is the opaque scalar cell tag. -/
@[reducible] def sourceIO {α : Type} (M : Algebra α) (program : IOProgram) : ProgramIO α α where
  inputs := [⟨.real,[program.layout.B],program.layout.inp⟩,⟨.real,[program.layout.B],program.layout.mask⟩]
  outputs := [⟨.real,[program.layout.B],program.layout.out⟩]
  scratch := []
  buffers := [program.layout.inp,program.layout.mask,program.layout.out]
  read := Fin.cases (fun _ j => program.layout.p*program.layout.stride+j.1.val)
    (Fin.cases (fun _ j => program.layout.p*program.layout.stride+j.1.val) (fun i => Fin.elim0 i))
  inputMask := Fin.cases (fun _ j => j.1.val < program.layout.n)
    (Fin.cases (fun _ j => j.1.val < program.layout.n ∧ program.layout.hasMask = Bool.true)
      (fun i => Fin.elim0 i))
  write := Fin.cases (fun _ j => program.layout.p*program.layout.stride+j.1.val) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ j => j.1.val < program.layout.n) (fun i => Fin.elim0 i)
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [invocation M program]
  encode := id
  observe := fun _ _ => id

def sourceVariant {α : Type} (M : Algebra α) (lhs rhs : IOProgram) :
    ProgramIO.Variant (sourceIO M lhs) where
  schedule := fun _ => [invocation M rhs]
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Pair

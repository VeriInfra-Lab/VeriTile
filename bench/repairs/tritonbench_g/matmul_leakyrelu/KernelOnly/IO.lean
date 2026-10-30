import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.KernelInterface
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.matmul_leakyrelu.Proofs.RealModel
import VeriTile.Triton.Float.LogicalProgramIO


/-! Complete matrix channels and the existing contracted source profile.
FP32 inputs/accumulation and FP16 result stores are retained. All legal grid
orders and the original read-only A/B alias scope are explicit. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

def inputs (p : KernelInterface.Parameters) : List IOChannel :=
  [⟨.fp32,[p.rows,p.inner],p.A⟩,⟨.fp32,[p.inner,p.columns],p.B⟩]
def outputs (p : KernelInterface.Parameters) : List IOChannel :=
  [⟨.fp16,[p.rows,p.columns],p.C⟩]
abbrev Inputs (α : Type) (p : KernelInterface.Parameters) := ChannelValues α (inputs p)
abbrev Outputs (α : Type) (p : KernelInterface.Parameters) := ChannelValues α (outputs p)

def matrixA {α : Type} {p : KernelInterface.Parameters} (vals : Inputs α p)
    (i : Fin p.rows) (k : Fin p.inner) : α := vals (0 : Fin 2) (i,k,())
def matrixB {α : Type} {p : KernelInterface.Parameters} (vals : Inputs α p)
    (k : Fin p.inner) (j : Fin p.columns) : α := vals (1 : Fin 2) (k,j,())

def memoryIO (α : Type) (p : KernelInterface.Parameters) : MemoryIO α where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.A,p.B,p.C]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.1.val*p.strideAM+j.2.1.val*p.strideAK
    | ⟨1,_⟩ => fun j => j.1.val*p.strideBK+j.2.1.val*p.strideBN
    | ⟨n+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun j => Whole.outputOffset p.request j.1.val j.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

def job {α : Type} (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (pid : Nat) : Invocation α where
  kernel := Whole.contractedKernel p.request
  backend := FPFused.backend M D
  pids := fun ax => if ax=0 then pid else 0
  numPids := some (fun ax => if ax=0 then Whole.programs p.request else 1)

def io {α : Type} (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : Schedule p.request) : ProgramIO α α where
  toMemoryIO := memoryIO α p
  schedule := fun _ => order.order.map (job M D p)
  encode := id
  observe := fun _ _ => id

def realReference (p : KernelInterface.Parameters) (vals : Inputs ℝ p) : Outputs ℝ p
  | ⟨0,_⟩ => fun j => Whole.mathematical p.request (matrixA vals) (matrixB vals) j.1 j.2.1
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

/-- Independent ordered K32 recurrence, activation and final half conversion.
The primitive parameters retain the original selected numerical boundary. -/
def nativeReference {α : Type} [Inhabited α] (Q : Whole.Contracted.Numerics α)
    (p : KernelInterface.Parameters) (vals : Inputs α p) : Outputs α p
  | ⟨0,_⟩ => fun j => Whole.Contracted.logicalReference Q p.request
      (matrixA vals) (matrixB vals) j.1 j.2.1
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Interface

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Original
open VeriTile.Triton FP.Structural
open Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

def job {α : Type} (M : Algebra α) (D : FPValue.Dot α)
    (p : KernelInterface.Parameters) (pid : Nat) : Invocation α where
  kernel := Whole.kernel p.request
  backend := FPValue.backend M D
  pids := fun ax => if ax=0 then pid else 0
  numPids := some (fun ax => if ax=0 then Whole.programs p.request else 1)

def io {α : Type} (M : Algebra α) (D : FPValue.Dot α)
    (p : KernelInterface.Parameters) (order : Schedule p.request) : ProgramIO α α where
  toMemoryIO := Interface.memoryIO α p
  schedule := fun _ => order.order.map (job M D p)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Original

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
variable {α : Type} [Inhabited α]

def variant (M : Algebra α) (D : FPValue.Dot α) (F : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : Schedule p.request) :
    ProgramIO.Variant (Original.io M D p order) where
  schedule := (Interface.io M F p order).schedule
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Pair


namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.ReaderIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
structure Inputs (p : KernelInterface.Parameters) where
  A : Fin p.rows → Fin p.inner → ℝ
  B : Fin p.inner → Fin p.columns → ℝ

def inputs (p : KernelInterface.Parameters) : Interface.Inputs ℝ p ≃ Inputs p where
  toFun := fun xs => ⟨(fun r k => xs (0 : Fin 2) (r,k,())), (fun k c => xs (1 : Fin 2) (k,c,()))⟩
  invFun := fun xs => Fin.cases (fun j => xs.A j.1 j.2.1)
    (Fin.cases (fun j => xs.B j.1 j.2.1) (fun i => i.elim0))
  left_inv := by intro xs; funext i j; fin_cases i <;> rcases j with ⟨a,b,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputs (p : KernelInterface.Parameters) : Interface.Outputs ℝ p ≃
    (Fin p.rows → Fin p.columns → ℝ) where
  toFun := fun ys r c => ys (0 : Fin 1) (r,c,())
  invFun := fun ys => Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by intro ys; funext i j; fin_cases i; rcases j with ⟨a,b,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- Complete original source schedule, with bijective logical matrix coordinates. -/
def view (p : KernelInterface.Parameters) (order : Schedule p.request) : LogicalProgramIO ℝ ℝ (Inputs p)
    (Fin p.rows → Fin p.columns → ℝ) where
  core := Original.io (Whole.idealAlgebra id) Whole.idealDot p order
  inputs := inputs p
  outputs := outputs p
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.ReaderIO

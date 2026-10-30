import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.matmul_triton_autotune.Proofs.KernelRaw
import VeriTile.Triton.Float.ProgramIO

/-! Complete selected matmul grid, with strided FP32 inputs, source-prescribed
FP16 output, every selected configuration and explicit program permutation. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
open VeriTile.Triton FP.Structural Whole
noncomputable section

def inputs (g : Geometry) : List IOChannel :=
  [⟨.fp32, [g.rows, g.inner], g.A⟩, ⟨.fp32, [g.inner, g.columns], g.B⟩]
def outputs (g : Geometry) : List IOChannel := [⟨.fp16, [g.rows, g.columns], g.C⟩]
abbrev Inputs (A : Type) (g : Geometry) := ChannelValues A (inputs g)
abbrev Outputs (A : Type) (g : Geometry) := ChannelValues A (outputs g)

def left (values : Inputs A g) (i : Fin g.rows) (k : Fin g.inner) : A :=
  values ⟨0, by simp [inputs]⟩ (i, k, PUnit.unit)
def right (values : Inputs A g) (k : Fin g.inner) (j : Fin g.columns) : A :=
  values ⟨1, by simp [inputs]⟩ (k, j, PUnit.unit)
def result (g : Geometry) (f : Fin g.rows → Fin g.columns → A) : Outputs A g :=
  fun o => match o with
  | ⟨0, _⟩ => fun j => f j.1 j.2.1
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def read (g : Geometry) (i : Fin (inputs g).length) : TileIndex (inputs g)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun j => j.1.val * g.strideAM + j.2.1.val * g.strideAK
  | ⟨1, _⟩ => fun j => j.1.val * g.strideBK + j.2.1.val * g.strideBN
  | ⟨n + 2, h⟩ => False.elim (by simp [inputs] at h)
def write (g : Geometry) (o : Fin (outputs g).length) : TileIndex (outputs g)[o].shape → Nat :=
  match o with
  | ⟨0, _⟩ => fun j => max g.columns 1 * j.1.val + j.2.1.val
  | ⟨n + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (g : Geometry) : MemoryIO A where
  inputs := inputs g
  outputs := outputs g
  scratch := []
  buffers := [g.A, g.B, g.C]
  read := fun i _ => read g i
  inputMask := fun _ _ _ => True
  write := fun o _ => write g o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry) (pid : Nat) : Invocation A where
  kernel := sourceKernel g.request
  backend := FPValue.backend M D
  pids := fun ax => if ax = 0 then pid else 0
  numPids := some (fun ax => if ax = 0 then programs g.request else 1)

def programIO (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry) (ids : List Nat) : ProgramIO A A :=
  { memory A g with
    schedule := fun _ => ids.map (invocation M D g)
    encode := id
    observe := fun _ _ => id }

end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete input channels reindexed as two ordinary logical matrices. -/
def matrixInputs (g : Geometry) :
    Inputs ℝ g ≃ ((Fin g.rows → Fin g.inner → ℝ) × (Fin g.inner → Fin g.columns → ℝ)) where
  toFun := fun xs => (fun row col => id (xs (0 : Fin 2) (row,col,PUnit.unit)),
    fun row col => id (xs (1 : Fin 2) (row,col,PUnit.unit)))
  invFun := fun ab => Fin.cases (fun j => id (ab.1 j.1 j.2.1))
    (Fin.cases (fun j => id (ab.2 j.1 j.2.1)) (fun port => Fin.elim0 port))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> rcases j with ⟨row,col,⟨⟩⟩ <;> rfl
  right_inv := by
    intro ab
    apply Prod.ext <;> funext row col <;> rfl

/-- Complete sole output channel reindexed as an ordinary logical matrix. -/
def matrixOutputs (g : Geometry) :
    Outputs ℝ g ≃ (Fin g.rows → Fin g.columns → ℝ) where
  toFun := fun ys row col => id (ys (0 : Fin 1) (row,col,PUnit.unit))
  invFun := fun f => Fin.cases (fun j => id (f j.1 j.2.1)) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    rfl
  right_inv := by
    intro f
    funext row col
    rfl

end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
open VeriTile.Triton FP.Structural

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (M : Algebra ℝ) (D : FPValue.DotAcc ℝ) (g : Geometry) (ids : List Nat) : LogicalProgramIO ℝ ℝ
    ((Fin g.rows → Fin g.inner → ℝ) × (Fin g.inner → Fin g.columns → ℝ)) (Fin g.rows → Fin g.columns → ℝ) where
  core := programIO M D g ids
  inputs := matrixInputs g
  outputs := matrixOutputs g

end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface

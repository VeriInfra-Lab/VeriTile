import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Raw
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Native
import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealModel
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Shared typed finite matrices, original strides and complete launch schedule. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO
open VeriTile.Triton FP.Structural FPValue
noncomputable section
variable {α : Type} [Inhabited α]

def cellType : Storage → TileDType | .half => .fp16 | .e4m3 => .f8e4

def payload (st : Storage) : Value α (cellType st) → α := by cases st <;> exact id

def tagged (st : Storage) : α → Value α (cellType st) := by cases st <;> exact id

def invocation (g : Geometry) (M : Algebra α) (D : DotAcc α) (pid : Nat) : Invocation α where
  kernel := Source.typed g.request.parameters g.storage
  backend := backend M D
  pids := fun dim => if dim=0 then pid else 0
  numPids := some (fun dim => if dim=0 then programs g.request else 1)

def inputs (g : Geometry) : List IOChannel :=
  [⟨cellType g.storage,[g.rows,g.inner],g.A⟩,⟨cellType g.storage,[g.inner,g.columns],g.B⟩]
def outputs (g : Geometry) : List IOChannel := [⟨cellType g.storage,[g.rows,g.columns],g.C⟩]

def io (g : Geometry) (M : Algebra α) (D : DotAcc α) (ids : List Nat) : ProgramIO α α where
  inputs := inputs g
  outputs := outputs g
  scratch := []
  buffers := [g.A,g.B,g.C]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.1.val*g.strideAM+lane.2.1.val*g.strideAK
    | ⟨1,_⟩ => fun lane => lane.1.val*g.strideBK+lane.2.1.val*g.strideBN
    | ⟨n+2,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => max g.columns 1*lane.1.val+lane.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => ids.map (invocation g M D)
  encode := id
  observe := fun _ _ => id

def arrayA (g : Geometry) (vals : ChannelValues α (inputs g)) (r : Fin g.rows) (k : Fin g.inner) : α :=
  payload g.storage (vals ⟨0,by simp [inputs]⟩ (r,k,()))
def arrayB (g : Geometry) (vals : ChannelValues α (inputs g)) (k : Fin g.inner) (n : Fin g.columns) : α :=
  payload g.storage (vals ⟨1,by simp [inputs]⟩ (k,n,()))

def expected (g : Geometry)
    (f : (Fin g.rows → Fin g.inner → α) → (Fin g.inner → Fin g.columns → α) → Fin g.rows → Fin g.columns → α)
    (vals : ChannelValues α (inputs g)) : ChannelValues α (outputs g) :=
  fun port => match port with
    | ⟨0,_⟩ => fun lane => tagged g.storage (f (arrayA g vals) (arrayB g vals) lane.1 lane.2.1)
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

/-- Logical matrices preserve every operand cell, including both storage modes. -/
abbrev MatrixInputs (g : Geometry) :=
  (Fin g.rows → Fin g.inner → ℝ) × (Fin g.inner → Fin g.columns → ℝ)

def matrixInputs (g : Geometry) : ChannelValues ℝ (inputs g) ≃ MatrixInputs g where
  toFun xs := (arrayA g xs,arrayB g xs)
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => tagged g.storage (xs.1 i.1 i.2.1)
    | ⟨1,_⟩ => fun i => tagged g.storage (xs.2 i.1 i.2.1)
  left_inv := by
    have inv : ∀ st (x : Value ℝ (cellType st)), tagged st (payload st x)=x := by
      intro st; cases st <;> intro x <;> rfl
    intro xs; funext port i; fin_cases port <;> rcases i with ⟨a,b,u⟩ <;> cases u <;>
      exact inv g.storage _
  right_inv := by
    have inv : ∀ st (x : ℝ), payload st (tagged st x)=x := by
      intro st; cases st <;> intro x <;> rfl
    intro xs; rcases xs with ⟨a,b⟩; apply Prod.ext <;> funext i j <;> exact inv g.storage _

def matrixOutputs (g : Geometry) : ChannelValues ℝ (outputs g) ≃ (Fin g.rows → Fin g.columns → ℝ) where
  toFun xs i j := payload g.storage (xs ⟨0,by simp [outputs]⟩ (i,j,()))
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => tagged g.storage (xs i.1 i.2.1)
  left_inv := by
    have inv : ∀ st (x : Value ℝ (cellType st)), tagged st (payload st x)=x := by
      intro st; cases st <;> intro x <;> rfl
    intro xs; funext port i; fin_cases port; rcases i with ⟨a,b,u⟩; cases u; exact inv g.storage _
  right_inv := by
    have inv : ∀ st (x : ℝ), payload st (tagged st x)=x := by
      intro st; cases st <;> intro x <;> rfl
    intro xs; funext i j; exact inv g.storage _

/-- The exact original complete schedule, with curried mathematical matrices. -/
def logicalRealIO (g : Geometry) (ids : List Nat) :
    LogicalProgramIO ℝ ℝ (MatrixInputs g) (Fin g.rows → Fin g.columns → ℝ) where
  core := io g KernelRealModel.ideal KernelRealModel.idealDot ids
  inputs := matrixInputs g
  outputs := matrixOutputs g

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO

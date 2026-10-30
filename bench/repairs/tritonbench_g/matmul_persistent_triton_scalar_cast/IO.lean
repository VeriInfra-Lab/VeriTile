import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.RawKernel
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
variable {α : Type}

namespace PublicData
abbrev inputs (p : Core.Parameters) : List IOChannel :=
  [⟨Core.dtype p.problem.profile,[p.problem.M,p.problem.K],"A"⟩,
   ⟨Core.dtype p.problem.profile,[p.problem.K,p.problem.N],"B"⟩]
abbrev outputs (p : Core.Parameters) : List IOChannel :=
  [⟨Core.dtype p.problem.profile,[p.problem.M,p.problem.N],"C"⟩]

/-- Only the payload carrier varies; storage precision remains source-selected. -/
def payload (profile : Storage) : Value α (Core.dtype profile) → α :=
  match profile with | .fp16 => id | .f8e4 => id

def value (profile : Storage) : α → Value α (Core.dtype profile) :=
  match profile with | .fp16 => id | .f8e4 => id

def read (p : Core.Parameters) (i : Fin (inputs p).length) :
    TileIndex ((inputs p)[i]).shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => j.1.val*p.am+j.2.1.val*p.ak
  | ⟨1,_⟩ => fun j => j.1.val*p.bk+j.2.1.val*p.bn

def write (p : Core.Parameters) (i : Fin (outputs p).length) :
    TileIndex ((outputs p)[i]).shape → Nat :=
  match i with | ⟨0,_⟩ => fun j => j.1.val*p.cm+j.2.1.val*p.cn

def job (P : Source.Arithmetic α α) (p : Core.Parameters) (pid : Nat) : Invocation α :=
  { kernel := Core.source p, backend := Core.backend P,
    pids := fun ax => if ax=0 then pid else 0 }

abbrev io (P : Source.Arithmetic α α) (p : Core.Parameters) : ProgramIO α α where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := ["A","B","C"]
  read := fun i _ => read p i
  inputMask := fun _ _ _ => True
  write := fun i _ => write p i
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (List.range p.problem.grid).map (job P p)
  encode := id
  observe := fun _ _ => id

end PublicData

namespace Pair
open Core
def candidate (p : Parameters) : ComputeKernel :=
  TypedSource.reordered p.problem.profile "A" "B" "C" p.problem.M p.problem.N p.problem.K
    p.am p.ak p.bk p.bn p.cm p.cn 128 256 p.problem.profile.blockK 8 p.problem.SMS

def job (P : Source.Arithmetic α α) (p : Parameters) (pid : Nat) : Invocation α :=
  { PublicData.job P p pid with kernel := candidate p }

def candidateIO (P : Source.Arithmetic α α) (p : Parameters) : ProgramIO α α :=
  { PublicData.io P p with schedule := fun _ => (List.range p.problem.grid).map (job P p) }

structure Program (α : Type) where
  arithmetic : Source.Arithmetic α α
  parameters : Parameters
  positiveSms : 0<parameters.problem.SMS
  reordered : Bool

def Program.kernel (p : Program α) : ComputeKernel :=
  if p.reordered then candidate p.parameters else source p.parameters

def Program.schedule (p : Program α) : List (Invocation α) :=
  if p.reordered then (List.range p.parameters.problem.grid).map (job p.arithmetic p.parameters)
  else (List.range p.parameters.problem.grid).map (PublicData.job p.arithmetic p.parameters)

def Program.io (p : Program α) : ProgramIO α α :=
  if p.reordered then candidateIO p.arithmetic p.parameters else PublicData.io p.arithmetic p.parameters

end Pair
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
noncomputable section

/-- Selected floating storage payloads and their exact inverse. -/
def payloadEquiv (profile : VeriTile.Bench.Repairs.MatmulPersistentTriton.Storage) :
    Value ℝ (Core.dtype profile) ≃ ℝ where
  toFun := payload profile
  invFun := value profile
  left_inv := by cases profile <;> exact fun _ => rfl
  right_inv := by cases profile <;> exact fun _ => rfl

/-- Complete input channels reindexed as two ordinary logical matrices. -/
def matrixInputs (p : Core.Parameters) :
    ChannelValues ℝ (inputs p) ≃ ((Fin p.problem.M → Fin p.problem.K → ℝ) × (Fin p.problem.K → Fin p.problem.N → ℝ)) where
  toFun := fun xs => (fun row col => payload p.problem.profile (xs (0 : Fin 2) (row,col,PUnit.unit)),
    fun row col => payload p.problem.profile (xs (1 : Fin 2) (row,col,PUnit.unit)))
  invFun := fun ab => Fin.cases (fun j => value p.problem.profile (ab.1 j.1 j.2.1))
    (Fin.cases (fun j => value p.problem.profile (ab.2 j.1 j.2.1)) (fun port => Fin.elim0 port))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> rcases j with ⟨row,col,⟨⟩⟩ <;> exact (payloadEquiv p.problem.profile).symm_apply_apply _
  right_inv := by
    intro ab
    apply Prod.ext <;> funext row col <;> exact (payloadEquiv p.problem.profile).apply_symm_apply _

/-- Complete sole output channel reindexed as an ordinary logical matrix. -/
def matrixOutputs (p : Core.Parameters) :
    ChannelValues ℝ (outputs p) ≃ (Fin p.problem.M → Fin p.problem.N → ℝ) where
  toFun := fun ys row col => payload p.problem.profile (ys (0 : Fin 1) (row,col,PUnit.unit))
  invFun := fun f => Fin.cases (fun j => value p.problem.profile (f j.1 j.2.1)) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    exact (payloadEquiv p.problem.profile).symm_apply_apply _
  right_inv := by
    intro f
    funext row col
    exact (payloadEquiv p.problem.profile).apply_symm_apply _

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (P : Source.Arithmetic ℝ ℝ) (p : Core.Parameters) : LogicalProgramIO ℝ ℝ
    ((Fin p.problem.M → Fin p.problem.K → ℝ) × (Fin p.problem.K → Fin p.problem.N → ℝ)) (Fin p.problem.M → Fin p.problem.N → ℝ) where
  core := io P p
  inputs := matrixInputs p
  outputs := matrixOutputs p

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData

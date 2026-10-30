import bench.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeReal

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelInterface
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer

def buffer (v : SignedView.View) : HostPlan.Buffer :=
  ⟨v.region,v.base,[],[],none⟩

def arguments (input output : SignedView.View) (n : Nat) : HostPlan.Arguments :=
  HostPlan.rankOneArguments (buffer input) (buffer output) n input.stride output.stride

end VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelInterface

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section

def channels (region : RegionName) (n : Nat) : List IOChannel := [⟨.fp32,[n],region⟩]
abbrev Inputs (α : Type) (input : SignedView.View) (n : Nat) := ChannelValues α (channels input.region n)
def index (r : RegionName) (n : Nat) : (i : Fin (channels r n).length) →
    TileIndex (channels r n)[i].shape → Fin n
  | ⟨0,_⟩ => fun j => j.1
  | ⟨i+1,h⟩ => False.elim (by simp [channels] at h)
def inputValue {α : Type} {input : SignedView.View} {n : Nat} (xs : Inputs α input n) (i : Fin n) : α :=
  xs ⟨0,by simp [channels]⟩ (i,PUnit.unit)
def result {α : Type} {r : RegionName} {n : Nat} (values : Fin n → α) : ChannelValues α (channels r n)
  | ⟨0,_⟩ => fun j => values j.1
  | ⟨i+1,h⟩ => False.elim (by simp [channels] at h)

/-- Complete signed view offsets are formed before conversion to cell indices.
The source's logical tail mask selects exactly these n input/output cells. -/
def memory (α : Type) (input output : SignedView.View) (n : Nat) : MemoryIO α where
  inputs := channels input.region n
  outputs := channels output.region n
  scratch := []
  buffers := [input.region,output.region]
  read := fun i _ j => input.address (index input.region n i j).val
  inputMask := fun _ _ _ => True
  write := fun i _ j => output.address (index output.region n i j).val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Exact selected source and CTA geometry. The backend retains the sole
runtime power-base scalar slot; no literal or source precision is substituted. -/
def invocation {α : Type} (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n pid : Nat) : Invocation α where
  kernel := sourceKernel (parameters base (KernelInterface.arguments input output n)) candidate
  backend := M
  pids := fun ax => if ax=0 then pid else 0
  numPids := some (fun ax => if ax=0 then (KernelInterface.arguments input output n).C else 1)

def programIO {α : Type} (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) : ProgramIO α α :=
  { memory α input output n with
    schedule := fun _ => (List.range (KernelInterface.arguments input output n).C).map
      (invocation M base candidate input output n)
    encode := id
    observe := fun _ _ => id }
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.LogicalView
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section

def vector (r : RegionName) (n : Nat) :
    ChannelValues ℝ (Interface.channels r n) ≃ (Fin n → ℝ) where
  toFun := fun xs i => xs ⟨0,by simp [Interface.channels]⟩ (i,())
  invFun := Interface.result
  left_inv := by
    intro xs; funext o j
    have ho : o=⟨0,by simp [Interface.channels]⟩ := Fin.eq_zero o
    subst o
    rcases j with ⟨i,u⟩
    cases u
    rfl
  right_inv := fun _ => rfl

/-- Logical vector coordinates; the actual signed windows and capped schedule
are precisely the old core. Both maps are equivalences, including empty vectors. -/
def powerIO (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) :
    LogicalProgramIO ℝ ℝ (Fin n → ℝ) (Fin n → ℝ) where
  core := Interface.programIO (realBackend rest) base candidate input output n
  inputs := vector input.region n
  outputs := vector output.region n
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.LogicalView

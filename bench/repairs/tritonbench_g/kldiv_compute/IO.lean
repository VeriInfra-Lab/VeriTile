import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.kldiv_compute.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.ProgramIO
import bench.tritonbench_g.vector_addition.Proofs.HostGrid

namespace VeriTile.Bench.Repairs.KldivCompute.FloatingIO
open VeriTile.Triton

/-- Two masked FP32 input windows and one output window, for any program ID.
The unchanged typed source fixes load, divide, log, multiply and store precision. -/
def original (p : Parameters) : MaskedKernelIO₂ where
  kernel := KernelDSL.kernel p
  projection := by rfl
  in1 := p.x
  in2 := p.y
  out := p.output
  B := p.block
  read1 := fun pid => pid * p.block
  read2 := fun pid => pid * p.block
  write := fun pid => pid * p.block
  mask := fun pid i => pid * p.block + i.val < p.n

/-- The existing candidate has precisely the same public memory interface. -/
def commuted (p : Parameters) : MaskedKernelIO₂ :=
  (original p).withKernel (KernelDSL.commutedKernel p)

end VeriTile.Bench.Repairs.KldivCompute.FloatingIO

/-! Canonical whole-grid logical interface; the FP reader also retains its
per-program masked view above. Both use the same selected source geometry. -/
namespace VeriTile.Bench.Repairs.KldivCompute.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
open ValueProofs (gridCount)
noncomputable section

def invocation (M : Algebra α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := KernelDSL.kernel p
  backend := M
  clearRegisters := Bool.true
  pids := fun ax => if ax = 0 then pid else 0
  numPids := some (fun ax => if ax = 0 then gridCount p else 1)

abbrev inputChannels (p : Parameters) : List IOChannel :=
  [⟨.real, [p.n], p.x⟩, ⟨.real, [p.n], p.y⟩]

abbrev outputChannels (p : Parameters) : List IOChannel := [⟨.real, [p.n], p.output⟩]

def read (p : Parameters) (i : Fin (inputChannels p).length) :
    ChannelValues α (inputChannels p) → TileIndex (inputChannels p)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ | ⟨1, _⟩ => fun _ j => j.1.val
  | ⟨_+2, h⟩ => False.elim (by simpa using h)

def write (p : Parameters) (i : Fin (outputChannels p).length) :
    ChannelValues α (inputChannels p) → TileIndex (outputChannels p)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun _ j => j.1.val
  | ⟨_+1, h⟩ => False.elim (by simpa using h)

abbrev io (M : Algebra α) (p : Parameters) : ProgramIO α α where
  inputs := inputChannels p
  outputs := outputChannels p
  scratch := []
  buffers := [p.x, p.y, p.output]
  read := read p
  inputMask := fun _ _ _ => True
  write := write p
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range (gridCount p)).map (invocation M p)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.KldivCompute.PublicIO

namespace VeriTile.Bench.Repairs.KldivCompute.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
structure LogicalInputs (α : Type) (p : Parameters) where
  x : Fin p.n → α
  y : Fin p.n → α

def logicalInputs (α : Type) (p : Parameters) : ChannelValues α (inputChannels p) ≃ LogicalInputs α p where
  toFun xs := ⟨fun i => xs 0 (i,()), fun i => xs 1 (i,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun i => xs.x i.1
    | ⟨1,_⟩ => fun i => xs.y i.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext i <;> rcases i with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (α : Type) (p : Parameters) : ChannelValues α (outputChannels p) ≃ (Fin p.n → α) where
  toFun xs := fun i => xs 0 (i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun i => xs i.1
  left_inv := by intro xs; funext o; fin_cases o; funext i; rcases i with ⟨i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl
end
end VeriTile.Bench.Repairs.KldivCompute.PublicIO

namespace VeriTile.Bench.Repairs.KldivCompute.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
def logicalIO (M : Algebra α) (p : Parameters) : LogicalProgramIO α α (LogicalInputs α p) (Fin p.n → α) where
  core := io M p
  inputs := logicalInputs α p
  outputs := logicalOutputs α p
end
end VeriTile.Bench.Repairs.KldivCompute.PublicIO

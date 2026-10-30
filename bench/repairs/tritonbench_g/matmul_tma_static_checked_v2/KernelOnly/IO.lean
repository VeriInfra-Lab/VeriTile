import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Kernels
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO
open VeriTile.Triton FP.Structural FPValue
noncomputable section

abbrev inputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp16, [p.blockRows,p.blockInner], p.a⟩, ⟨.fp16, [p.blockInner,p.blockColumns], p.b⟩]
abbrev outputChannels (p : Parameters) (half : Bool) : List IOChannel :=
  [⟨(outputDType half).toTileDType, [p.blockRows,p.blockColumns], p.c⟩]

def read (p : Parameters) (i : Fin (inputChannels p).length) :
    ChannelValues α (inputChannels p) → TileIndex (inputChannels p)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun _ j => aOffset p j
  | ⟨1, _⟩ => fun _ j => bOffset p j
  | ⟨_+2, h⟩ => False.elim (by simpa using h)

def write (p : Parameters) (half : Bool) (i : Fin (outputChannels p half).length) :
    ChannelValues α (inputChannels p) → TileIndex (outputChannels p half)[i].shape → Nat :=
  match i with
  | ⟨0, _⟩ => fun _ j => cOffset p j
  | ⟨_+1, h⟩ => False.elim (by simpa using h)

def invocation (M : Algebra α) (D : Dot α) (p : Parameters) (half : Bool) : Invocation α where
  kernel := source p half
  backend := backend M D
  pids := fun _ => 0

abbrev io (M : Algebra α) (D : Dot α) (p : Parameters) (half : Bool) : ProgramIO α α where
  inputs := inputChannels p
  outputs := outputChannels p half
  scratch := []
  buffers := [p.a, p.b, p.c]
  read := read p
  inputMask := fun _ _ _ => True
  write := write p half
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation M D p half]
  encode := id
  observe := fun _ _ => id

def formula (p : Parameters) (half : Bool)
    (f : (AIndex p → α) → (BIndex p → α) → CIndex p → α)
    (xs : ChannelValues α (inputChannels p)) : ChannelValues α (outputChannels p half) := fun o =>
  match o with
  | ⟨0, _⟩ => fun j => ofFloat (outputDType half) (f (xs ⟨0, by simp⟩) (xs ⟨1, by simp⟩) j)
  | ⟨_+1, h⟩ => False.elim (by simpa using h)

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.IOPair
open VeriTile.Triton FP.Structural FPValue Host StaticChecked.Native
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def io (rest : Algebra Word) (I : Instructions) (profile : Profile) (h : Shape) (f : Flags) :=
  PublicIO.io (StaticChecked.Native.algebra rest I) (sourceDot I profile) (params h f) f.half

def variant (rest : Algebra Word) (I : Instructions) (profile : Profile) (h : Shape) (f : Flags) :
    ProgramIO.Variant (io rest I profile h f) where
  schedule := fun _ => [{ PublicIO.invocation (StaticChecked.Native.algebra rest I)
    (sourceDot I profile) (params h f) f.half with kernel := kernelDirectOutput h f }]
  observe := (io rest I profile h f).observe

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.IOPair

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Complete input channels reindexed as two ordinary logical matrices. -/
def matrixInputs (p : Parameters) (half : Bool) :
    ChannelValues ℝ (inputChannels p) ≃ ((Fin p.blockRows → Fin p.blockInner → ℝ) × (Fin p.blockInner → Fin p.blockColumns → ℝ)) where
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
def matrixOutputs (p : Parameters) (half : Bool) :
    ChannelValues ℝ (outputChannels p half) ≃ (Fin p.blockRows → Fin p.blockColumns → ℝ) where
  toFun := fun ys row col => toFloat (outputDType half) (ys (0 : Fin 1) (row,col,PUnit.unit))
  invFun := fun f => Fin.cases (fun j => ofFloat (outputDType half) (f j.1 j.2.1)) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    cases half <;> rfl
  right_inv := by
    intro f
    funext row col
    cases half <;> rfl

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (M : Algebra ℝ) (D : FPValue.Dot ℝ) (p : FPValue.Parameters) (half : Bool) : LogicalProgramIO ℝ ℝ
    ((Fin p.blockRows → Fin p.blockInner → ℝ) × (Fin p.blockInner → Fin p.blockColumns → ℝ)) (Fin p.blockRows → Fin p.blockColumns → ℝ) where
  core := io M D p half
  inputs := matrixInputs p half
  outputs := matrixOutputs p half

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO

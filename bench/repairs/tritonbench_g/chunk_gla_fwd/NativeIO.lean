import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativePorts
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsGeometry

/-! Public native IO for the existing complete FP32 producer and standalone
consumer. The source keeps its FP32 loads/stores, TF32 dot options, reduction
layout and runtime scalar ABI. The two attention channels partition one
tensor by its actual Structural storage tag; both observe the same FP32 words.
The masks use geometry and fixed store types only, never computed values.
-/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeIO
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word Expr producerInputs consumerInputs producerOutputs consumerOutputs)
noncomputable section
set_option maxRecDepth 30000

def scratchSize (c : Host.Dimensions) : Nat :=
  if c.heads*c.T=0 ∨ c.K≤256 then 0 else Host.scratchSize c

def producer (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    FillProgramIO Word Expr where
  inputs := producerInputs c
  outputs := producerOutputs c
  scratch := [⟨.real,[scratchSize c],"scratch"⟩]
  buffers := ["q","k","g","attention","scratch"]
  read := Fin.cases (fun _ i=>i.1.val)
    (Fin.cases (fun _ i=>i.1.val) (Fin.cases (fun _ i=>i.1.val) (fun i=>Fin.elim0 i)))
  inputMask := fun _ _ _=>True
  write := Fin.cases (fun _ i=>i.1.val) (Fin.cases (fun _ i=>i.1.val) (fun i=>Fin.elim0 i))
  outputMask := Fin.cases (fun _ i=>NativeTags.producerExplicit c i.1.val=Bool.false)
    (Fin.cases (fun _ i=>NativeTags.producerExplicit c i.1.val=Bool.true) (fun i=>Fin.elim0 i))
  scratchWrite := Fin.cases (fun _ i=>i.1.val) (fun i=>Fin.elim0 i)
  scratchMask := fun _ _ _=>True
  stages := fun _=>NativeLaunch.producer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag
  encode := NativeStaging.Expr.bits
  observe := fun _ _ value=>(NativeStaging.lower value).eval I

def consumer (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    FillProgramIO Word Expr where
  inputs := consumerInputs c
  outputs := consumerOutputs c
  scratch := []
  buffers := ["q","g","v","h","attention","o"]
  read := Fin.cases (fun _ i=>i.1.val) (Fin.cases (fun _ i=>i.1.val)
    (Fin.cases (fun _ i=>i.1.val) (Fin.cases (fun _ i=>i.1.val) (Fin.cases (fun _ i=>i.1.val) (fun i=>Fin.elim0 i)))))
  inputMask := fun _ _ _=>True
  write := Fin.cases (fun _ i=>i.1.val) (fun i=>Fin.elim0 i)
  outputMask := fun _ _ _=>True
  scratchWrite := fun channel=>Fin.elim0 channel
  scratchMask := fun channel=>Fin.elim0 channel
  stages := fun _=>NativeLaunch.consumer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag
  encode := NativeStaging.Expr.bits
  observe := fun _ _ value=>(NativeStaging.lower value).eval I

/-- The original ordered instruction recurrence, evaluated under the explicit
local ISA. There is no new optimization or equality with Real arithmetic. -/
def attention (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    ChannelValues Word (producerInputs c) → ChannelValues Word (producerOutputs c) :=
  fun values=>Fin.cases (fun i=>NativePorts.attentionWord c I w scale values i.1.val)
    (Fin.cases (fun i=>NativePorts.attentionWord c I w scale values i.1.val) (fun i=>Fin.elim0 i))

def output (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    ChannelValues Word (consumerInputs c) → ChannelValues Word (consumerOutputs c) :=
  fun values=>Fin.cases (fun i=>NativePorts.outputWord c I w scale values i.1) (fun i=>Fin.elim0 i)

#specNonCircular attention avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,FP.Structural.exec,
  NativeLaunch.producer,runFillSchedule]
#specNonCircular output avoiding [FPStages.execute,FPGrid.runJobs,FPGrid.invoke,FP.Structural.exec,
  NativeLaunch.consumer,runFillSchedule]
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeIO

import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.IO
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.NumericalReference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NativeIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option autoImplicit false

abbrev Word := BitVec 32

/-- The actual host/JIT scale word is used at the source's nonzero scalar
literal site; zero denotes FP32 +0. No equality to ideal 1/sqrt(D) is assumed. -/
def literal (actualScale : Word) (x : ℝ) : Word := if x=0 then 0 else actualScale
@[simp] theorem literal_zero (actualScale : Word) : literal actualScale 0=0 := by simp [literal]
@[simp] theorem literal_scale (c : Shape) (actualScale : Word) :
    literal actualScale (KernelIO.realScale c)=actualScale := by
  simp [literal,KernelIO.realScale_ne_zero]

/-- Actual shared FP32 quote, full grid, output/history channels and native
primitive observer. This entry describes source-to-native observation. -/
def io (native : ISA) (c : Shape) (actualScale : Word) (storeFlag ifcond : Bool) :
    ProgramIO Word (Term Word) :=
  KernelIO.io (wordOps native) (literal actualScale) c (KernelIO.realScale c) storeFlag ifcond

/-- Unchanged two local numerical equations and their reached operand domains.
Access safety, termination and output correctness are not assumptions here. -/
def domain (native : ISA) (dom : NumericDomain) (c : Shape) (actualScale : Word)
    (ifcond : Bool) (values : KernelIO.Values Word c) : Prop :=
  NumericalReference.BitNumerical native dom c (KernelIO.logicalInputs c values) actualScale ifcond

/-- Independent FP32 scalar recurrence, retaining every source rounding site. -/
def formula (c : Shape) (actualScale : Word) (ifcond : Bool) (values : KernelIO.Values Word c) :
    ChannelValues Word (KernelIO.outputs c)
  | ⟨0,_⟩,i=>NumericalReference.SourceFP.output c (KernelIO.logicalInputs c values) actualScale
      (KernelIO.mainHead c i.1.val) (KernelIO.mainRow c i.1.val) (KernelIO.mainColumn c i.1.val) ifcond
  | ⟨1,_⟩,i=>NumericalReference.SourceFP.history c (KernelIO.logicalInputs c values) actualScale
      (KernelIO.historyHead c i.1.val) (KernelIO.historyChunk c i.1.val)
      (KernelIO.historyRow c i.1.val) (KernelIO.historyColumn c i.1.val) ifcond

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NativeIO

import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedScratch
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

abbrev Kind := RealScratch.FloatingKind

def channel : Kind → FloatDType
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .fp32 | .fp64 => .real

def firstKernel : Kind → RegionName → RegionName → Region .int → Nat → Nat → Nat → Bool → ComputeKernel
  | .fp32 => TypedSource.stage1FP32
  | .fp16 => TypedSource.stage1FP16
  | .bf16 => TypedSource.stage1BF16
  | .fp64 => TypedSource.stage1FP64

def secondKernel : Kind → RegionName → Region .int → Region .int → Nat → Nat → ComputeKernel
  | .fp32 => TypedSource.stage2FP32
  | .fp16 => TypedSource.stage2FP16
  | .bf16 => TypedSource.stage2BF16
  | .fp64 => TypedSource.stage2FP64

/-- Carrier words represent the FP32-exact decoded value of a floating storage
cell. All requested floating scratch values originate from FP32, so this also
represents FP64 scratch losslessly for comparisons. Rounding into FP16/BF16 is
an explicit scalar operation; this is not a forced FP32 scratch allocation. -/
structure Rounding where
  fp16 : Word → Word
  bf16 : Word → Word
  fp16Padding : fp16 0xff800000=0xff800000
  bf16Padding : bf16 0xff800000=0xff800000

def Rounding.cast (r : Rounding) : Kind → Word → Word
  | .fp16 => r.fp16
  | .bf16 => r.bf16
  | .fp32 | .fp64 => id

def indexed (dtype : FloatDType) {shape : TileShape} (ax : Fin shape.length)
    (values : Values Word dtype.toTileDType shape) :
    Option (Values Word dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values Word .nat (TileShape.eraseAxis shape ax)) :=
  if positive : 0<TileShape.axisDim shape ax then
    let pick := fun out => MachineMax.best .fp32
      (fun lane => toFloat dtype (values (TileShape.insertAxisIndex shape ax out lane))) positive
    some ((fun out => values (TileShape.insertAxisIndex shape ax out (pick out))),fun out => (pick out).val)
  else none

def backend (r : Rounding) (rest : Algebra Word) : Algebra Word :=
  { Typed.backend rest with
    indexedMax := fun _ dtype => indexed dtype
    cast := fun _ source target x => match source,target with
      | .real,.fp16 => r.fp16 x
      | .real,.bf16 => r.bf16 x
      | _,_ => x }

def firstWritten (kind : Kind) (r : Rounding) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := Typed.firstIndex s X size block base positive
  (s.write V (s.pids 0+base)
    (.mk (channel kind).toTileDType
      (ofFloat (channel kind) (r.cast kind ((s.mem X selected).read .real))))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

theorem stage1_run (kind : Kind) (r : Rounding) (rest : Algebra Word) (s : State Word)
    (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend r rest) (firstKernel kind X V I size block base index64) s=some t ∧
      t.mem=(firstWritten kind r s X V I size block base positive).mem := by
  cases kind <;> cases index64 <;>
    simp [firstKernel,TypedSource.stage1FP32,TypedSource.stage1FP16,TypedSource.stage1BF16,TypedSource.stage1FP64,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,Typed.backend,State.setReg,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,Typed.globalLoaded,Typed.firstIndex,firstWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,channel,Rounding.cast]
  all_goals rfl

def secondIndex (kind : Kind) (s : State Word) (V : RegionName)
    (size block : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (fun lane : Fin block =>
    if lane.val<size then toFloat (channel kind) ((s.mem V lane.val).read (channel kind).toTileDType)
      else 0xff800000)

def secondWritten (kind : Kind) (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Word :=
  s.write (Region.cast O) 0
    (.mk .int ((s.mem (Region.cast I) (secondIndex kind s V size block positive)).read .int))

theorem stage2_run (kind : Kind) (r : Rounding) (rest : Algebra Word) (s : State Word)
    (V : RegionName) (I O : Region .int) (size block : Nat) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend r rest) (secondKernel kind V I O size block) s=some t ∧
      t.mem=(secondWritten kind s V I O size block positive).mem := by
  have pad16 : r.fp16 (4286578688#32) = (4286578688#32) := r.fp16Padding
  have padbf : r.bf16 (4286578688#32) = (4286578688#32) := r.bf16Padding
  cases kind <;>
    simp [secondKernel,TypedSource.stage2FP32,TypedSource.stage2FP16,TypedSource.stage2BF16,TypedSource.stage2FP64,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,Typed.backend,State.setReg,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
      TypedReduction.selected,flag,clean,secondIndex,secondWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,channel,r.fp16Padding,r.bf16Padding]
  all_goals try simp only [pad16,padbf]
  all_goals rfl

#axiomsClean stage1_run
#axiomsClean stage2_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedScratch

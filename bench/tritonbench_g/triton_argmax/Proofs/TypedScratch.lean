import bench.tritonbench_g.triton_argmax.Proofs.TypedGlobal

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

def secondIndex (kind : Kind) (s : State Word) (V : RegionName)
    (size block : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (fun lane : Fin block =>
    if lane.val<size then toFloat (channel kind) ((s.mem V lane.val).read (channel kind).toTileDType)
      else 0xff800000)

def secondWritten (kind : Kind) (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Word :=
  s.write (Region.cast O) 0
    (.mk .int ((s.mem (Region.cast I) (secondIndex kind s V size block positive)).read .int))

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedScratch

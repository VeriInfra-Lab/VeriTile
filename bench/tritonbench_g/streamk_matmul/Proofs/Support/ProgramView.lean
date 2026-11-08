import bench.tritonbench_g.streamk_matmul.IO
import VeriTile.Spec

/-! Concrete native source implementations under the original fixed public
interface. The six supplied bodies are all decoded. The relation retains
completed runs, all read/mutation events, the full final state, observations
and exterior frames; no result or runner is supplied as implementation data.
-/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ProgramView
open Whole001 Source VeriTile Triton Concurrent
noncomputable section

/-- The same whole-source decoder as ClosedSource.sourceTape, with the actual
source set explicit so a second implementation can supply its six bodies. -/
def sourceTape (sources : Reader.Sources) (p : Parameters) : Option (List Event) := do
  let program ← Reader.compile sources
  if p.h.m=0 ∨ p.h.n=0 then some [] else
    let events ← Reader.events p program ClosedSource.symbolic
    let reads ← ClosedSource.firstReads p program
    some (reads ++ events.first.map (ClosedSource.lowerEvent p) ++
      (List.range (Plan.select p.h).streamTiles).flatMap (ClosedSource.reduceReads p program.reduce) ++
      events.reduce.map (ClosedSource.lowerEvent p) ++ ClosedSource.fullReads p program.full ++
      events.full.map (ClosedSource.lowerEvent p))

structure Implementation (reference : Native.Reference) (p : Parameters) where
  sources : Reader.Sources

def Implementation.io (x : Implementation reference p) :
    EventProgramIO Reader.Sources Native.Word EventIO.NativeObservation Native.Word :=
  { EventIO.nativeIO p with source:=x.sources,decoded:=(sourceTape x.sources p).map Program.tape }

def Equivalent (left right : Implementation reference p) : Prop :=
  ∀ input before, (EventIO.nativeIO p).InputBinding input before →
    ∃ a z events final outputs,
      left.io.decoded = some a ∧ right.io.decoded = some z ∧
      a.Runs (EventIO.nativeArithmetic reference) before events final ∧
      z.Runs (EventIO.nativeArithmetic reference) before events final ∧
      (∀ bounds, left.io.InBounds bounds → TapeSafe bounds events) ∧
      (∀ bounds, right.io.InBounds bounds → TapeSafe bounds events) ∧
      left.io.OutputBinding outputs final ∧ right.io.OutputBinding outputs final ∧
      left.io.Frame before final ∧ right.io.Frame before final

instance : Spec.ProgramSyntax (Implementation reference p) where
  Statement := Implementation reference p
  Signature := Unit
  signature := fun _ => ()
  body := fun x => [x]
  structural := some Equivalent
  sameContext := Eq

end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ProgramView

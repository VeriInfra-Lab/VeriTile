import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.ClosedSource

/-! Read-only events of the actual decoded masks/pointers, including source
loads in lanes whose eventual C store/atomic is masked. Bounds are not tested
here: a bad active source pointer must fail the shared safety conclusion. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section

def inputReads (tile : Reader.Tile) (env : Reader.Env) (lane : Reader.Lanes) (iter : Nat) :
    List Concurrent.Event :=
  let e:=Reader.loopEnv tile env lane iter
  let pointers:=Reader.loopPointers tile env lane iter
  (if tile.loop.maskA.eval e lane then
    [.readOnly (.input 0 (tile.loop.left.index e pointers lane))] else []) ++
  (if tile.loop.maskB.eval e lane then
    [.readOnly (.input 1 (tile.loop.right.index e pointers lane))] else [])

def tileReads (p : Parameters) (tile : Reader.Tile) (env : Reader.Env) : List Concurrent.Event :=
  (Source.lanes p).flatMap fun lane=>
    let l:=Reader.mlane (env "pid") lane 0
    let e:=Reader.tileEnv tile env l
    let lo:=tile.loop.lo.eval e l
    let hi:=tile.loop.hi.eval e l
    (List.range' lo (hi-lo)).flatMap fun iter=>
      (List.range p.h.bk).flatMap fun k=>
        inputReads tile env (Reader.mlane (env "pid") lane k) iter

def firstReads (p : Parameters) (program : Reader.Program) : Option (List Concurrent.Event) := do
  if 0<(Plan.select p.h).streamTiles then
    let nested ← (List.finRange p.h.programs).mapM fun pid=>do
      let l : Reader.Lanes:=⟨pid.val,0,0,0⟩
      let e:=Reader.put (Reader.parameters p) "pid" (program.first.pid.eval (Reader.parameters p) l)
      let chunks ← Reader.chunks program.first e (program.first.stop.eval e l) (program.first.start.eval e l)
      some (chunks.flatMap fun c=>tileReads p program.mac.core (Reader.macEnv p program.mac pid.val c))
    some nested.flatten
  else some []

def reduceReads (p : Parameters) (program : Reader.Reduce) (tile : Nat) : List Concurrent.Event :=
  let lo:=program.lo.eval (Reader.parameters p) ⟨tile,0,0,0⟩
  let hi:=program.hi.eval (Reader.parameters p) ⟨tile,0,0,0⟩
  (List.range' lo (hi-lo)).flatMap fun pid=>
    (Source.lanes p).flatMap fun lane=>
      let l:=Reader.mlane tile lane 0
      let e:=Reader.reduceEnv p program tile pid lane
      if program.condition.eval e l then
        [.readOnly (.load 1 (program.load.index e Reader.zeroPointers l))] else []

def fullReads (p : Parameters) (program : Reader.Full) : List Concurrent.Event :=
  (List.range (Plan.select p.h).blockingTiles).flatMap fun pid=>
    tileReads p program.core (Reader.fullEnv p program pid)

/-- All six quoted bodies are decoded first. Read observations preserve every
active source access; mutation events use the same closed symbolic decoder as
mutationTape. Read observations leave the executor state unchanged. -/
def sourceTape (p : Parameters) : Option (List Concurrent.Event) := do
  let program ← Reader.compile kernels
  if p.h.m=0 ∨ p.h.n=0 then some [] else
    let events ← Reader.events p program symbolic
    let reads ← firstReads p program
    some (reads ++ events.first.map (lowerEvent p) ++
      (List.range (Plan.select p.h).streamTiles).flatMap (reduceReads p program.reduce) ++
      events.reduce.map (lowerEvent p) ++ fullReads p program.full ++ events.full.map (lowerEvent p))

#specNonCircular sourceTape avoiding [Source.call,Source.firstWave,Source.reduceWave,Source.fullWave,Memory.run,Memory.step]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource

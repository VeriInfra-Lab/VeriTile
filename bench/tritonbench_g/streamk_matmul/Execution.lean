import bench.tritonbench_g.streamk_matmul.Kernels

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxRecDepth 20000

def parameters (p : Source.Parameters) : Env
  | "M" => p.h.m | "N" => p.h.n | "K" => p.h.k
  | "BLOCK_M" => p.h.bm | "BLOCK_N" => p.h.bn | "BLOCK_K" => p.h.bk
  | "GROUP_M" => 8
  | "stride_am" => p.strideAM | "stride_ak" => p.strideAK
  | "stride_bk" => p.strideBK | "stride_bn" => p.strideBN
  | "stride_cm" => max p.h.n 1 | "stride_cn" => 1
  | "iters_per_tile" => (Plan.select p.h).iterations
  | "total_tiles_streamk" => (Plan.select p.h).streamTiles
  | "total_full_tiles_streamk" => (Plan.select p.h).fullIterations
  | "total_partial_tiles_streamk" => (Plan.select p.h).partialIterations
  | "total_programs_streamk" => p.h.programs
  | _ => 0

def mapped (g : Reader.Geometry) (env : Env) (lane : Lanes) : Env :=
  let ctx :=  put env "tile_id" (g.tile.eval env lane)
  let m := if g.condition.eval ctx lane then g.swizzle else g.linear
  let ctx :=  bindings m.prepare ctx lane
  put (put ctx "pid_m" (m.m.eval ctx lane)) "pid_n" (m.n.eval ctx lane)

def tileEnv (t : Tile) (env : Env) (lane : Lanes) : Env :=
  let e := mapped t.geometry env lane
  let e := put e "rm" (t.rm.eval e lane)
  let e := put e "rn" (t.rn.eval e lane)
  let e := put e "rk" (t.rk.eval e lane)
  let e := put e "step_a" (t.stepA.eval e lane)
  put e "step_b" (t.stepB.eval e lane)

/-- Element offsets retain the source pointer expression. Base-region identity
is checked by the reader schema; byte bases are supplied by the memory binding. -/
def Pointer.index (env pointerEnv : Env) (lane : Lanes) : Pointer → Nat
  | .variable n => pointerEnv n
  | .offset p e => p.index env pointerEnv lane+e.eval env lane

def Pointer.root : Pointer → String
  | .variable n => n
  | .offset p _ => p.root

def zeroPointers : Env := fun _ => 0

def affineAdvance (pointer : Pointer) (root : String) : Bool :=
  match pointer with
  | .offset (.variable name) _ => name == root
  | _ => Bool.false

/-- Source checking limits this interpreter to two immutable input regions,
FP32 accumulation and the explicit output/scratch regions. It checks instruction
forms, never a computed output or a successful run. -/
def typed (s : Program) : Bool :=
  let tile := fun (t : Tile) =>
    t.pointerA.root == "A" && t.pointerB.root == "B" &&
    t.loop.left == .variable "A" && t.loop.right == .variable "B" &&
    affineAdvance t.loop.advanceA "A" && affineAdvance t.loop.advanceB "B" &&
    t.zeroM == .variable "BLOCK_M" && t.zeroN == .variable "BLOCK_N"
  tile s.mac.core && tile s.full.core &&
  s.mac.output.root == "C" && s.mac.scratch.root == "partials" &&
  s.full.output.root == "C" && s.reduce.load.root == "partials" && s.reduce.output.root == "C"

def chunkEnv (p : Source.Parameters) (program : Nat) (c : Protocol.Chunk) : Env :=
  put (put (put (put (parameters p) "pid" program) "program" program)
    "start_iter" c.lo) "end_iter" c.hi

def macEnv (p : Source.Parameters) (s : Mac) (program : Nat) (c : Protocol.Chunk) : Env :=
  let e := chunkEnv p program c
  put e "tile_id" (s.tile.eval e ⟨program,0,0,0⟩)

def fullEnv (p : Source.Parameters) (s : Full) (pid : Nat) : Env :=
  put (put (parameters p) "pid" pid) "tile_id" (s.tile.eval (parameters p) ⟨pid,0,0,0⟩)

def loopEnv (t : Tile) (env : Env) (lane : Lanes) (iter : Nat) : Env :=
  let e := put (tileEnv t env lane) t.loop.index iter
  put e "kk" (t.loop.kk.eval e lane)

/-- Closed execution of the extracted affine pointer update. The iteration
count is relative to the extracted source range start, not a tile oracle. -/
def loopPointers (t : Tile) (env : Env) (lane : Lanes) (iter : Nat) : Env :=
  let e := tileEnv t env lane
  let count := iter-t.loop.lo.eval e lane
  let a := t.pointerA.index e zeroPointers lane
  let b := t.pointerB.index e zeroPointers lane
  let da := t.loop.advanceA.index e zeroPointers lane
  let db := t.loop.advanceB.index e zeroPointers lane
  put (put zeroPointers "A" (a+count*da)) "B" (b+count*db)

def inputA (p : Source.Parameters) (t : Tile) (ops : Instructions α) (a : A p → α)
    (env : Env) (lane : Lanes) (iter : Nat) : α :=
  let e := loopEnv t env lane iter
  let i := t.loop.left.index e (loopPointers t env lane iter) lane
  if t.loop.maskA.eval e lane then
    if h : i<p.h.m*p.h.k then a ⟨i,h⟩ else ops.zero
  else ops.zero

def inputB (p : Source.Parameters) (t : Tile) (ops : Instructions α) (b : B p → α)
    (env : Env) (lane : Lanes) (iter : Nat) : α :=
  let e := loopEnv t env lane iter
  let i := t.loop.right.index e (loopPointers t env lane iter) lane
  if t.loop.maskB.eval e lane then
    if h : i<p.h.k*p.h.n then b ⟨i,h⟩ else ops.zero
  else ops.zero

/-- The only arithmetic callback is one fixed K8 MMA instruction. Actual A/B
operands come from the extracted pointer expressions and masks above. -/
def dotBlock (p : Source.Parameters) (t : Tile) (ops : Instructions α)
    (a : A p → α) (b : B p → α) (env : Env) (lane : Lane p) (iter : Nat) (seed : α) : α :=
  (List.range ((p.h.bk+7)/8)).foldl (fun acc group =>
    ops.mma8
      (fun j => inputA p t ops a env ⟨env "pid",lane.1.val,lane.2.val,group*8+j.val⟩ iter)
      (fun j => inputB p t ops b env ⟨env "pid",lane.1.val,lane.2.val,group*8+j.val⟩ iter) acc) seed

def dotValue (p : Source.Parameters) (t : Tile) (ops : Instructions α)
    (a : A p → α) (b : B p → α) (env : Env) (lane : Lane p) : α :=
  let e := tileEnv t env ⟨env "pid",lane.1.val,lane.2.val,0⟩
  let lo := t.loop.lo.eval e ⟨env "pid",lane.1.val,lane.2.val,0⟩
  let hi := t.loop.hi.eval e ⟨env "pid",lane.1.val,lane.2.val,0⟩
  (List.range' lo (hi-lo)).foldl (fun acc iter => dotBlock p t ops a b env lane iter acc) ops.zero

def outputEvent (p : Source.Parameters) (index : Nat)
    (value : (A p → α) → (B p → α) → α) : List (Source.Event α p) :=
  if h : index<p.h.m*p.h.n then [.storeOutput ⟨index,h⟩ value] else []
def scratchEvent (p : Source.Parameters) (index : Nat)
    (value : (A p → α) → (B p → α) → α) : List (Source.Event α p) :=
  if h : index<p.h.programs*p.h.bm*p.h.bn then [.storeScratch ⟨index,h⟩ value] else []
def atomicEvent (p : Source.Parameters) (output scratch : Nat) : List (Source.Event α p) :=
  if hc : output<p.h.m*p.h.n then
    if hs : scratch<p.h.programs*p.h.bm*p.h.bn then [.atomic ⟨output,hc⟩ ⟨scratch,hs⟩] else []
  else []

def macEvents (p : Source.Parameters) (s : Mac) (ops : Instructions α) (program : Fin p.h.programs)
    (c : Protocol.Chunk) : List (Source.Event α p) :=
  let env := macEnv p s program.val c
  let terminal := s.terminal.eval env ⟨program.val,0,0,0⟩
  (Source.lanes p).flatMap fun lane =>
    let l : Lanes := ⟨program.val,lane.1.val,lane.2.val,0⟩
    let e := tileEnv s.core env l
    let value := fun a b => dotValue p s.core ops a b env lane
    if terminal then
      if s.mask.eval e l then outputEvent p (s.output.index e zeroPointers l) value else []
    else
      let e := put e "slots" (s.slots.eval e l)
      scratchEvent p (s.scratch.index e zeroPointers l) value

def fullEvents (p : Source.Parameters) (s : Full) (ops : Instructions α) (pid : Nat) : List (Source.Event α p) :=
  let env := fullEnv p s pid
  (Source.lanes p).flatMap fun lane =>
    let l : Lanes := ⟨pid,lane.1.val,lane.2.val,0⟩
    let e := tileEnv s.core env l
    let e := put e "rm" (s.rematM.eval e l)
    let e := put e "rn" (s.rematN.eval e l)
    if s.mask.eval e l then
      outputEvent p (s.output.index e zeroPointers l)
        (fun a b => ops.halfRoundTrip (dotValue p s.core ops a b env lane))
    else []

def reduceEnv (p : Source.Parameters) (s : Reduce) (tile program : Nat) (lane : Lane p) : Env :=
  let l : Lanes := ⟨tile,lane.1.val,lane.2.val,0⟩
  let e := put (parameters p) "tile_id" (s.tile.eval (parameters p) l)
  let e := mapped s.geometry e l
  let e := put e "rm" (s.rm.eval e l)
  let e := put e "rn" (s.rn.eval e l)
  let e := put e "offsets" (s.offsets.eval e l)
  let e := put e "mask" (if s.mask.eval e l then 1 else 0)
  let e := put e s.index program
  let e := put e "pid" (s.pid.eval e l)
  let e := put e "begin" (s.begin.eval e l)
  let e := put e "end" (s.finish.eval e l)
  put e "slots" (s.slots.eval e l)

def reduceEvents (p : Source.Parameters) (s : Reduce) (tile : Nat) : List (Source.Event α p) :=
  let lo := s.lo.eval (parameters p) ⟨tile,0,0,0⟩
  let hi := s.hi.eval (parameters p) ⟨tile,0,0,0⟩
  (List.range' lo (hi-lo)).flatMap fun program =>
    (Source.lanes p).flatMap fun lane =>
      let l : Lanes := ⟨tile,lane.1.val,lane.2.val,0⟩
      let e := reduceEnv p s tile program lane
      if s.condition.eval e l ∧ s.atomicMask.eval e l then
        atomicEvent p (s.output.index e zeroPointers l) (s.load.index e zeroPointers l)
      else []

/-- Failure records an unsupported nonprogressing source loop. For the selected
source strict progress is proved from positive iterations-per-tile. -/
def chunks (s : First) (env : Env) (stop current : Nat) : Option (List Protocol.Chunk) :=
  let e := put (put env "last_iter" stop) "start_iter" current
  let lane : Lanes := ⟨env "pid",0,0,0⟩
  if s.condition.eval e lane then
    let finish := s.finish.eval e lane
    let advance := s.advance.eval (put e "end_iter" finish) lane
    if h : current<advance ∧ advance≤stop then do
      let rest ← chunks s env stop advance
      some (⟨current,finish⟩::rest)
    else none
  else some []
termination_by stop-current
decreasing_by
  exact Nat.sub_lt_sub_left (lt_of_lt_of_le h.1 h.2) h.1

def firstEvents (p : Source.Parameters) (s : Program) (ops : Instructions α) : Option (List (Source.Event α p)) := do
  if 0<(Plan.select p.h).streamTiles then
    let nested ← (List.finRange p.h.programs).mapM fun program => do
      let l : Lanes := ⟨program.val,0,0,0⟩
      let e := put (parameters p) "pid" (s.first.pid.eval (parameters p) l)
      let cs ← chunks s.first e (s.first.stop.eval e l) (s.first.start.eval e l)
      some (cs.flatMap (macEvents p s.mac ops program))
    some nested.flatten
  else some []

structure Events (α : Type) (p : Source.Parameters) where
  first : List (Source.Event α p)
  reduce : List (Source.Event α p)
  «full» : List (Source.Event α p)

def events (p : Source.Parameters) (s : Program) (ops : Instructions α) : Option (Events α p) := do
  if typed s then
    return ⟨← firstEvents p s ops,
      (List.range (Plan.select p.h).streamTiles).flatMap (reduceEvents p s.reduce),
      (List.range (Plan.select p.h).blockingTiles).flatMap (fullEvents p s.full ops)⟩
  else none

def execute (source : Sources) (p : Source.Parameters) (ops : Instructions α)
    (initial : Source.State α p exterior) : Option (Source.State α p exterior) := do
  let program ← compile source
  if p.h.m=0 ∨ p.h.n=0 then some initial else
    let e ← events p program ops
    some (Memory.run ops.atomicAdd e.full (Memory.run ops.atomicAdd e.reduce
      (Memory.run ops.atomicAdd e.first initial)))

end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader

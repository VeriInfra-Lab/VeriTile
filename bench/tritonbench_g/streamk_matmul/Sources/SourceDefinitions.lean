import bench.tritonbench_g.streamk_matmul.Sources.Foundation
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Source
open Protocol
set_option maxHeartbeats 8000000
noncomputable section
local instance (q r I tile program : Nat) : Decidable (readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

structure Parameters where
  h : HostInput
  strideAM : Nat
  strideAK : Nat
  strideBK : Nat
  strideBN : Nat
  blockM : 0 < h.bm
  blockN : 0 < h.bn
  blockK : 0 < h.bk
  nativeK : h.bk%8=0
  inputA : Geometry.Contiguous h.m h.k strideAM strideAK
  inputB : Geometry.Contiguous h.k h.n strideBK strideBN

/-- Each arithmetic interface is one fixed instruction or scalar conversion.
No tile-dot, chunk, reduction, kernel, or returned-output callback. -/
structure Instructions (α : Type) where
  zero : α
  mma8 : (Fin 8 → α) → (Fin 8 → α) → α → α
  halfRoundTrip : α → α
  atomicAdd : α → α → α

abbrev A (p : Parameters) := Fin (p.h.m*p.h.k)
abbrev B (p : Parameters) := Fin (p.h.k*p.h.n)
abbrev C (p : Parameters) := Fin (p.h.m*p.h.n)
abbrev S (p : Parameters) := Fin (p.h.programs*p.h.bm*p.h.bn)
abbrev Lane (p : Parameters) := Fin p.h.bm × Fin p.h.bn
abbrev Event (α : Type) (p : Parameters) :=
  Memory.Event α (p.h.m*p.h.k) (p.h.k*p.h.n) (p.h.m*p.h.n) (p.h.programs*p.h.bm*p.h.bn)
abbrev State (α : Type) (p : Parameters) (exterior : Nat) :=
  Memory.State α (p.h.m*p.h.k) (p.h.k*p.h.n) (p.h.m*p.h.n) (p.h.programs*p.h.bm*p.h.bn) exterior

def loadA (p : Parameters) (zero : α) (a : A p → α) (row k : Nat) : α :=
  if h:row < p.h.m ∧ k < p.h.k then
    a ⟨row*p.strideAM+k*p.strideAK,Geometry.input_mask_safe _ _ _ _ _ _ p.inputA h.1 h.2⟩
  else zero

def loadB (p : Parameters) (zero : α) (b : B p → α) (k col : Nat) : α :=
  if h:k < p.h.k ∧ col < p.h.n then
    b ⟨k*p.strideBK+col*p.strideBN,Geometry.input_mask_safe _ _ _ _ _ _ p.inputB h.1 h.2⟩
  else zero

def row (p : Parameters) (tile : Nat) (lane : Lane p) : Nat :=
  (let first := tile/(8*ceilDiv p.h.n p.h.bn)*8
   first+tile%min (ceilDiv p.h.m p.h.bm-first) 8)*p.h.bm+lane.1.val

def col (p : Parameters) (tile : Nat) (lane : Lane p) : Nat :=
  (let first := tile/(8*ceilDiv p.h.n p.h.bn)*8
   (tile%(8*ceilDiv p.h.n p.h.bn))/min (ceilDiv p.h.m p.h.bm-first) 8)*p.h.bn+lane.2.val

def lanes (p : Parameters) : List (Lane p) :=
  (List.finRange p.h.bm).flatMap fun r=>(List.finRange p.h.bn).map fun c=>(r,c)

def scratch (p : Parameters) (program : Fin p.h.programs) (lane : Lane p) : S p :=
  ⟨program.val*p.h.bm*p.h.bn+lane.1.val*p.h.bn+lane.2.val,
    Geometry.scratch_bounds _ _ _ _ _ _ program.isLt lane.1.isLt lane.2.isLt⟩

def active (p : Parameters) (tile : Nat) (lane : Lane p) : Prop :=
  row p tile lane < p.h.m ∧ col p tile lane < p.h.n
instance (p : Parameters) (tile : Nat) (lane : Lane p) : Decidable (active p tile lane) :=
  inferInstanceAs (Decidable (_ ∧ _))

def output (p : Parameters) (tile : Nat) (lane : Lane p) (h : active p tile lane) : C p :=
  ⟨row p tile lane*p.h.n+col p tile lane,Geometry.packed_bound _ _ _ _ h.1 h.2⟩

/-- SM80 compiler-selected accumulator-fed TF32 graph, with exactly eight
logical K operands per MMA instruction. Loop seed is the current accumulator. -/
def block (p : Parameters) (ops : Instructions α) (a : A p → α) (b : B p → α)
    (tile iter : Nat) (lane : Lane p) (seed : α) : α :=
  (List.range ((p.h.bk+7)/8)).foldl (fun acc group =>
    ops.mma8
      (fun j=>loadA p ops.zero a (row p tile lane)
        ((iter%(Plan.select p.h).iterations)*p.h.bk+group*8+j.val))
      (fun j=>loadB p ops.zero b
        ((iter%(Plan.select p.h).iterations)*p.h.bk+group*8+j.val) (col p tile lane)) acc) seed

def chunkValue (p : Parameters) (ops : Instructions α) (a : A p → α) (b : B p → α)
    (tile lo hi : Nat) (lane : Lane p) : α :=
  (List.range' lo (hi-lo)).foldl (fun acc iter=>block p ops a b tile iter lane acc) ops.zero

def storeTile (p : Parameters) (value : Lane p → (A p → α) → (B p → α) → α)
    (tile : Nat) : List (Event α p) :=
  (lanes p).flatMap fun lane=>
    if h:active p tile lane then [.storeOutput (output p tile lane h) (value lane)] else []

/-- Lowered source mac_loop: terminal chunk writes masked C; its alternative
writes exactly one full scratch tile belonging to the current program. -/
def macLoop (p : Parameters) (ops : Instructions α) (program : Fin p.h.programs)
    (c : Chunk) : List (Event α p) :=
  let I:=(Plan.select p.h).iterations
  let tile:=c.tile I
  let value:=fun lane a b=>chunkValue p ops a b tile c.lo c.hi lane
  if c.hi%I=0 then storeTile p value tile
  else (lanes p).map fun lane=>.storeScratch (scratch p program lane) (value lane)

def firstWave (p : Parameters) (ops : Instructions α) : List (Event α p) :=
  if hs:0 < (Plan.select p.h).streamTiles then
    let hI:=Plan.iterations_positive p.h (Plan.selected_active p.h hs).2 p.blockK
    (List.finRange p.h.programs).flatMap fun program=>
      (programChunks (Plan.select p.h).streamTiles (Plan.select p.h).iterations p.h.programs program.val hI).flatMap
        (macLoop p ops program)
  else []

/-- Source reduction loops tile then ascending program ID. The only reads of
scratch are selected by the proved integer producer predicate. -/
def programPairs (p : Parameters) (tile : Nat) (program : Fin p.h.programs) : List (C p × S p) :=
  if readsScratch (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations
      (Plan.select p.h).iterations tile program.val then
    (lanes p).flatMap fun lane=>
      if h:active p tile lane then [(output p tile lane h,scratch p program lane)] else []
  else []

def tilePairs (p : Parameters) (tile : Nat) : List (C p × S p) :=
  (List.finRange p.h.programs).flatMap (programPairs p tile)

def reducePairs (p : Parameters) : List (C p × S p) :=
  (List.range (Plan.select p.h).streamTiles).flatMap (tilePairs p)

def reduceWave (p : Parameters) : List (Event α p) :=
  (reducePairs p).map fun pair=>.atomic pair.1 pair.2

def fullWave (p : Parameters) (ops : Instructions α) : List (Event α p) :=
  (List.range (Plan.select p.h).blockingTiles).flatMap fun pid=>
    let tile:=pid+(Plan.select p.h).streamTiles
    storeTile p (fun lane a b=>ops.halfRoundTrip
      (chunkValue p ops a b tile 0 (Plan.select p.h).iterations lane)) tile

/-- Actual ordered three-launch memory path; torch.empty output and scratch
remain unconstrained initial finite payloads. All source writes are explicit. -/
def call (p : Parameters) (ops : Instructions α) (initial : State α p exterior) : State α p exterior :=
  if p.h.m=0 ∨ p.h.n=0 then initial else
    Memory.run ops.atomicAdd (fullWave p ops)
      (Memory.run ops.atomicAdd (reduceWave p)
        (Memory.run ops.atomicAdd (firstWave p ops) initial))

end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001

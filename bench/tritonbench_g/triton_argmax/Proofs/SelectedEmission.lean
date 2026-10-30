import bench.tritonbench_g.triton_argmax.Proofs.Emission

/-! The compiler returns raw scalarized instruction syntax. A separate local
extraction certificate follows the actual value/index def-use graph. Shuffle
sources are computed from the emitting lane and shuffle mask; shared-memory
sources require the preceding writer and absence of an intervening write.
No tree or reduction result is part of the compiler API's return contract. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.SelectedEmission
open Reduction Source Emission

structure Ref where
  lane : Nat
  register : Nat
  deriving DecidableEq,Repr

/-- Structural lane predicates are evaluated from lane IDs and literal
constants, separately from all input floating values. -/
inductive LaneExpr where
  | lane
  | literal : Nat → LaneExpr
  | bitAnd : LaneExpr → LaneExpr → LaneExpr
  | shiftRight : LaneExpr → Nat → LaneExpr
  deriving DecidableEq,Repr

def LaneExpr.eval (lane : Nat) : LaneExpr → Nat
  | .lane => lane
  | .literal value => value
  | .bitAnd a b => (a.eval lane) &&& (b.eval lane)
  | .shiftRight a shift => (a.eval lane) / 2^shift

inductive LaneGuard where
  | always
  | equal : LaneExpr → LaneExpr → LaneGuard
  | less : LaneExpr → LaneExpr → LaneGuard
  | both : LaneGuard → LaneGuard → LaneGuard
  | either : LaneGuard → LaneGuard → LaneGuard
  deriving DecidableEq,Repr

def LaneGuard.eval (lane : Nat) : LaneGuard → Bool
  | .always => Bool.true
  | .equal a b => decide (a.eval lane=b.eval lane)
  | .less a b => decide (a.eval lane < b.eval lane)
  | .both a b => a.eval lane && b.eval lane
  | .either a b => a.eval lane || b.eval lane

inductive Opcode (n : Nat) where
  | input : Fin n → Opcode n
  | laneIndex : Fin n → Opcode n
  | greater : Ref → Ref → Opcode n
  | equal : Ref → Ref → Opcode n
  | indexLess : Ref → Ref → Opcode n
  | boolAnd : Ref → Ref → Opcode n
  | boolOr : Ref → Ref → Opcode n
  | selectValue : Ref → Ref → Ref → Opcode n
  | selectIndex : Ref → Ref → Ref → Opcode n
  | copy : Ref → Opcode n
  | shuffleXor : Nat → Nat → Nat → Nat → Opcode n
  | sharedStore : Nat → Nat → Ref → LaneGuard → Opcode n
  | sharedLoad : Nat → Nat → LaneGuard → Opcode n
  | barrier : Nat → Opcode n
  | splitLow : Ref → Opcode n
  | splitHigh : Ref → Opcode n
  | zeroExtend32 : Ref → Opcode n
  | shiftLeft32 : Ref → Opcode n
  | bitOr64 : Ref → Ref → Opcode n
  | other : String → Opcode n
  deriving DecidableEq,Repr

structure Graph (n : Nat) where
  instruction : Ref → Option (Opcode n)
  blockThreads : Nat
  /-- Reached completion/visibility order after dependencies. A barrier node
  records its collective release, not a lane arrival at that barrier. -/
  time : Ref → Nat
  uniqueTime : ∀ a b,instruction a≠none → instruction b≠none → time a=time b → a=b

def shuffled (destination : Ref) (sourceRegister mask : Nat) : Ref :=
  ⟨destination.lane/32*32+Nat.xor (destination.lane%32) mask,sourceRegister⟩

def overlaps (a widthA b widthB : Nat) : Prop := a < b+widthB ∧ b < a+widthA

instance (a widthA b widthB : Nat) : Decidable (overlaps a widthA b widthB) := by
  unfold overlaps
  infer_instance

/-- Local shared-memory load-from-store edge. The certificate names the actual
store, address and source SSA value and excludes every intervening store to
that address. These are instruction/schedule facts, not numerical premises. -/
structure SharedEdge {n : Nat} (graph : Graph n) (destination source : Ref) where
  writer : Ref
  slot : Nat
  width : Nat
  widthPositive : 0 < width
  storeGuard : LaneGuard
  loadGuard : LaneGuard
  load : graph.instruction destination=some (.sharedLoad slot width loadGuard)
  store : graph.instruction writer=some (.sharedStore slot width source storeGuard)
  sourceLane : source.lane=writer.lane
  sourceAvailable : graph.instruction source≠none
  sourceEarlier : graph.time source < graph.time writer
  storeActive : storeGuard.eval writer.lane=Bool.true
  loadActive : loadGuard.eval destination.lane=Bool.true
  writerLane : writer.lane < graph.blockThreads
  readerLane : destination.lane < graph.blockThreads
  barrierRegister : Nat
  barrierId : Nat
  synchronized : ∀ lane,lane < graph.blockThreads →
    graph.instruction ⟨lane,barrierRegister⟩=some (.barrier barrierId)
  writerBefore : ∀ lane,lane < graph.blockThreads → graph.time writer < graph.time ⟨lane,barrierRegister⟩
  readerAfter : ∀ lane,lane < graph.blockThreads → graph.time ⟨lane,barrierRegister⟩ < graph.time destination
  lastWriter : ∀ other otherSlot otherWidth payload guard,
    graph.instruction other=some (.sharedStore otherSlot otherWidth payload guard) →
    guard.eval other.lane=Bool.true → overlaps slot width otherSlot otherWidth →
    graph.time other < graph.time destination → graph.time other ≤ graph.time writer

/-- This checked rule is for the emitted full-warp clamp (31). Other clamp
forms require their own local rule; they are not silently interpreted as 31.
Both members and every named participating lane must reach the same shuffle. -/
structure ShuffleEdge {n : Nat} (graph : Graph n) (destination source : Ref) where
  sourceRegister : Nat
  mask : Nat
  members : Nat
  operation : graph.instruction destination=some (.shuffleXor sourceRegister mask members 31)
  maskBound : mask < 32
  selected : source=shuffled destination sourceRegister mask
  destinationMember : Nat.testBit members (destination.lane%32)=Bool.true
  sourceMember : Nat.testBit members (source.lane%32)=Bool.true
  sourceAvailable : graph.instruction source≠none
  sourceEarlier : graph.time source < graph.time destination
  participants : ∀ lane,lane < 32 → Nat.testBit members lane=Bool.true →
    graph.instruction ⟨destination.lane/32*32+lane,destination.register⟩=
      some (.shuffleXor sourceRegister mask members 31)

/-- Exact 64-bit shuffle macro: the two b32 shuffles consume the low and high
halves of one actual source word. The destination is the emitted u64 widen,
shift-by-32 and OR chain. All references and dependency ordering are syntax. -/
structure WideShuffle {n : Nat} (graph : Graph n) (destination source : Ref) where
  lowSource : Ref
  highSource : Ref
  lowShuffle : Ref
  highShuffle : Ref
  lowWide : Ref
  highWide : Ref
  shifted : Ref
  lowSplit : graph.instruction lowSource=some (.splitLow source)
  highSplit : graph.instruction highSource=some (.splitHigh source)
  lowSourceLane : lowSource.lane=source.lane
  highSourceLane : highSource.lane=source.lane
  lowEdge : ShuffleEdge graph lowShuffle lowSource
  highEdge : ShuffleEdge graph highShuffle highSource
  sameMask : lowEdge.mask=highEdge.mask
  sameMembers : lowEdge.members=highEdge.members
  lowLane : lowShuffle.lane=destination.lane
  highLane : highShuffle.lane=destination.lane
  lowWideLane : lowWide.lane=destination.lane
  highWideLane : highWide.lane=destination.lane
  shiftedLane : shifted.lane=destination.lane
  lowExtend : graph.instruction lowWide=some (.zeroExtend32 lowShuffle)
  highExtend : graph.instruction highWide=some (.zeroExtend32 highShuffle)
  shift : graph.instruction shifted=some (.shiftLeft32 highWide)
  join : graph.instruction destination=some (.bitOr64 lowWide shifted)
  sourceBeforeLow : graph.time source<graph.time lowSource
  sourceBeforeHigh : graph.time source<graph.time highSource
  lowBeforeWide : graph.time lowShuffle<graph.time lowWide
  highBeforeWide : graph.time highShuffle<graph.time highWide
  wideBeforeShift : graph.time highWide<graph.time shifted
  lowBeforeJoin : graph.time lowWide<graph.time destination
  highBeforeJoin : graph.time shifted<graph.time destination

inductive Transport {n : Nat} (graph : Graph n) : Ref → Ref → Prop where
  | copy {destination source} : graph.instruction destination=some (.copy source) →
      graph.time source < graph.time destination → Transport graph destination source
  | shuffle {destination source} : ShuffleEdge graph destination source → Transport graph destination source
  | shared {destination source} : SharedEdge graph destination source → Transport graph destination source
  | wide {destination source} : WideShuffle graph destination source → Transport graph destination source

/-- Alias closure uses only the local copy/shuffle/shared transport rules.
It accommodates bitcast aliases around a select without changing the raw
instruction's operands or treating an arbitrary value equality as syntax. -/
inductive Alias {n : Nat} (graph : Graph n) : Ref → Ref → Prop where
  | refl (r : Ref) : Alias graph r r
  | transport {a b} : Transport graph a b → Alias graph a b
  | symm {a b} : Alias graph a b → Alias graph b a
  | trans {a b c} : Alias graph a b → Alias graph b c → Alias graph a c

/-- The exact scalar def-use pattern of the argmax combiner. Both selects must
consume this same predicate and the same ordered operand pairs. -/
structure PredicateEdges {n : Nat} (graph : Graph n) (p va vb ia ib : Ref) where
  g : Ref
  e : Ref
  l : Ref
  a : Ref
  greater : graph.instruction g=some (.greater va vb)
  equal : graph.instruction e=some (.equal va vb)
  less : graph.instruction l=some (.indexLess ia ib)
  andEdge : graph.instruction a=some (.boolAnd e l) ∨ graph.instruction a=some (.boolAnd l e)
  orEdge : graph.instruction p=some (.boolOr g a) ∨ graph.instruction p=some (.boolOr a g)

inductive PairDerived {n : Nat} (graph : Graph n) : Ref → Ref → Tree n → Prop where
  | leaf {v i lane} : graph.instruction v=some (.input lane) → graph.instruction i=some (.laneIndex lane) →
      PairDerived graph v i (.leaf lane)
  | transport {v i sourceV sourceI tree} : Transport graph v sourceV → Transport graph i sourceI →
      PairDerived graph sourceV sourceI tree → PairDerived graph v i tree
  | valueTransport {v i sourceV tree} : Transport graph v sourceV →
      PairDerived graph sourceV i tree → PairDerived graph v i tree
  | indexTransport {v i sourceI tree} : Transport graph i sourceI →
      PairDerived graph v sourceI tree → PairDerived graph v i tree
  | choose {v i p va vb ia ib a b} :
      PairDerived graph va ia a → PairDerived graph vb ib b → PredicateEdges graph p va vb ia ib →
      graph.instruction v=some (.selectValue p va vb) → graph.instruction i=some (.selectIndex p ia ib) →
      PairDerived graph v i (.fork a b)
  | chooseAlias {v i p va vb ia ib ga gb ja jb a b} :
      PairDerived graph va ia a → PairDerived graph vb ib b → PredicateEdges graph p ga gb ja jb →
      Alias graph ga va → Alias graph gb vb → Alias graph ja ia → Alias graph jb ib →
      graph.instruction v=some (.selectValue p va vb) → graph.instruction i=some (.selectIndex p ia ib) →
      PairDerived graph v i (.fork a b)

structure RawArtifact (n : Nat) where
  bytesHash : String
  graph : Graph n
  rootValue : Ref
  rootIndex : Ref

structure CompilerAPI where
  compile : (key : Key) → Except String (RawArtifact key.block)

structure Produced (compiler : CompilerAPI) (key : Key) where
  raw : RawArtifact key.block
  selected : compiler.compile key=Except.ok raw

/-- Independent checked syntax witness over the exact selected artifact.
The compiler primitive does not supply this witness. -/
structure Extraction {n : Nat} (raw : RawArtifact n) where
  tree : Tree n
  derivation : PairDerived raw.graph raw.rootValue raw.rootIndex tree
  covers : tree.Complete

def Extraction.artifact {n : Nat} {raw : RawArtifact n} (extraction : Extraction raw) : Emission.Artifact n :=
  ⟨extraction.tree,extraction.covers⟩

end VeriTile.Bench.Repairs.TritonArgmax.Revision004.SelectedEmission

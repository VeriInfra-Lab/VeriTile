import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPReference

/-! Independent array formulation of the established native recurrence.
The arrays retain every typed read, including aliases between floating roles
and integer metadata. A real-tagged store resets aliased integer readers to
their existing default zero. No Structural state or source executor is used.
-/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeFormula
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

structure Arrays (α : Type) where
  query : Nat → α
  key : Nat → α
  value : Nat → α
  start : Nat → Int
  sequence : Nat → Int
  prompt : Nat → Int

/-- Launch metadata only. A raw single invocation may retain its existing
grid dimensions; an explicit grid invocation supplies them. -/
structure Launch where
  pids : Nat → Nat
  numPids : Option (Nat → Nat) := none

def operands (p : Layout) (job : Launch) (a : Arrays α) : FPReference.Inputs α where
  query := a.query
  key := a.key
  value := a.value
  batch := job.pids 1/p.heads
  head := job.pids 1%p.heads
  kvHead := (job.pids 1%p.heads)/p.kvGroup
  block := job.pids 0
  start := a.start (job.pids 1/p.heads)
  length := a.sequence (job.pids 1/p.heads)-a.prompt (job.pids 1/p.heads)
  prompt := a.prompt (job.pids 1/p.heads)

def active (p : Layout) (a : FPReference.Inputs α) (i : TileIndex [p.BM,p.D]) : Prop :=
  (FPReference.row p a i.1.val : Int)<a.length

def address (p : Layout) (a : FPReference.Inputs α) (i : TileIndex [p.BM,p.D]) : Nat :=
  ((a.start+FPReference.row p a i.1.val)*p.outRow+
    (a.head*p.outHead : Nat)+(i.2.1.val*p.outDim : Nat)).toNat

def writes (p : Layout) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) (job : Launch) (a : Arrays α) : List (Nat×α) :=
  let input:=operands p job a
  let values:=FPReference.output p M dots scale input
  (TileShape.allIndices [p.BM,p.D]).filterMap fun i=>
    if active p input i then some (address p input i,values i) else none

/-- Simultaneous role update for one real-cell store. Region equality is
explicit; aliased metadata observes the existing typed-read default. -/
def writeArrays (r : Regions) (a : Arrays α) (off : Nat) (value : α) : Arrays α where
  query := if r.query=r.out then Function.update a.query off value else a.query
  key := if r.key=r.out then Function.update a.key off value else a.key
  value := if r.value=r.out then Function.update a.value off value else a.value
  start := if r.start=r.out then Function.update a.start off 0 else a.start
  sequence := if r.sequence=r.out then Function.update a.sequence off 0 else a.sequence
  prompt := if r.prompt=r.out then Function.update a.prompt off 0 else a.prompt

def update (r : Regions) (changes : List (Nat×α)) (a : Arrays α) : Arrays α :=
  changes.foldl (fun current w=>writeArrays r current w.1 w.2) a

/-- Initial arrays visible to the indexed launch after preceding stores. -/
def before (p : Layout) (r : Regions) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) : List Launch → Arrays α → Nat → Arrays α
  | [],a,_=>a
  | _::_,a,0=>a
  | job::rest,a,k+1=>before p r M dots scale rest (update r (writes p M dots scale job a) a) k

/-- Source-free ordered store list, preserving arbitrary scheduling and all
last-writer behavior, including duplicate output addresses within a tile. -/
def allWrites (p : Layout) (r : Regions) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) : List Launch → Arrays α → List (Nat×α)
  | [],_=>[]
  | job::rest,a=>
    let changes:=writes p M dots scale job a
    changes++allWrites p r M dots scale rest (update r changes a)

def lastValue (changes : List (Nat×α)) (off : Nat) : Option α :=
  changes.foldl (fun old w=>if off=w.1 then some w.2 else old) none

def finalValue [Inhabited α] (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) (a : Arrays α)
    (off : Nat) : α := (lastValue (allWrites p r M dots scale jobs a) off).getD default

/-- Metadata determines a finite signed key-step list. -/
def keySteps (p : Layout) (cur stop : Int) : List Int :=
  if hz : p.BN=0 then [] else
  if hc : cur<stop then cur::keySteps p (cur+p.BN) stop else []
termination_by (stop-cur).toNat
decreasing_by omega

def queryReads (p : Layout) (a : FPReference.Inputs α) : List Nat :=
  (TileShape.allIndices [p.BM,p.D]).filterMap fun i=>
    if active p a i then some
      (((a.start+FPReference.row p a i.1.val)*p.qRow+
        (a.head*p.qHead : Nat)+(i.2.1.val*p.qDim : Nat)).toNat) else none

def keyReads (p : Layout) (a : FPReference.Inputs α) : List Nat :=
  (keySteps p 0 ((FPReference.blockMask p a : Int)*FPReference.keyEnd p a)).flatMap fun cur=>
    (TileShape.allIndices [p.D,p.BN]).filterMap fun i=>
      if cur+i.2.1.val<FPReference.keyEnd p a then some
        (((a.batch*p.kBatch : Nat)+(cur+i.2.1.val)*p.kRow+
          (a.kvHead*p.kHead : Nat)+(i.1.val*p.kDim : Nat)).toNat) else none

def valueReads (p : Layout) (a : FPReference.Inputs α) : List Nat :=
  (keySteps p 0 ((FPReference.blockMask p a : Int)*FPReference.keyEnd p a)).flatMap fun cur=>
    (TileShape.allIndices [p.BN,p.D]).filterMap fun i=>
      if cur+i.1.val<FPReference.keyEnd p a then some
        (((a.batch*p.vBatch : Nat)+(cur+i.1.val)*p.vRow+
          (a.kvHead*p.vHead : Nat)+(i.2.1.val*p.vDim : Nat)).toNat) else none

/-- The named role's finite footprint over the evolving logical arrays. -/
def readFootprint (p : Layout) (r : Regions) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) (role : Fin 6) : List Launch → Arrays α → List Nat
  | [],_=>[]
  | job::jobs,a=>
    let input:=operands p job a
    let here:=if role.val=0 then queryReads p input else if role.val=1 then keyReads p input
      else if role.val=2 then valueReads p input else [input.batch]
    here++readFootprint p r M dots scale role jobs (update r (writes p M dots scale job a) a)

#specNonCircular allWrites avoiding [FPReference.inputs,FP.Structural.exec,FP.Structural.run,kernel]
#specNonCircular readFootprint avoiding [FPReference.inputs,FP.Structural.exec,FP.Structural.run,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeFormula

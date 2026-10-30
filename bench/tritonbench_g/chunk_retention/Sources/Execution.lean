import bench.tritonbench_g.chunk_retention.Sources.Source

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

inductive Region where
  | q | k | v | seed | incoming | terminal
  | history | finalState | output | dhistory | dq | dk | dvPart | dv
  deriving DecidableEq

structure Address where
  region : Region
  head : Nat
  slice : Nat
  row : Nat
  col : Nat
  deriving DecidableEq

abbrev Heap (α : Type) := Address → α

def Heap.write (m : Heap α) (a : Address) (value : α) : Heap α :=
  fun b => if b=a then value else m b

inductive Phase where
  | h | o | dh | qkv | sum
  deriving DecidableEq

def inputRegions : List Region := [.q,.k,.v,.seed,.incoming,.terminal]
def Phase.reads : Phase → List Region
  | .h => inputRegions
  | .o => .history::inputRegions
  | .dh => inputRegions
  | .qkv => .history::.dhistory::inputRegions
  | .sum => [.dvPart]
def Phase.writes : Phase → List Region
  | .h => [.history,.finalState]
  | .o => [.output]
  | .dh => [.dhistory]
  | .qkv => [.dq,.dk,.dvPart]
  | .sum => [.dv]

theorem Phase.separate (p : Phase) (r : Region) (hw : r∈p.writes) : r∉p.reads := by
  cases p <;> cases r <;> simp_all [Phase.writes,Phase.reads,inputRegions]

def project (ops : Ops α) (p : Phase) (m : Heap α) : Heap α :=
  fun a => if a.region∈p.reads then m a else ops.zero

def inputsOfHeap (m : Heap α) : Inputs α :=
  ⟨fun h r e => m ⟨.q,h,0,r,e⟩,fun h r e => m ⟨.k,h,0,r,e⟩,
   fun h r j => m ⟨.v,h,0,r,j⟩,fun h e j => m ⟨.seed,h,0,e,j⟩,
   fun h r j => m ⟨.incoming,h,0,r,j⟩,fun h e j => m ⟨.terminal,h,0,e,j⟩⟩

namespace Source
/-- Register recurrences are expanded into their scalar expressions. Reads use
the current phase memory; H/dH operands below are actual previously stored words.
This is a scalarized source interpreter, not the generic DSL interpreter. -/
def bodyRHS (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) (m : Heap α) (a : Address) : α :=
  let xs := inputsOfHeap m
  match p with
  | .h => if a.region=.history then ops.cast (Reference.history ops c xs a.head a.slice a.row a.col)
      else Reference.finalState ops c xs a.head a.row a.col
  | .o => Body.output ops c xs scale a.head (a.row/64)
      (fun e j => m ⟨.history,a.head,a.row/64,e,j⟩) (a.row%64) a.col
  | .dh => ops.cast (Reference.dhistory ops c xs scale a.head a.slice a.row a.col)
  | .qkv => if a.region=.dq then Body.dq ops c xs scale a.head (a.row/64)
        (fun e j => m ⟨.history,a.head,a.row/64,e,j⟩) (a.row%64) a.col
      else if a.region=.dk then Body.dk ops c xs scale a.head (a.row/64)
        (fun e j => m ⟨.dhistory,a.head,a.row/64,e,j⟩) (a.row%64) a.col
      else Body.dvPartial ops c xs scale a.slice a.head (a.row/64)
        (fun e j => m ⟨.dhistory,a.head,a.row/64,e,j⟩) (a.row%64) a.col
  | .sum => schedule.eval ops (fun block => m ⟨.dvPart,a.head,block,a.row,a.col⟩)

def rhs (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) (m : Heap α) := bodyRHS ops c scale schedule p (project ops p m)

/-- Every event receives the memory returned by its predecessor. -/
def execute (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) : List Address → Heap α → Heap α
  | [],m => m
  | a::rest,m => execute ops c scale schedule p rest
      (m.write a (rhs ops c scale schedule p m a))

theorem project_write (ops : Ops α) (p : Phase) (m : Heap α) (a : Address) (value : α)
    (ha : a.region∈p.writes) : project ops p (m.write a value)=project ops p m := by
  funext b
  unfold project Heap.write
  by_cases hb : b.region∈p.reads
  · have hne : b≠a := by
      intro h
      subst b
      exact p.separate a.region ha hb
    simp [hb,hne]
  · simp [hb]

theorem rhs_write (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) (m : Heap α) (a : Address) (value : α) (ha : a.region∈p.writes) :
    rhs ops c scale schedule p (m.write a value)=rhs ops c scale schedule p m := by
  unfold rhs
  rw [project_write ops p m a value ha]

/-- Complete current-memory execution theorem. The footprint hypothesis is
structural and later instantiated with every event in all four concrete grids. -/
theorem execute_spec (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) (events : List Address) (footprint : ∀ a∈events, a.region∈p.writes)
    (entry : Heap α) (point : Address) :
    execute ops c scale schedule p events entry point=
      if point∈events then rhs ops c scale schedule p entry point else entry point := by
  induction events generalizing entry with
  | nil => simp [execute]
  | cons a rest ih =>
    have ha := footprint a (by simp)
    have hr : ∀ b∈rest, b.region∈p.writes := fun b hb => footprint b (by simp [hb])
    simp only [execute,ih hr,rhs_write ops c scale schedule p entry a _ ha]
    by_cases hrest : point∈rest
    · simp [hrest]
    · by_cases he : point=a
      · subst point
        simp [hrest,Heap.write]
      · simp [hrest,he,Heap.write]

theorem execute_frame (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (p : Phase) (events : List Address) (footprint : ∀ a∈events, a.region∈p.writes)
    (entry : Heap α) (point : Address) (outside : point.region∉p.writes) :
    execute ops c scale schedule p events entry point=entry point := by
  rw [execute_spec ops c scale schedule p events footprint]
  exact if_neg (fun h => outside (footprint point h))

/-- Logical lane enumeration with exactly the source's padding mask. -/
def tile (tag : Region) (h s rb cb nr nc rows cols : Nat) : List Address :=
  (List.range nr).flatMap fun i => (List.range nc).filterMap fun j =>
    if rb+i<rows ∧ cb+j<cols then some ⟨tag,h,s,rb+i,cb+j⟩ else none

def programs (nx ny nz : Nat) (body : Nat→Nat→Nat→List Address) : List Address :=
  (List.range nz).flatMap fun h => (List.range ny).flatMap fun y =>
    (List.range nx).flatMap fun x => body x y h

def eventsH (c : Shape) (returnFinal : Bool) : List Address :=
  programs c.keyBlocks c.valueBlocks c.heads fun kb vb h =>
    ((List.range c.chunks).flatMap fun t =>
      tile .history h t (kb*c.BK) (vb*c.BV) c.BK c.BV c.K c.V) ++
    if returnFinal then tile .finalState h 0 (kb*c.BK) (vb*c.BV) c.BK c.BV c.K c.V else []
def eventsO (c : Shape) : List Address :=
  programs c.valueBlocks c.chunks c.heads fun vb t h =>
    tile .output h 0 (t*64) (vb*c.BV) 64 c.BV c.T c.V
def eventsDH (c : Shape) : List Address :=
  programs c.keyBlocks c.valueBlocks c.heads fun kb vb h =>
    (List.range c.chunks).flatMap fun n =>
      tile .dhistory h (c.chunks-1-n) (kb*c.BK) (vb*c.BV) c.BK c.BV c.K c.V
def eventsQKV (c : Shape) : List Address :=
  programs c.keyBlocks c.chunks c.heads fun kb t h =>
    ((List.range c.valueBlocks).flatMap fun vb =>
      tile .dvPart h kb (t*64) (vb*c.BV) 64 c.BV c.T c.V) ++
    tile .dq h 0 (t*64) (kb*c.BK) 64 c.BK c.T c.K ++
    tile .dk h 0 (t*64) (kb*c.BK) 64 c.BK c.T c.K
def eventsSum (c : Shape) : List Address :=
  programs c.heads 1 1 fun h _ _ => tile .dv h 0 0 0 c.T c.V c.T c.V

/-- Explicit forward/backward launch order, followed by the ordinary Tensor
scalar-add schedule. Empty grids enumerate no stores; T=0 H still stores seed. -/
def run (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree)
    (returnFinal : Bool) (entry : Heap α) : Heap α :=
  let mh := execute ops c scale schedule .h (eventsH c returnFinal) entry
  let mo := execute ops c scale schedule .o (eventsO c) mh
  let mdh := execute ops c scale schedule .dh (eventsDH c) mo
  let mqkv := execute ops c scale schedule .qkv (eventsQKV c) mdh
  execute ops c scale schedule .sum (eventsSum c) mqkv
end Source

#axiomsClean Source.execute_spec
#axiomsClean Source.execute_frame
#specNonCircular Reference.output avoiding [Source.run,Source.execute,Source.bodyRHS]
#specNonCircular Reference.dq avoiding [Source.run,Source.execute,Source.bodyRHS]
#specNonCircular Reference.dk avoiding [Source.run,Source.execute,Source.bodyRHS]
#specNonCircular Reference.dv avoiding [Source.run,Source.execute,Source.bodyRHS]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001

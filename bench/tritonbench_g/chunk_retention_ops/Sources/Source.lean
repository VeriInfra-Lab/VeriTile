import bench.tritonbench_g.chunk_retention_ops.Sources.Math
import bench.tritonbench_g.chunk_retention_ops.Sources.Arithmetic
import bench.tritonbench_g.chunk_retention_ops.Sources.Kernels
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
abbrev Tile (α : Type) := Nat → Nat → α

def Inputs.map (f : α→β) (xs : Inputs α) : Inputs β :=
  ⟨fun h r e => f (xs.q h r e),fun h r e => f (xs.k h r e),
   fun h r j => f (xs.v h r j),fun h e j => f (xs.seed h e j),
   fun h r j => f (xs.incoming h r j),fun h e j => f (xs.finalIncoming h e j)⟩

/-- Retains the actual source int-to-FP, subtraction, exp2, log2 and multiply sites.
The source's exact multiply by 1 is removed by all inspected compiler profiles. -/
def logRate (ops : Ops α) (c : Shape) (head : Nat) : α :=
  ops.log2 (ops.sub (ops.integer 1)
    (ops.exp2 (ops.sub (ops.integer (-5)) (ops.integer (head%c.H)))))
def decay (ops : Ops α) (c : Shape) (head : Nat) (exponent : Int) : α :=
  ops.exp2 (ops.mul (ops.integer exponent) (logRate ops c head))

def masked (ops : Ops α) (rows cols : Nat) (x : Tile α) (r d : Nat) : α :=
  if r<rows ∧ d<cols then x r d else ops.zero

def qAt (ops : Ops α) (c : Shape) (xs : Inputs α) (h r e : Nat) :=
  masked ops c.T c.K (xs.q h) r e
def kAt (ops : Ops α) (c : Shape) (xs : Inputs α) (h r e : Nat) :=
  masked ops c.T c.K (xs.k h) r e
def vAt (ops : Ops α) (c : Shape) (xs : Inputs α) (h r j : Nat) :=
  masked ops c.T c.V (xs.v h) r j
def gAt (ops : Ops α) (c : Shape) (xs : Inputs α) (h r j : Nat) :=
  masked ops c.T c.V (xs.incoming h) r j

def causalWeight (ops : Ops α) (c : Shape) (h row time : Nat) : α :=
  if time≤row then decay ops c h ((row:Int)-time) else ops.zero

namespace Body
/-- H is saved before this update. All 64 temporal lanes participate, with
explicit zero-padding loads. Carry is the fused dot seed after decay rounding. -/
def nextH (ops : Ops α) (c : Shape) (xs : Inputs α) (h chunk : Nat) (carry : Tile α) : Tile α :=
  fun e j => chain ops 64
    (fun t => kAt ops c xs h (chunk*64+t) e)
    (fun t => ops.cast (ops.mul (vAt ops c xs h (chunk*64+t) j)
      (decay ops c h ((c.length chunk:Int)-t-1))))
    (ops.mul (decay ops c h (c.length chunk)) (carry e j))

def nextDH (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (h chunk : Nat) (carry : Tile α) : Tile α :=
  fun e j => chain ops 64
    (fun t => ops.cast (ops.mul (qAt ops c xs h (chunk*64+t) e) scale))
    (fun t => ops.cast (ops.mul (gAt ops c xs h (chunk*64+t) j) (decay ops c h (t+1))))
    (ops.mul (decay ops c h (c.length chunk)) (carry e j))

def outputCarry (ops : Ops α) (c : Shape) (xs : Inputs α)
    (h chunk row j : Nat) (history : Tile α) : α :=
  blockChain ops c.BK
    (fun e => ops.cast (ops.mul (qAt ops c xs h (chunk*64+row) e) (decay ops c h (row+1))))
    (fun e => masked ops c.K c.V history e j) c.keyBlocks ops.zero

def score (ops : Ops α) (c : Shape) (xs : Inputs α) (h chunk row time : Nat) : α :=
  blockChain ops c.BK (fun e => qAt ops c xs h (chunk*64+row) e)
    (fun e => kAt ops c xs h (chunk*64+time) e) c.keyBlocks ops.zero

def output (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (h chunk : Nat) (history : Tile α) : Tile α :=
  fun r j => ops.cast (ops.mul
    (chain ops 64
      (fun t => ops.cast (ops.mul (score ops c xs h chunk r t) (causalWeight ops c h r t)))
      (fun t => vAt ops c xs h (chunk*64+t) j)
      (outputCarry ops c xs h chunk r j history)) scale)

def ds (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h chunk row time : Nat) : α :=
  ops.cast (ops.mul
    (blockChain ops c.BV (fun j => gAt ops c xs h (chunk*64+row) j)
      (fun j => vAt ops c xs h (chunk*64+time) j) c.valueBlocks ops.zero)
    (ops.mul (causalWeight ops c h row time) scale))

def dq (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (h chunk : Nat) (history : Tile α) : Tile α :=
  fun r e => ops.cast (chain ops 64 (fun t => ds ops c xs scale h chunk r t)
    (fun t => kAt ops c xs h (chunk*64+t) e)
    (ops.mul
      (blockChain ops c.BV (fun j => gAt ops c xs h (chunk*64+r) j)
        (fun j => masked ops c.K c.V history e j) c.valueBlocks ops.zero)
      (ops.cast (ops.mul (decay ops c h (r+1)) scale))))

/-- dK's final add stays separate after the transposed dot/layout transfer.
This is established for every saved dqkv LLVM/PTX profile. -/
def dk (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (h chunk : Nat) (dhistory : Tile α) : Tile α :=
  fun t e => ops.cast (ops.add
    (ops.mul
      (blockChain ops c.BV (fun j => vAt ops c xs h (chunk*64+t) j)
        (fun j => masked ops c.K c.V dhistory e j) c.valueBlocks ops.zero)
      (decay ops c h ((c.length chunk:Int)-t-1)))
    (chain ops 64 (fun r => qAt ops c xs h (chunk*64+r) e)
      (fun r => ds ops c xs scale h chunk r t) ops.zero))

def partialScore (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (block h chunk time row : Nat) : α :=
  ops.mul
    (chain ops c.BK (fun e => kAt ops c xs h (chunk*64+time) (block*c.BK+e))
      (fun e => qAt ops c xs h (chunk*64+row) (block*c.BK+e)) ops.zero)
    (ops.mul (causalWeight ops c h row time) scale)

def dvPartial (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (block h chunk : Nat) (dhistory : Tile α) : Tile α :=
  fun t j => ops.cast (chain ops 64
    (fun r => ops.cast (partialScore ops c xs scale block h chunk t r))
    (fun r => gAt ops c xs h (chunk*64+r) j)
    (ops.mul
      (chain ops c.BK (fun e => kAt ops c xs h (chunk*64+t) (block*c.BK+e))
        (fun e => masked ops c.K c.V dhistory (block*c.BK+e) j) ops.zero)
      (decay ops c h ((c.length chunk:Int)-t-1))))
end Body

/-- Explicit scalar-add tree for torch.sum(0); the runtime supplies its actual
schedule. Complete contribution coverage is a structural permutation property. -/
inductive SumTree where
  | zero : SumTree
  | leaf : Nat → SumTree
  | add : SumTree → SumTree → SumTree

def SumTree.leaves : SumTree → List Nat
  | .zero => []
  | .leaf n => [n]
  | .add a b => a.leaves++b.leaves

def SumTree.eval (ops : Ops α) (values : Nat → α) : SumTree → α
  | .zero => ops.zero
  | .leaf n => values n
  | .add a b => ops.add (a.eval ops values) (b.eval ops values)

def SumTree.Complete (tree : SumTree) (count : Nat) : Prop := tree.leaves.Perm (List.range count)

def SumTree.ascending : Nat → SumTree
  | 0 => .zero
  | n+1 => .add (ascending n) (.leaf n)

theorem SumTree.ascending_complete (n : Nat) : (ascending n).Complete n := by
  have h : (ascending n).leaves=List.range n := by
    induction n with
    | zero => rfl
    | succ n ih => simp [ascending,leaves,ih,List.range_succ]
  rw [Complete,h]

namespace Reference
/-- Input-only forward state graph, independent of memory allocation and stores. -/
def history (ops : Ops α) (c : Shape) (xs : Inputs α) (h : Nat) : Nat → Tile α
  | 0 => fun e j => ops.cast (masked ops c.K c.V (xs.seed h) e j)
  | n+1 => Body.nextH ops c xs h n (history ops c xs h n)

/-- Reverse count counts completed chunks from the end. -/
def reverseCarry (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h : Nat) : Nat → Tile α
  | 0 => fun e j => ops.cast (masked ops c.K c.V (xs.finalIncoming h) e j)
  | n+1 => Body.nextDH ops c xs scale h (c.chunks-1-n) (reverseCarry ops c xs scale h n)

def dhistory (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h chunk : Nat) : Tile α :=
  reverseCarry ops c xs scale h (c.chunks-1-chunk)
def output (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h row j : Nat) : α :=
  Body.output ops c xs scale h (row/64) (fun e j => ops.cast (history ops c xs h (row/64) e j)) (row%64) j
def finalState (ops : Ops α) (c : Shape) (xs : Inputs α) (h e j : Nat) : α :=
  ops.cast (history ops c xs h c.chunks e j)
def dq (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h row e : Nat) : α :=
  Body.dq ops c xs scale h (row/64) (fun e j => ops.cast (history ops c xs h (row/64) e j)) (row%64) e
def dk (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (h row e : Nat) : α :=
  Body.dk ops c xs scale h (row/64) (fun e j => ops.cast (dhistory ops c xs scale h (row/64) e j)) (row%64) e
def dvPart (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (block h row j : Nat) : α :=
  Body.dvPartial ops c xs scale block h (row/64) (fun e j => ops.cast (dhistory ops c xs scale h (row/64) e j)) (row%64) j
def dv (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (schedule : SumTree)
    (h row j : Nat) : α := schedule.eval ops (fun block => dvPart ops c xs scale block h row j)
end Reference

#axiomsClean SumTree.ascending_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001


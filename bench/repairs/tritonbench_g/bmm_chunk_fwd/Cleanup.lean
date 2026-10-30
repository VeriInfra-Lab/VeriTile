import bench.repairs.tritonbench_g.bmm_chunk_fwd.Kernels

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical

def config : Host.Config := ⟨32,64,32,5,2⟩
abbrev PID (h : Host.Call) := GridIndex (h.grid config)
def pid0 {h : Host.Call} (p : PID h) := (p ⟨0,by change 0<3; decide⟩).val
def pid1 {h : Host.Call} (p : PID h) := (p ⟨1,by change 1<3; decide⟩).val
def pid2 {h : Host.Call} (p : PID h) := (p ⟨2,by change 2<3; decide⟩).val
def row (h : Host.Call) (p : PID h) (i : Fin 32) := pid0 p/Host.cdiv h.shape.chunk 64*32+i.val
def column (h : Host.Call) (p : PID h) (j : Fin 64) := pid0 p%Host.cdiv h.shape.chunk 64*64+j.val
def offset (h : Host.Call) (p : PID h) (i : Fin 32) (j : Fin 64) :=
  ((pid1 p*(h.shape.chunks*h.shape.groups)+pid2 p)*h.shape.chunk+row h p i)*h.shape.chunk+column h p j
def active (h : Host.Call) (p : PID h) (i : Fin 32) (j : Fin 64) : Prop :=
  row h p i<h.shape.chunk ∧ column h p j<h.shape.chunk ∧ row h p i<column h p j
/-- Actual 32-by-64 clearing launch, including both tail masks. -/
def addresses (h : Host.Call) : List Nat :=
  (Finset.univ.toList : List (PID h)).flatMap fun p =>
    (List.finRange 32).flatMap fun i =>
      (List.finRange 64).filterMap fun j => if active h p i j then some (offset h p i j) else none

def store {α : Type} (zero : α) (state : Nat → α) (address : Nat) : Nat → α := Function.update state address zero
def run {α : Type} (h : Host.Call) (zero : α) (state : Nat → α) := (addresses h).foldl (store zero) state

def runAt {α : Type} (h : Host.Call) (base : Nat) (zero : α) (state : Nat → α) :=
  ((addresses h).map (base+·)).foldl (store zero) state

def Upper (h : Host.Call) (a : Nat) : Prop :=
  ∃ b ch r c, b<h.shape.batch ∧ ch<h.shape.chunks*h.shape.groups ∧
    r<h.shape.chunk ∧ c<h.shape.chunk ∧ r<c ∧ a=((b*(h.shape.chunks*h.shape.groups)+ch)*h.shape.chunk+r)*h.shape.chunk+c

namespace Native
def word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
def row (h : Host.Call) (p : PID h) (i : Fin 32) :=
  (word (pid0 p)).sdiv ((word h.shape.chunk+64-1).sdiv 64)*32+word i.val
def column (h : Host.Call) (p : PID h) (j : Fin 64) :=
  (word (pid0 p)).srem ((word h.shape.chunk+64-1).sdiv 64)*64+word j.val
def offset (h : Host.Call) (p : PID h) (i : Fin 32) (j : Fin 64) :=
  (word (pid1 p)*word (h.shape.chunks*h.shape.groups)+word (pid2 p))*word h.shape.chunk*word h.shape.chunk+
    row h p i*word h.shape.chunk+column h p j
def mask (h : Host.Call) (p : PID h) (i : Fin 32) (j : Fin 64) :=
  (row h p i).slt (word h.shape.chunk) && (column h p j).slt (word h.shape.chunk) &&
    (row h p i).slt (column h p j)
def addresses (h : Host.Call) : List Nat :=
  (Finset.univ.toList : List (PID h)).flatMap fun p =>
    (List.finRange 32).flatMap fun i =>
      (List.finRange 64).filterMap fun j => if mask h p i j then some (offset h p i j).toNat else none
end Native
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup

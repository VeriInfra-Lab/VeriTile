import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Indices
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Lowering

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

abbrev Matrix (α : Type) := TileIndex [16,16] → Replicas α
abbrev Rows (α : Type) := Fin 16 → Replicas α

def csrColumn {α : Type} [Inhabited α] (p : Parameters) (s : State α) (slot : Int) : Int :=
  (s.mem "cols" ((layout p s * p.colStride : Nat) + slot).toNat).read .int

def query {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State (Replicas α)) (block : Nat) : Matrix α := fun i =>
  if p.evenM || decide (row s i.1 < p.length) then
    (s.mem "Q" (qPointer p s i + block*16*p.qd)).read .real
  else broadcast (a.literal 0)
def key {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State (Replicas α)) (column : Int) (block : Nat) : Matrix α := fun i =>
  if p.evenN || decide ((i.2.1.val : Int)+column*16 < (p.length : Int)) then
    (s.mem "K" (kPointer p s i+(column*16*p.kn).toNat+block*16*p.kd)).read .real
  else broadcast (a.literal 0)
def value {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (s : State (Replicas α)) (column : Int) : Matrix α := fun i =>
  if p.evenN || decide ((i.1.val : Int)+column*16 < (p.length : Int)) then
    (s.mem "V" (vPointer p s i+(column*16*p.vn).toNat)).read .real
  else broadcast (a.literal 0)

/-- The source branch adds a freshly seeded dot. The selected native branch
passes the old QK tile to the first fixed K8 instruction. -/
def scoreStep {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (column : Int)
    (old : Matrix α) (block : Nat) : Matrix α :=
  if native then tensor a (batch := []) (query a p s block) (key a p s column block) old
  else fun i => binary a .add (old i)
    (tensor a (batch := []) (query a p s block) (key a p s column block) (fun _ => broadcast a.zero) i)
def scoreAccumulation {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (column : Int) (blocks : Nat) : Matrix α :=
  (List.range blocks).foldl (scoreStep a native p s column) (fun _ => broadcast (a.literal 0))

structure Carry (α : Type) where
  maximum : Rows α
  denominator : Rows α
  output : Matrix α
def initialCarry {α : Type} (a : Arithmetic α) : Carry α :=
  ⟨fun _ => broadcast (a.add (a.literal 0) a.negInf),
    fun _ => broadcast (a.literal 0),fun _ => broadcast (a.literal 0)⟩

/-- Initial-memory specification. Its inputs are logical coordinates and the
CSR occurrence, never interpreter registers or an execution result. -/
def update {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (column : Int) (old : Carry α) : Carry α :=
  let raw := scoreAccumulation a native p s column p.blocks
  let scores : Matrix α := fun i o =>
    let mask := if column*16+i.2.1.val ≤ (row s i.1 : Int) then a.literal 0 else a.negInf
    if native then a.fma (a.argument p.scale) (raw i o) mask
    else a.add (a.mul (raw i o) (a.argument p.scale)) mask
  let maximum : Rows α := fun i => reduceLine a a.max (fun j => scores (i,j,PUnit.unit))
  let probability : Matrix α := fun i o => a.exp (a.sub (scores i o) (maximum i.1 o))
  let denominator : Rows α := fun i => reduceLine a a.add (fun j => probability (i,j,PUnit.unit))
  let nextMax : Rows α := fun i o => a.max (old.maximum i o) (maximum i o)
  let alpha : Rows α := fun i o => a.exp (a.sub (old.maximum i o) (nextMax i o))
  let beta : Rows α := fun i o => a.exp (a.sub (maximum i o) (nextMax i o))
  let nextDen : Rows α := fun i o =>
    if native then a.fma (old.denominator i o) (alpha i o) (a.mul (denominator i o) (beta i o))
    else a.add (a.mul (alpha i o) (old.denominator i o)) (a.mul (beta i o) (denominator i o))
  let scaledProbability : Matrix α := fun i o =>
    a.mul (probability i o) (a.div (beta i.1 o) (nextDen i.1 o))
  let scaledOutput : Matrix α := fun i o => a.mul (old.output i o)
    (if native then a.mul (alpha i.1 o) (a.div (old.denominator i.1 o) (nextDen i.1 o))
     else a.mul (a.div (old.denominator i.1 o) (nextDen i.1 o)) (alpha i.1 o))
  let out : Matrix α := if native then tensor a (batch := []) scaledProbability (value a p s column) scaledOutput
    else fun i => binary a .add (scaledOutput i)
      (tensor a (batch := []) scaledProbability (value a p s column) (fun _ => broadcast a.zero) i)
  ⟨nextMax,nextDen,out⟩

def accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) (visits : Nat) : Carry α :=
  (List.range visits).foldl (fun old visit =>
    update a native p s (csrColumn p s (first p s + visit)) old) (initialCarry a)
def visitCount {α : Type} [Inhabited α] (p : Parameters) (s : State α) :=
  (last p s - first p s).toNat
def output {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State (Replicas α)) : Matrix α :=
  (accumulation a native p s (visitCount p s)).output

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

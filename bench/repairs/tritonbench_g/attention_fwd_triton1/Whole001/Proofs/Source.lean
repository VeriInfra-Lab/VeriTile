import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Arithmetic
import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Kernels

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

abbrev Tile (α : Type) := Nat → Nat → α

structure Inputs (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α

def inputsOf (c : Shape) (abi : ABI c) (m : Memory α) : Inputs α :=
  ⟨fun h r d => m (abi.q.base+c.mainOffset h r d),
   fun h r d => m (abi.k.base+c.mainOffset h r d),
   fun h r d => m (abi.v.base+c.mainOffset h r d)⟩

/-- Temporal masking is the actual source zero-fill. The feature check only
totalizes this function outside its finite tile, where the source has no lane. -/
def masked (ops : Ops α) (c : Shape) (input : Nat → Nat → Nat → α)
    (head row col : Nat) : α := if row<c.T ∧ col<c.D then input head row col else ops.zero

namespace Body

def query (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (head chunk : Nat) : Tile α :=
  fun i d => ops.cast (ops.mul (masked ops c xs.q head (chunk*32+i) d) scale)
def key (ops : Ops α) (c : Shape) (xs : Inputs α) (head chunk : Nat) : Tile α :=
  fun d i => masked ops c xs.k head (chunk*32+i) d
def value (ops : Ops α) (c : Shape) (xs : Inputs α) (head chunk : Nat) : Tile α :=
  fun i d => masked ops c xs.v head (chunk*32+i) d

/-- Explicit seeded scalar-FMA lowering. The compiler's source operand/layout
mapping remains ordinary; no unrestricted dot evaluator is passed in. -/
def score (ops : Ops α) (c : Shape) (q k : Tile α) : Tile α :=
  fun i j => chain ops c.D (q i) (fun d => k d j) ops.zero
def localOutput (ops : Ops α) (score v : Tile α) : Tile α :=
  fun i j => chain ops 32 (fun t => ops.cast (score i t)) (fun t => v t j) ops.zero
def output (ops : Ops α) (c : Shape) (q carry within : Tile α) (first : Bool) : Tile α :=
  if first then within else
    fun i j => chain ops c.D (q i) (fun d => ops.cast (carry d j)) (within i j)
def nextCarry (ops : Ops α) (k v carry : Tile α) (first : Bool) : Tile α :=
  fun i j => chain ops 32 (k i) (fun t => v t j) (if first then ops.zero else carry i j)

structure Result (α : Type) where
  output : Tile α
  carry : Tile α

def evaluate (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (head chunk : Nat) (carry : Tile α) (ifcond : Bool) : Result α :=
  let q := query ops c xs scale head chunk
  let k := key ops c xs head chunk
  let v := value ops c xs head chunk
  let s := score ops c q k
  let within := localOutput ops s v
  let first := ifcond && decide (chunk=0)
  ⟨output ops c q carry within first,nextCarry ops k v carry first⟩
end Body

namespace Reference
/-- Input-only recurrence for the actual native graph, independent of the
source memory stores/grid interpreter. Real mathematics is defined separately. -/
def carry (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (head : Nat) (ifcond : Bool) : Nat → Tile α
  | 0 => fun _ _ => ops.zero
  | n+1 => (Body.evaluate ops c xs scale head n (carry ops c xs scale head ifcond n) ifcond).carry

def output (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (head chunk : Nat) (ifcond : Bool) : Tile α :=
  (Body.evaluate ops c xs scale head chunk (carry ops c xs scale head ifcond chunk) ifcond).output
end Reference

namespace Source
structure State (α : Type) where
  memory : Memory α
  carry : Tile α

def activeRows (c : Shape) (chunk : Nat) := min 32 (c.T-chunk*32)
def historyBase (c : Shape) (abi : ABI c) (head chunk : Nat) :=
  abi.h.base+head*c.historyHead+chunk*c.D*c.D
def outputBase (c : Shape) (abi : ABI c) (head chunk : Nat) :=
  abi.o.base+head*c.headSize+chunk*32*c.D

/-- The optional H store precedes all three loads, exactly as in the source.
Reads below observe the current memory after that store. -/
def save (ops : Ops α) (c : Shape) (abi : ABI c) (store : Bool)
    (head chunk : Nat) (s : State α) : Memory α :=
  if store then Memory.store s.memory (historyBase c abi head chunk) (c.D*c.D)
    (fun lane => ops.cast (s.carry (lane/c.D) (lane%c.D))) else s.memory

def step (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head chunk : Nat) (s : State α) : State α :=
  let saved := save ops c abi store head chunk s
  let values := Body.evaluate ops c (inputsOf c abi saved) scale head chunk s.carry ifcond
  let final := Memory.store saved (outputBase c abi head chunk) (activeRows c chunk*c.D)
    (fun lane => ops.cast (values.output (lane/c.D) (lane%c.D)))
  ⟨final,values.carry⟩

def loop (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head : Nat) (entry : Memory α) : Nat → State α
  | 0 => ⟨entry,fun _ _ => ops.zero⟩
  | n+1 => step ops c abi scale store ifcond head n (loop ops c abi scale store ifcond head entry n)

def kernel (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (head : Nat) (entry : Memory α) : Memory α :=
  (loop ops c abi scale store ifcond head entry c.chunks).memory

/-- Every program receives the memory returned by the preceding program. -/
def grid (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (entry : Memory α) : Nat → Memory α
  | 0 => entry
  | n+1 => kernel ops c abi scale store ifcond n (grid ops c abi scale store ifcond entry n)

def run (ops : Ops α) (c : Shape) (abi : ABI c) (scale : α)
    (store ifcond : Bool) (entry : Memory α) : Memory α :=
  if c.T=0 ∨ c.heads=0 then entry else grid ops c abi scale store ifcond entry c.heads
end Source

#specNonCircular Reference.carry avoiding [Source.step,Source.loop,Source.grid,Source.run]
#specNonCircular Reference.output avoiding [Source.step,Source.loop,Source.grid,Source.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001

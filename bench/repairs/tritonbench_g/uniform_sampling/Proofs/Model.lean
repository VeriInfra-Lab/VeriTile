import bench.tritonbench_g.uniform_sampling.Proofs.ScalarSpec
import bench.tritonbench_g.uniform_sampling.Proofs.SourceDomain
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Copy
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.LegacySource

/-! Shared, source-bound execution definitions for the separately named FP32
repair. The integer generator is the existing ten-round Philox implementation.
Only local floating primitives are parameters; indices, masks, generator state,
logical addresses, copying, identity, and frames are not numerical parameters.
The source uses explicit FMA, preserving its recorded default native contraction.
-/
namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

abbrev block := SourceDomain.wrapperBlock
abbrev grid := SourceDomain.grid
abbrev outputIndex := SourceDomain.outputIndex

def counterCount (n : Nat) : Nat := grid n (block n) * block n

def signed64 (x : Nat) : Int := if x < 2^63 then x else (x : Int)-2^64

def alignedStart (offset B : Nat) : Nat := offset + (B-offset%B)%B

structure Generator where
  device : Nat
  seed : Nat
  offset : Nat
  deriving DecidableEq, Repr

structure Reservation where
  seed : Nat
  start : Nat
  stop : Nat
  deriving DecidableEq, Repr

/-- The checked arithmetic within the actual ATen generator mutex. The C++
primitive validates ownership before acquiring it, then reads seed and offset,
checks both unsigned additions, and commits exactly once. -/
def reserve (owner B count : Nat) (g : Generator) : Except Nat (Reservation × Generator) :=
  if g.device ≠ owner then .error 0
  else if ¬(B=512 ∨ B=1024) then .error 1
  else if count=0 ∨ count%B≠0 then .error 2
  else if alignedStart g.offset B + count > 2^64-1 then .error 3
  else .ok (⟨g.seed,alignedStart g.offset B,alignedStart g.offset B+count⟩,
    {g with offset:=alignedStart g.offset B+count})

structure Chunk where
  done : Nat
  count : Nat
  deriving DecidableEq, Repr

def nextCount (start total done : Nat) : Nat :=
  min (total-done) (2^32-(start+done)%2^32)

/-- The Python while-loop, with a decreasing remaining-counter measure. -/
def chunksAux (start : Nat) : (remaining done : Nat) → List Chunk
  | 0, _ => []
  | remaining+1, done =>
    let count := min (remaining+1) (2^32-(start+done)%2^32)
    ⟨done,count⟩ :: chunksAux start (remaining+1-count) (done+count)
termination_by remaining => remaining

def chunks (n start : Nat) : List Chunk := chunksAux start (counterCount n) 0

def chunkSize (n : Nat) (c : Chunk) : Nat := min (n-4*c.done) (4*c.count)

def chunkPrograms (B : Nat) (c : Chunk) : Nat := c.count/B

def field (f : Fin 4) : Random.Direct.Field :=
  if f.val=0 then .c0 else if f.val=1 then .c1 else if f.val=2 then .c2 else .c3

/-- Actual runtime scalar arguments start as Python FP64 payloads, then the
launcher extracts FP32. The numerical meaning is local and explicitly named. -/
structure Arithmetic (α : Type) where
  extract : Float64Bits → α
  uniform32 : Random.Word → α
  sub32 : α → α → α
  fma32 : α → α → α → α
  neg32 : α → α

structure Scalars where
  lower : Float64Bits
  upper : Float64Bits

/-- Every Philox word, including masked lanes, is evaluated. The source uses
signed64 extraction, uint32 low addition, and all four tuple fields. -/
def sourceWord (seed offset : Int) (pid B : Nat) (f : Fin 4) (lane : Fin B) : Random.Word :=
  ScalarSpec.word seed offset (pid*B+lane.val) (field f)

def affine (M : Arithmetic α) (scalars : Scalars) (word : Random.Word) : α :=
  let lower:=M.extract scalars.lower
  let upper:=M.extract scalars.upper
  M.fma32 (M.uniform32 word) (M.sub32 upper lower) lower

structure View (n : Nat) where
  handle : Nat
  device : Nat
  region : Nat
  address : Nat → Nat
  extent : Nat
  negative : Bool
  bounded : ∀ i,i<n → address i+4≤extent
  separate : ∀ i j,i<n→j<n→i≠j→Apart (address i) (address j)

structure Scratch (n : Nat) where
  region : Nat
  base : Nat
  extent : Nat
  bounded : base+4*n≤extent
  finite : extent≤2^64

def scratchAddress (scratch : Scratch n) (i : Nat) : Nat := scratch.base+4*i

def physical (M : Arithmetic α) (v : View n) (x : α) : α :=
  if v.negative then M.neg32 x else x

def logical (api : API α Store Byte) (M : Arithmetic α) (m : Nat→Store)
    (v : View n) (i : Nat) : α :=
  if v.negative then M.neg32 (api.read (m v.region) (v.address i))
  else api.read (m v.region) (v.address i)

/-- Exact source lane order within a legal serialization of one launch. -/
def lanes (B programs : Nat) : List (Nat × Fin 4 × Fin B) :=
  (List.range programs).flatMap fun p=>
    (List.finRange 4).flatMap fun f=>(List.finRange B).map fun i=>(p,f,i)

end
end VeriTile.Bench.Repairs.UniformSampling

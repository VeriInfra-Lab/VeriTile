import bench.repairs.tritonbench_g.rms_rbe_matmul.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- The canonical wrapper supplies dense strides and fixed 16×64×64 tiles.
Storage offsets remain in the original data pointers, not in these strides. -/
structure Parameters where
  (M N K headDim : Nat)
  start : Int
  rotary : Bool

def arguments (p : Parameters) : TypedSource.Arguments :=
  ⟨"X", "W", "RMS", "OUT", p.M, p.N, p.K, p.M*p.K, p.K, 1, 1, p.K, 1,
    p.M*p.N, p.N, 1, 16, 64, 64, p.headDim, p.start, p.rotary, 10000, 1/1000000⟩
def kernel (fp8 : Bool) (p : Parameters) : ComputeKernel := TypedSource.kernel fp8 (arguments p)
def prelude (fp8 : Bool) (p : Parameters) := (kernel fp8 p).surfaceBody.take 12
def loop (fp8 : Bool) (p : Parameters) : List ComputeStmt :=
  match (kernel fp8 p).surfaceBody[12]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def statistics (fp8 : Bool) (p : Parameters) := ((kernel fp8 p).surfaceBody.drop 13).take 3
def rotation (fp8 : Bool) (p : Parameters) := ((kernel fp8 p).surfaceBody.drop 16).take 1
def stores (fp8 : Bool) (p : Parameters) := (kernel fp8 p).surfaceBody.drop 17
def steps (p : Parameters) := (p.K+63)/64

/-- Generic proof infrastructure only. The public mathematical and native
interpretations construct the dot operation from sums or fixed instruction
lanes; no public premise provides an arbitrary whole-dot result. -/
structure Operations (α : Type) where
  scalar : Algebra α
  power : Option ComputeDType → α → Int → α
  decode : FloatDType → Float8E5Bits → α
  dot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    TileIndex (batch++[m,n]) → α

def engine (a : Operations α) : Algebra α :=
  let base : Algebra α := {a.scalar with
    libdevicePowi := fun p => some (a.power p)
    float8E5Bits := some a.decode
    dot := fun p {batch} {m k n} x w => some (a.dot p .real .real .fp32 {}
      (batch:=batch) (m:=m) (k:=k) (n:=n) x w)
    dotTyped := fun p left right out options {batch} {m k n} x w =>
      some (fun i => ofFloat out (a.dot p left right out options
        (fun j => toFloat left (x j)) (fun j => toFloat right (w j)) i))}
  base.withDefaultPrecision .fp32

variable {α : Type} [Inhabited α]
def pidM (p : Parameters) (s : State α) := s.pids 1 / ((p.N+63)/64)
def pidN (p : Parameters) (s : State α) := s.pids 1 % ((p.N+63)/64)
def row (p : Parameters) (s : State α) (i : Fin 16) := pidM p s*16+i.val
def col (p : Parameters) (s : State α) (j : Fin 64) := pidN p s*64+j.val
def xOffset (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [16,64]) :=
  s.pids 0*(p.M*p.K)+(row p s i.1%p.M)*p.K+i.2.1.val+step*64
def wOffset (p : Parameters) (s : State α) (step : Nat) (i : TileIndex [64,64]) :=
  i.1.val+(col p s i.2.1%p.N)*p.K+step*64
def rOffset (step : Nat) (i : TileIndex [1,64]) := i.2.1.val+step*64

def zero (a : Operations α) := a.scalar.literal (some .fp32) .real 0
def halfZero (a : Operations α) := a.scalar.cast (some .fp32) .real .fp16 (zero a)
def xValue (a : Operations α) (p : Parameters) (s : State α) (step : Nat)
    (i : TileIndex [16,64]) : α :=
  a.scalar.fp32Load (if i.2.1.val<p.K-step*64 then
    (s.mem "X" (xOffset p s step i)).read .real else zero a)
def rmsHalf (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (step lane : Nat) : α :=
  if fp8 then
    let byte := if lane<p.K-step*64 then (s.mem "RMS" (lane+step*64)).read .int else 0
    a.scalar.cast (some .fp32) .f8e5 .fp16 (a.decode .f8e5 (Float8E5Bits.ofInt byte))
  else if lane<p.K-step*64 then (s.mem "RMS" (lane+step*64)).read .fp16 else halfZero a
def wHalf (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (step : Nat) (i : TileIndex [64,64]) : α :=
  if fp8 then
    let byte := if i.1.val<p.K-step*64 then (s.mem "W" (wOffset p s step i)).read .int else 0
    a.scalar.cast (some .fp32) .real .fp16
      (a.scalar.cast (some .fp32) .f8e5 .real (a.decode .f8e5 (Float8E5Bits.ofInt byte)))
  else if i.1.val<p.K-step*64 then (s.mem "W" (wOffset p s step i)).read .fp16 else halfZero a
def scaledX (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (step : Nat) (i : TileIndex [16,64]) : α :=
  a.scalar.binary (some .fp32) .real .mul (xValue a p s step i)
    (a.scalar.cast (some .fp32) .fp16 .real (rmsHalf a fp8 p s step i.2.1.val))
def weight (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (step : Nat) (i : TileIndex [64,64]) : α :=
  a.scalar.cast (some .fp32) .fp16 .real (wHalf a fp8 p s step i)

structure Accumulator (α : Type) where
  product : TileIndex [16,64] → α
  squares : TileIndex [16,64] → α
def initial (a : Operations α) : Accumulator α := ⟨fun _ => zero a, fun _ => zero a⟩
/-- The two-argument source dot has its own FP32 result. The DSL represents
FP32 compute through its real value channel; the outer carry addition remains
a separate FP32 node. There is no source cast between this dot and the add. -/
def update (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (step : Nat) (acc : Accumulator α) : Accumulator α where
  product := fun i => a.scalar.binary (some .fp32) .real .add (acc.product i)
    (a.dot (some .fp32) .real .real .fp32 {} (batch:=[])
      (scaledX a fp8 p s step) (weight a fp8 p s step) i)
  squares := fun i => a.scalar.binary (some .fp32) .real .add (acc.squares i)
    (a.power (some .fp32)
      (a.scalar.cast (some .fp32) .real .real (xValue a p s step i)) 2)
def accumulated (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :=
  foldRange (update a fp8 p s) 0 (steps p) 1 (initial a)

/-- Signed-byte storage is an ordinary memory-format fact. It applies to
loaded input bytes, without imposing any finite-floating-value restriction. -/
def ByteMemory (s : State α) : Prop :=
  ∀ region, region="W" ∨ region="RMS" → ∀ off,
    Float8E5Bits.isInt8 ((s.mem region off).read .int)=Bool.true

end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource

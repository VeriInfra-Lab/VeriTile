import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelTailModel

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGraph
open VeriTile.Triton FP.Structural
open KernelSource (Operations Parameters)
noncomputable section

/-- Logical input coordinates, before source tiling. The RMS functions describe
the original scalar Tensor.to conversion when the wrapper needs it. Raw signed
bytes and decoded floating values remain distinct input channels. -/
structure Input (α : Type) where
  x : Nat → Nat → Nat → α
  weightHalf : Nat → Nat → α
  rmsHalf : Nat → α
  weightByte : Nat → Nat → Int
  rmsByte : Nat → Int

structure Tile where
  (batch row column : Nat)

variable {α : Type}
def zero (a : Operations α) := a.scalar.literal (some .fp32) .real 0
def halfZero (a : Operations α) := a.scalar.cast (some .fp32) .real .fp16 (zero a)
def x (a : Operations α) (p : Parameters) (v : Input α) (t : Tile)
    (block : Nat) (i : TileIndex [16,64]) :=
  a.scalar.fp32Load (if i.2.1.val<p.K-block*64 then
    v.x t.batch ((t.row*16+i.1.val)%p.M) (i.2.1.val+block*64) else zero a)
def rms (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (block lane : Nat) :=
  if fp8 then a.scalar.cast (some .fp32) .f8e5 .fp16
    (a.decode .f8e5 (Float8E5Bits.ofInt
      (if lane<p.K-block*64 then v.rmsByte (lane+block*64) else 0)))
  else if lane<p.K-block*64 then v.rmsHalf (lane+block*64) else halfZero a
def weight (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (t : Tile) (block : Nat) (i : TileIndex [64,64]) :=
  a.scalar.cast (some .fp32) .fp16 .real
    (if fp8 then a.scalar.cast (some .fp32) .real .fp16
      (a.scalar.cast (some .fp32) .f8e5 .real
        (a.decode .f8e5 (Float8E5Bits.ofInt
          (if i.1.val<p.K-block*64 then
            v.weightByte ((t.column*64+i.2.1.val)%p.N) (i.1.val+block*64) else 0))))
    else if i.1.val<p.K-block*64 then
      v.weightHalf ((t.column*64+i.2.1.val)%p.N) (i.1.val+block*64) else halfZero a)
def scaled (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (t : Tile) (block : Nat) (i : TileIndex [16,64]) :=
  a.scalar.binary (some .fp32) .real .mul (x a p v t block i)
    (a.scalar.cast (some .fp32) .fp16 .real (rms a fp8 p v block i.2.1.val))

/-- Ordered source operations. Each K64 dot starts independently; only its
result is added to the previous product. The 64 moment lanes have their own
sequential K-block carries. -/
def products (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (t : Tile) : Nat → TileIndex [16,64] → α
  | 0 => fun _ => zero a
  | count+1 => fun i => a.scalar.binary (some .fp32) .real .add
      (products a fp8 p v t count i)
      (a.dot (some .fp32) .real .real .fp32 {} (batch:=[])
        (scaled a fp8 p v t count) (weight a fp8 p v t count) i)
def moments (a : Operations α) (p : Parameters) (v : Input α)
    (t : Tile) : Nat → TileIndex [16,64] → α
  | 0 => fun _ => zero a
  | count+1 => fun i => a.scalar.binary (some .fp32) .real .add
      (moments a p v t count i)
      (a.power (some .fp32) (a.scalar.cast (some .fp32) .real .real (x a p v t count i)) 2)
def projection (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (t : Tile) (i : TileIndex [16,64]) :=
  a.scalar.binary (some .fp32) .real .mul
    (products a fp8 p v t ((p.K+63)/64) i)
    (a.scalar.unary (some .fp32) .rsqrt
      (a.scalar.binary (some .fp32) .real .add
        (a.scalar.binary (some .fp32) .real .div
          (a.scalar.reduceSum (some .fp32) (shape:=[16,64]) ⟨1,by simp⟩ Bool.false
            (moments a p v t ((p.K+63)/64)) (i.1,PUnit.unit))
          (a.scalar.literal (some .fp32) .real p.K))
        (a.scalar.literal (some .fp32) .real (1/1000000))))

def evenLane (j : Fin 64) : Fin 64 := ⟨j.val/2*2,by omega⟩
def oddLane (j : Fin 64) : Fin 64 := ⟨j.val/2*2+1,by omega⟩
def phase (a : Operations α) (p : Parameters) (t : Tile) (i : TileIndex [16,64]) :=
  a.scalar.binary (some .fp32) .real .div
    (a.scalar.cast (some .fp32) .real .real
      (a.scalar.fromInt (some .fp32) (p.start+(t.row*16+i.1.val : Nat))))
    (a.scalar.binary (some .fp32) .real .pow
      (a.scalar.literal (some .fp32) .real 10000)
      (a.scalar.binary (some .fp32) .real .div
        (a.scalar.cast (some .fp32) .real .real
          (a.scalar.fromNat (some .fp32) ((t.column*64+evenLane i.2.1)%p.headDim)))
        (a.scalar.literal (some .fp32) .real p.headDim)))

/-- An input-only expression graph. It uses neither the source interpreter,
register state, output memory nor a completed native result. -/
def output (a : Operations α) (fp8 : Bool) (p : Parameters) (v : Input α)
    (t : Tile) (i : TileIndex [16,64]) : α :=
  let value := if p.rotary then
    let u := projection a fp8 p v t (i.1,evenLane i.2.1,PUnit.unit)
    let w := projection a fp8 p v t (i.1,oddLane i.2.1,PUnit.unit)
    let phi := phase a p t i
    let cosine := a.scalar.unary (some .fp32) .cos phi
    let sine := a.scalar.unary (some .fp32) .sin phi
    let mul := a.scalar.binary (some .fp32) .real .mul
    if i.2.1.val%2=0 then a.scalar.binary (some .fp32) .real .sub (mul u cosine) (mul w sine)
    else a.scalar.binary (some .fp32) .real .add (mul u sine) (mul w cosine)
  else projection a fp8 p v t i
  a.scalar.cast (some .fp32) .real (if fp8 then .f8e5 else .fp16) value

#specNonCircular output avoiding [FP.Structural.exec, KernelSource.update, KernelSource.accumulated,
  KernelSource.normalized, KernelSource.rotated, KernelSource.scatter]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGraph

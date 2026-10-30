import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def norm (a : Operations α) (p : Parameters) (acc : Accumulator α) (i : Fin 16) : α :=
  a.scalar.unary (some .fp32) .rsqrt
    (a.scalar.binary (some .fp32) .real .add
      (a.scalar.binary (some .fp32) .real .div
        (a.scalar.reduceSum (some .fp32) (shape:=[16,64]) ⟨1,by simp⟩ Bool.false
          acc.squares (i,PUnit.unit))
        (a.scalar.literal (some .fp32) .real p.K))
      (a.scalar.literal (some .fp32) .real (1/1000000)))
def normalized (a : Operations α) (p : Parameters) (acc : Accumulator α)
    (i : TileIndex [16,64]) : α :=
  a.scalar.binary (some .fp32) .real .mul (acc.product i) (norm a p acc i.1)

def pairIndex (i : TileIndex [16,32]) (side : Fin 2) : TileIndex [16,64] :=
  TileShape.reshapeIndex (input:=[16,64]) (output:=[16,32,2]) (by decide)
    (TileShape.appendAxisIndex [16,32] i side)
def angle (a : Operations α) (p : Parameters) (s : State α) (i : TileIndex [16,32]) : α :=
  a.scalar.binary (some .fp32) .real .div
    (a.scalar.cast (some .fp32) .real .real
      (a.scalar.fromInt (some .fp32) (p.start+(row p s i.1 : Nat))))
    (a.scalar.binary (some .fp32) .real .pow
      (a.scalar.literal (some .fp32) .real 10000)
      (a.scalar.binary (some .fp32) .real .div
        (a.scalar.cast (some .fp32) .real .real
          (a.scalar.fromNat (some .fp32) ((pidN p s*64+2*i.2.1.val)%p.headDim)))
        (a.scalar.literal (some .fp32) .real p.headDim)))
def rotatedPair (a : Operations α) (p : Parameters) (s : State α)
    (values : TileIndex [16,64] → α) (i : TileIndex [16,32]) : α × α :=
  let phi := angle a p s i
  let real := values (pairIndex i 0)
  let imag := values (pairIndex i 1)
  let cosine := a.scalar.unary (some .fp32) .cos phi
  let sine := a.scalar.unary (some .fp32) .sin phi
  let mul := a.scalar.binary (some .fp32) .real .mul
  (a.scalar.binary (some .fp32) .real .sub (mul real cosine) (mul imag sine),
   a.scalar.binary (some .fp32) .real .add (mul real sine) (mul imag cosine))
def rotated (a : Operations α) (p : Parameters) (s : State α)
    (values : TileIndex [16,64] → α) (i : TileIndex [16,64]) : α :=
  if p.rotary then
    let idx := TileShape.reshapeIndex (input:=[16,32,2]) (output:=[16,64]) (by decide) i
    let parts := TileShape.splitLastIndex [16,32] idx
    let pair := rotatedPair a p s values parts.1
    if parts.2.val=0 then pair.1 else pair.2
  else values i

def outputDType (fp8 : Bool) : FloatDType := if fp8 then .f8e5 else .fp16
def outputCell (a : Operations α) (fp8 : Bool) (value : α) : Cell α :=
  .mk (outputDType fp8).toTileDType
    (ofFloat (outputDType fp8) (a.scalar.cast (some .fp32) .real (outputDType fp8) value))
def outputOffset (p : Parameters) (s : State α) (i : TileIndex [16,64]) :=
  s.pids 0*(p.M*p.N)+row p s i.1*p.N+col p s i.2.1
def active (p : Parameters) (s : State α) (i : TileIndex [16,64]) : Bool :=
  (row p s i.1<p.M)&&(col p s i.2.1<p.N)
def scatter (a : Operations α) (fp8 : Bool) (p : Parameters) (s base : State α)
    (values : TileIndex [16,64] → α) : State α :=
  (TileShape.allIndices [16,64]).foldl (fun t i =>
    if active p s i then t.write "OUT" (outputOffset p s i) (outputCell a fp8 (values i)) else t) base

end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource

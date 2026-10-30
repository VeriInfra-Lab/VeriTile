import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Returned
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Half
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Bits := BitVec 16

def pack (negative : Bool) (exp frac : Nat) : Bits :=
  BitVec.ofNat 16 ((if negative then 32768 else 0)+exp*1024+frac)

def roundEven (q : Rat) : Nat :=
  let n:=q.num.natAbs/q.den
  let rem:=q.num.natAbs%q.den
  if 2*rem > q.den || (2*rem==q.den && n%2==1) then n+1 else n

/-- Executable binary16 RN-even scalar conversion. It retains signed zero,
subnormals and infinity, and selects canonical quiet NaN 0x7e00. -/
def narrow (x : Word) : Bits :=
  match x.decode with
  | .nan .. => BitVec.ofNat 16 0x7e00
  | .infinity negative => pack negative 31 0
  | .finite negative magnitude =>
    if magnitude==0 then pack negative 0 0 else
      let estimate : Int := (magnitude.num.natAbs.log2 : Int)-magnitude.den.log2
      let exponent:=if magnitude < FP.pow2 estimate then estimate-1 else estimate
      let step : Int:=max exponent (-14)-10
      let sig:=roundEven (magnitude/FP.pow2 step)
      let carry:=sig>=2048
      let sig:=if carry then sig/2 else sig
      let exponent:=step+10+(if carry then 1 else 0)
      if exponent > 15 then pack negative 31 0
      else if sig < 1024 then pack negative 0 sig
      else pack negative (exponent+15).toNat (sig-1024)

/-- Exact binary16 to FP32 widening with explicit canonical NaN policy. -/
def widen (x : Bits) : Word :=
  let negative:=x.toNat/32768!=0
  let exponent:=x.toNat/1024%32
  let fraction:=x.toNat%1024
  if exponent==31 then
    if fraction==0 then FP.Value.infinity .fp32 negative else FP.Value.qNaN .fp32
  else
    let sig:=if exponent==0 then fraction else 1024+fraction
    let scale : Int:=(if exponent==0 then -14 else (exponent:Int)-15)-10
    let magnitude:=Rat.ofInt sig*FP.pow2 scale
    FP.round .fp32 .ieeeValue (if negative then -magnitude else magnitude) negative

#specNonCircular narrow avoiding [Source.call,Memory.run,Returned.value,Source.chunkValue]
#specNonCircular widen avoiding [Source.call,Memory.run,Returned.value,Source.chunkValue]
end Half
end VeriTile.Bench.Repairs.StreamK.Whole001

import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware
open VeriTile.Triton FP.Structural
abbrev Word := QuantizeKvTransform.MixedBits.Word
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Binary16 storage in the low sixteen bits, binary32 registers in all 32.
The native dot callback is supplied separately. This profile implements every
scalar dequantization/cast/add operation in the selected int4 source. -/
noncomputable def scalar (rest : Algebra Word) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest .ieeeValue
  { base with
    fromInt := fun p n => if p = none ∨ p = some .fp32 then FP.round .fp32 .ieeeValue (n : Rat)
      else base.fromInt p n
    binary := fun p d op x y =>
      if (p = none ∨ p = some .fp32) ∧ d = .fp16 then
        match op with
        | .add => QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.add .ieeeValue (QuantizeKvTransform.MixedBits.half x) (QuantizeKvTransform.MixedBits.half y))
        | .sub => QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.sub .ieeeValue (QuantizeKvTransform.MixedBits.half x) (QuantizeKvTransform.MixedBits.half y))
        | .mul => QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.mul .ieeeValue (QuantizeKvTransform.MixedBits.half x) (QuantizeKvTransform.MixedBits.half y))
        | _ => base.binary p d op x y
      else base.binary p d op x y
    cast := fun p src dst x =>
      match src, dst with
      | .fp16, .real | .fp16, .fp32 => QuantizeKvTransform.MixedBits.widen x
      | .real, .fp16 | .fp32, .fp16 => QuantizeKvTransform.MixedBits.narrow .ieeeValue x
      | .fp16, .fp16 => QuantizeKvTransform.MixedBits.embed (QuantizeKvTransform.MixedBits.half x)
      | .fp32, .real => x
      | _, _ => base.cast p src dst x }

/-- The source's int32 difference fits both formats exactly, so materializing
its fp32 carrier before the actual fp16 cast causes no double rounding. -/
theorem bounded_integer_cast (n : Int) (hn : -15 ≤ n ∧ n ≤ 15) :
    QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.round .fp32 .ieeeValue (n : Rat)) =
      QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.round .ieeeValue (n : Rat)) := by
  rcases hn with ⟨hl, hu⟩
  interval_cases n <;> decide +kernel

/-- The software dot has an explicit sequential fp32 accumulation order.
Binding this schedule to a GPU dot is a separate obligation; it is not an
associativity law or an interpretation as an exact real sum. -/
def dot {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → Word)
    (b : TileIndex (batch ++ [k,n]) → Word) : TileIndex (batch ++ [m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc t =>
      FP.add .ieeeValue acc (FP.mul .ieeeValue
        (QuantizeKvTransform.MixedBits.widen (a (i.1, t, PUnit.unit)))
        (QuantizeKvTransform.MixedBits.widen (b (t, i.2.1, PUnit.unit))))) (FP.Value.zero .fp32)
  | _ :: tail => fun i => dot (batch := tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

#axiomsClean bounded_integer_cast
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware

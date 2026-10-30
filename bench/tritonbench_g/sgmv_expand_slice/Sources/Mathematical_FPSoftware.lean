import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section

/-- A software comparison profile for the native dot boundary. Decode/widen
both actual low-precision operands, then increasing-k separate fp32 mul/add
with +0 seed. No equality to a tensor-core/fused dot is presumed. -/
def sequentialDot (cfg : FP.Config) (left right : Convert.Format) :
    (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    (TileIndex (batch++[m,n]) → Word)
  | [], _, k, _, a, b => fun i => (List.finRange k).foldl
      (fun acc j => FP.add cfg acc (FP.mul cfg
        (Convert.cast cfg left .single (a (i.1,j,())))
        (Convert.cast cfg right .single (b (j,i.2.1,()))))) (FP.Value.zero .fp32)
  | _::tail, _, _, _, a, b => fun i => sequentialDot cfg left right tail
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def softwareDot (cfg : FP.Config) : Dot where
  apply := fun _ left right output _ {batch} {_} {_} {_} a b =>
    match LoraExpandGemv.FPValue.formatOf left,LoraExpandGemv.FPValue.formatOf right,
        LoraExpandGemv.FPValue.formatOf output with
    | some l,some r,some o => fun i => Convert.cast cfg .single o (sequentialDot cfg l r batch a b i)
    | _,_,_ => fun _ => FP.Value.qNaN .fp32

/-- Format boundaries are direct decoded-value conversions, with same-format
identity and explicit cross-format NaN canonicalization. These examples ensure
neither half nor bfloat is replaced by a real or opaque named fp32 operation. -/
theorem format_vectors :
    Convert.cast .ieeeValue .single .half (FP.Value.ofBits .fp32 0x3f801000) =
      FP.Value.ofBits .fp32 0x3c00 ∧
    Convert.cast .ieeeValue .single .bfloat (FP.Value.ofBits .fp32 0x3f808000) =
      FP.Value.ofBits .fp32 0x3f80 ∧
    Convert.cast .ieeeValue .half .single (FP.Value.ofBits .fp32 0x8000) =
      FP.Value.ofBits .fp32 0x80000000 := by decide +kernel
#axiomsClean format_vectors
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue

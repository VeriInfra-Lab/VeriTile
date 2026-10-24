import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.StagedArithmetic
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Numeric

/-! The source interpreter uses only fixed scalar instruction ports. Their
conformance equations quantify over all input words, independently of sequence
lengths, buffers, intermediate carries, reductions and expected results. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged
open VeriTile.Triton FP.Structural
noncomputable section

def nativeArithmetic (native : Numeric.ScalarPorts) : Arithmetic Numeric.Word where
  zero := FP.Value.ofBits .fp32 0
  one := FP.Value.ofBits .fp32 1065353216
  negFive := FP.Value.ofBits .fp32 3231711232
  fromNat := fun n => native.castU32 (BitVec.ofNat 32 n)
  add := native.add
  sub := native.sub
  mul := native.mul
  fma := native.fma
  exp2 := native.exp2FTZ
  scale := fun n => native.hostScale (BitVec.ofNat 32 n)

theorem native_arithmetic_eq (u : Numeric.UnaryModels) (native : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u native) : nativeArithmetic native=Numeric.arithmetic u := by
  have ha : native.add=(Numeric.arithmetic u).add := funext (fun x => funext (fun y => laws.add x y))
  have hs : native.sub=(Numeric.arithmetic u).sub := funext (fun x => funext (fun y => laws.sub x y))
  have hm : native.mul=(Numeric.arithmetic u).mul := funext (fun x => funext (fun y => laws.mul x y))
  have hf : native.fma=(Numeric.arithmetic u).fma :=
    funext (fun x => funext (fun y => funext (fun z => laws.fma x y z)))
  have he : native.exp2FTZ=(Numeric.arithmetic u).exp2 := funext laws.exp2FTZ
  have hc : (fun n => native.castU32 (BitVec.ofNat 32 n))=(Numeric.arithmetic u).fromNat :=
    funext (fun n => laws.castU32 (BitVec.ofNat 32 n))
  have hscale : (fun n => native.hostScale (BitVec.ofNat 32 n))=(Numeric.arithmetic u).scale :=
    funext (fun n => laws.hostScale (BitVec.ofNat 32 n))
  unfold nativeArithmetic
  rw [ha,hs,hm,hf,he,hc,hscale]
  rfl

theorem native_algebra_eq (u : Numeric.UnaryModels) (native : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u native) (plans : Plans) :
    algebra (nativeArithmetic native) plans=algebra (Numeric.arithmetic u) plans := by
  rw [native_arithmetic_eq u native laws]

#axiomsClean native_arithmetic_eq
#axiomsClean native_algebra_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged

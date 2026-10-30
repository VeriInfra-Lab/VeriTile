import bench.tritonbench_g.geglu_tanh_triton.Proofs.Math
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Scalar

namespace VeriTile.Bench.Repairs.GegluTanhTriton.RealExecution
open VeriTile.Triton FP.Structural Math.Original
noncomputable section

def formula (ch : Channel) (a b dc : ℝ) : ℝ :=
  match ch with
  | .c => forward a b
  | .da => (backward dc a b).1
  | .db => (backward dc a b).2

/-- Arbitrary Real values remain arbitrary Real values; no byte encoding or
floating-point to Real surjection is used. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .mul => x*y | .add => x+y | .sub => x-y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .libdeviceTanh => Real.tanh x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => 0

theorem scalar_real (ch : Channel) (a b dc : ℝ) :
    Tile.scalar algebra ch a b dc=formula ch a b dc := by
  have argument : 0.7978845608028654 * (a+0.044715*(a*a*a)) =
      Math.Original.argument a := by
    unfold Math.Original.argument VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.argument
      VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.scale
    ring
  cases ch <;> simp only [Tile.scalar,algebra,formula,forward,backward,gate,gateFactor,
    VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.forward,
    VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.backward,
    VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.gate,
    VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.gateFactor,
    VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.scale,id_eq,argument] <;> ring

#specNonCircular formula avoiding [Tile.scalar]
#axiomsClean scalar_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.RealExecution

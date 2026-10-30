import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_RandomDomain
import VeriTile.Triton.Float.Structural

/-! Typed expression evaluation retains the canonical integer generator and
the uniform decoder's exact fp32 payload. The floating algebra receives those
bits through its existing fp32Bits operation; no random oracle is introduced.
All five child evaluations are required before the Philox field is produced.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedRandom
open VeriTile.Triton ScalarSpec FP.Structural
noncomputable section

theorem eval_philox {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (hseed : Random.IntegerType.i64.inRange seed = true)
    {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (seedOp c1Op : Op .int []) (c0Op c2Op c3Op : Op .int [B])
    (hs : FP.Structural.evalOp M p seedOp s = some (fun _ => seed))
    (h0 : FP.Structural.evalOp M p c0Op s =
      some (fun j => ((counterLow offset (i4 j)).toNat : Int)))
    (h1 : FP.Structural.evalOp M p c1Op s =
      some (fun _ => ((counterHigh offset).toNat : Int)))
    (h2 : FP.Structural.evalOp M p c2Op s = some (fun _ => 0))
    (h3 : FP.Structural.evalOp M p c3Op s = some (fun _ => 0)) :
    FP.Structural.evalOp M p
      (.philox4x32 RandomDomain.profile (Random.Direct.uniformPlan [B] 9) f
        seedOp c0Op c1Op c2Op c3Op) s =
      some (RandomDomain.wordField seed offset i4 f) := by
  rw [evalOp_unfold]
  dsimp only [Random.Direct.uniformInputShapes]
  rw [hs]
  dsimp only [bind, Option.bind]
  rw [h0]
  dsimp only [bind, Option.bind]
  rw [h1]
  dsimp only [bind, Option.bind]
  rw [h2]
  dsimp only [bind, Option.bind]
  rw [h3]
  dsimp only [bind, Option.bind]
  change (Random.Direct.evalField RandomDomain.profile (Random.Direct.uniformPlan [B] 9) f
    seed (RandomDomain.inputs offset i4)).bind _ = _
  rw [RandomDomain.field_value seed offset i4 f hseed]
  rfl

theorem eval_uniform {shape : TileShape} (v : TileIndex shape → Random.Word)
    {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (wordOp : Op .nat shape)
    (hw : FP.Structural.evalOp M p wordOp s = some (fun i => (v i).toNat)) :
    FP.Structural.evalOp M p (.uniform32 wordOp) s =
      some (fun i => M.fp32Bits ⟨(Random.uniform (v i)).bits⟩) := by
  change (do
    let values ← FP.Structural.evalOp M p wordOp s
    if Random.Direct.uniformInputsValid values then
      some (fun i => M.fp32Bits ⟨(Random.uniform (BitVec.ofNat 32 (values i))).bits⟩)
    else none) = _
  rw [hw]
  dsimp only [bind, Option.bind]
  rw [RandomDomain.uniform_inputs_valid]
  simp only [if_true, BitVec.ofNat_toNat, BitVec.setWidth_eq]

#axiomsClean eval_philox
#axiomsClean eval_uniform

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedRandom

import bench.tritonbench_g.uniform_sampling.Proofs.ScalarSpec
import VeriTile.Triton.Semantics.EvalOp

/-! Actual random expression semantics on the source's scalar/vector profile.
Each child evaluation is explicit. Range validation follows from the signed64
seed ABI and the source modular counter words, including all inactive lanes.
These are local expression lemmas, not assumed complete-kernel executions.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.RandomDomain
open VeriTile.Triton ScalarSpec
noncomputable section

def profile : Random.Direct.Profile := ⟨.i64, .u32, .u32, .u32, .u32⟩

def inputs {B : Nat} (offset : Int) (i4 : TileIndex [B] → Nat) :
    Random.Direct.Inputs (Random.Direct.uniformInputShapes [B]) :=
  ⟨fun j => (counterLow offset (i4 j)).toNat,
    fun _ => (counterHigh offset).toNat, fun _ => 0, fun _ => 0⟩

theorem word_valid (w : Random.Word) :
    Random.Direct.WordType.u32.inRange (w.toNat : Int) = true := by
  have h := w.isLt
  simp only [Random.Direct.WordType.inRange, Random.IntegerType.inRange, decide_eq_true_eq]
  constructor
  · positivity
  · exact_mod_cast h

theorem all_inputs_valid {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (hseed : Random.IntegerType.i64.inRange seed = true) :
    profile.inputsValid seed (inputs offset i4) = true := by
  have hz : Random.Direct.WordType.u32.inRange 0 = true := by decide
  simp only [Random.Direct.Profile.inputsValid, profile, inputs, hseed, word_valid, hz]
  simp

theorem inputs_words {B : Nat} (offset : Int) (i4 : TileIndex [B] → Nat) :
    (inputs offset i4).words =
      Random.Direct.uniformInput (fun j => counterLow offset (i4 j))
        (fun _ => 0) (fun _ => 0) (counterHigh offset) := by
  apply Random.Direct.Words.ext <;> funext j
  all_goals simp [inputs, Random.Direct.Inputs.words, Random.Direct.uniformInput]

/-- The dependent projection keeps the core's actual output shape. For each
concrete field this reduces to the corresponding scalar specification word. -/
def wordField {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) : TileIndex ((Random.Direct.sameShapes [B]).at f) → Nat :=
  fun i => ((Random.Direct.ofPointwise (fun j => words seed offset (i4 j))).at f i).toNat

theorem field_value {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (hseed : Random.IntegerType.i64.inRange seed = true) :
    Random.Direct.evalField profile (Random.Direct.uniformPlan [B] 9) f seed (inputs offset i4) =
      some (wordField seed offset i4 f) := by
  unfold Random.Direct.evalField
  rw [all_inputs_valid seed offset i4 hseed]
  simp only [if_pos rfl]
  rw [inputs_words, Random.Direct.uniformPlan_pointwise]
  cases f <;> rfl

theorem eval_philox {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (hseed : Random.IntegerType.i64.inRange seed = true)
    (s : BlockState) (seedOp c1Op : Op .int []) (c0Op c2Op c3Op : Op .int [B])
    (hs : evalOp seedOp s = some (Tile.scalar seed))
    (h0 : evalOp c0Op s = some ⟨fun j => (counterLow offset (i4 j)).toNat⟩)
    (h1 : evalOp c1Op s = some (Tile.scalar (counterHigh offset).toNat))
    (h2 : evalOp c2Op s = some ⟨fun _ => 0⟩)
    (h3 : evalOp c3Op s = some ⟨fun _ => 0⟩) :
    evalOp (.philox4x32 profile (Random.Direct.uniformPlan [B] 9) f
      seedOp c0Op c1Op c2Op c3Op) s =
      some ⟨wordField seed offset i4 f⟩ := by
  rw [evalOp.eq_def]
  dsimp only [Random.Direct.uniformInputShapes]
  rw [hs]
  dsimp only [bind, Option.bind, Tile.scalar]
  rw [h0]
  dsimp only [bind, Option.bind]
  rw [h1]
  dsimp only [bind, Option.bind, Tile.scalar]
  rw [h2]
  dsimp only [bind, Option.bind]
  rw [h3]
  dsimp only [bind, Option.bind]
  change (Random.Direct.evalField profile (Random.Direct.uniformPlan [B] 9) f
    seed (inputs offset i4)).bind _ = _
  rw [field_value seed offset i4 f hseed]
  rfl

theorem uniform_inputs_valid {shape : TileShape} (v : TileIndex shape → Random.Word) :
    Random.Direct.uniformInputsValid (fun i => (v i).toNat) = true := by
  simp only [Random.Direct.uniformInputsValid, List.all_eq_true, decide_eq_true_eq]
  exact fun i _ => (v i).isLt

theorem eval_uniform {shape : TileShape} (v : TileIndex shape → Random.Word)
    (s : BlockState) (wordOp : Op .nat shape)
    (hw : evalOp wordOp s = some ⟨fun i => (v i).toNat⟩) :
    evalOp (.uniform32 wordOp) s = some ⟨fun i => some (Random.uniformReal (v i))⟩ := by
  rw [evalOp.eq_def]
  simp only [hw, bind, Option.bind]
  rw [uniform_inputs_valid]
  simp

#axiomsClean all_inputs_valid
#axiomsClean inputs_words
#axiomsClean field_value
#axiomsClean eval_philox
#axiomsClean eval_uniform

end
end VeriTile.Bench.TritonBenchG.UniformSampling.RandomDomain

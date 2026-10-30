import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Index
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.DSL.RandomProfile

/-! Source ABI validity follows from the mathematical launch domain before
generation. Runtime p has no restriction here. Every offset lane is checked,
including the inactive tail lanes on which the source still computes rand. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.RandomDomain
open VeriTile.Triton SourceDomain
noncomputable section

def profile (seedType : Random.IntegerType) : Random.SourceProfile :=
  ⟨seedType, .i32, 10⟩

theorem arange_domain {B : Nat} (hB : LegalBlock B) : Random.ArangeSizeDomain B := by
  have hf := legalBlock_arange_checks hB
  obtain ⟨k, _, hk⟩ := hB
  exact ⟨hf.1, hf.2.1, k, hk⟩

theorem all_inputs_valid (seedType : Random.IntegerType) (seed : Int)
    (hseed : seedType.inRange seed = Bool.true) {N B pid : Nat}
    (hB : LegalBlock B) (hN : N < 2^31) (hpid : pid < grid N B) :
    (profile seedType).inputsValid (shape := [B]) seed
      (fun j => (Index.offset pid B j.1 : Int)) = Bool.true := by
  rw [Random.SourceProfile.inputsValid_iff]
  refine ⟨hseed, ?_⟩
  intro j
  have hi := (source_intermediates_i32 hB hN hpid j.1.isLt).2.2.2.2.2
  simpa only [profile, Random.IntegerType.inRange, decide_eq_true_eq, Index.offset, I32Rep] using hi

/-- Operand evaluation is kept explicit so the lemma cannot bypass failing
seed/offset expressions. It introduces no random backend or callback. -/
theorem eval_rand (seedType : Random.IntegerType) (seed : Int)
    (hseed : seedType.inRange seed = Bool.true) {N B pid : Nat}
    (hB : LegalBlock B) (hN : N < 2^31) (hpid : pid < grid N B)
    (u : BlockState) (seedOp : Op .int []) (offsetOp : Op .int [B])
    (hs : evalOp seedOp u = some (Tile.scalar seed))
    (ho : evalOp offsetOp u = some ⟨fun j => (Index.offset pid B j.1 : Int)⟩) :
    evalOp (.rand (profile seedType) seedOp offsetOp) u =
      some ⟨fun j => some (Random.randReal seed (Index.offset pid B j.1 : Int) .bits32 10)⟩ := by
  rw [evalOp.eq_def]
  simp only [hs, ho, bind, Option.bind, Tile.scalar]
  rw [all_inputs_valid seedType seed hseed hB hN hpid]
  rfl

end
end VeriTile.Bench.TritonBenchG.SeededDropout.RandomDomain

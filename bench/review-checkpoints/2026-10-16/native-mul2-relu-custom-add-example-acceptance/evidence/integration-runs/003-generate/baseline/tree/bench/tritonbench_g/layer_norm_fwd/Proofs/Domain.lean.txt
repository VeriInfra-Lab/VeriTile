import bench.tritonbench_g.layer_norm_fwd.Proofs.Contracts
import VeriTile.Triton.DSL.RandomProfile
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Float.UnsignedCast

/-! Bounds for the actual complete-row accesses and seven-round source RNG.
These lemmas discharge primitive bounds and profile obligations; assembling
them into the complete source safety/execution proof is still required. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

namespace Domain

def randomProfile : Random.SourceProfile := ⟨.u32, .i32, 7⟩

/-- Real operational semantics of the source's int64-load-to-uint32 cast,
before the generator accepts its mathematical integer argument. -/
theorem eval_seed_cast (seed : Int) (state : BlockState) (loaded : Op .int [])
    (hloaded : evalOp loaded state = some (Tile.scalar seed)) :
    evalOp (.castNatToInt (Op.intToUnsigned 32 loaded)) state =
      some (Tile.scalar (Formula.seedU32 seed)) := by
  have hcast (value : Op .nat []) :
      evalOp (.castNatToInt value) state = (do
        let va ← evalOp value state
        pure (⟨fun i => Int.ofNat (va.data i)⟩ : Tile .int [])) := by
    simp only [evalOp, pure]
  rw [hcast, eval_intToUnsigned, hloaded]
  simp only [bind, Option.bind, pure, Tile.scalar]
  congr 1
  ext j
  exact Int.toNat_of_nonneg (Formula.seedU32_range seed).1

theorem col_i32 {block : Nat} (hBlock : Random.ArangeSizeDomain block)
    (col : Fin block) : (col.val : Int) < 2^31 := by
  have hc := col.isLt
  have hb := hBlock.2.1
  norm_num at hb ⊢
  omega

/-- All RNG lanes, including masked-off lanes, have valid i32 offsets. The
loaded seed has already undergone the exact modular uint32 cast. -/
theorem random_inputs_valid (seed : Int) {block : Nat}
    (hBlock : Random.ArangeSizeDomain block) :
    randomProfile.inputsValid (shape := [block]) (Formula.seedU32 seed)
      (fun j => (j.1.val : Int)) = Bool.true := by
  rw [Random.SourceProfile.inputsValid_iff]
  refine ⟨?_, ?_⟩
  · simpa only [randomProfile, Random.IntegerType.inRange, decide_eq_true_eq]
      using Formula.seedU32_range seed
  · intro j
    simp only [randomProfile, Random.IntegerType.inRange, decide_eq_true_eq]
    exact ⟨by omega, col_i32 hBlock j.1⟩

theorem eval_rand (seed : Int) {block : Nat} (hBlock : Random.ArangeSizeDomain block)
    (state : BlockState) (seedOp : Op .int []) (colOp : Op .int [block])
    (hseed : evalOp seedOp state = some (Tile.scalar (Formula.seedU32 seed)))
    (hcol : evalOp colOp state = some ⟨fun j => (j.1.val : Int)⟩) :
    evalOp (.rand randomProfile seedOp colOp) state =
      some ⟨fun j => some (Random.randReal (Formula.seedU32 seed) (j.1.val : Int) .bits32 7)⟩ := by
  rw [evalOp.eq_def]
  simp only [hseed, hcol, bind, Option.bind, Tile.scalar]
  rw [random_inputs_valid seed hBlock]
  rfl

theorem row_address_lt {M N row col stride : Nat}
    (hrow : row < M) (hcol : col < N) (hstride : N ≤ stride) :
    row * stride + col < M * stride := by
  have hmul := Nat.mul_le_mul_right stride hrow
  have hmul' : (row + 1) * stride ≤ M * stride := hmul
  simp only [Nat.add_mul, Nat.one_mul] at hmul'
  omega

/-- Exact backing span for arbitrary nonnegative input strides, including
broadcast rows whose stride is smaller than N. No input-stride injectivity is
required merely to read the complete input row. -/
theorem row_address_lt_span {M N row col stride : Nat}
    (hrow : row < M) (hcol : col < N) :
    row * stride + col < (M - 1) * stride + N := by
  have hrow' : row ≤ M - 1 := by omega
  have hmul := Nat.mul_le_mul_right stride hrow'
  omega

theorem second_row_lt {M row : Nat} (hrow : row < M) : M + row < 2*M := by omega

theorem first_mask_lt {M N row col : Nat} (hrow : row < M) (hcol : col < N) :
    row * N + col < M * N := row_address_lt hrow hcol (Nat.le_refl N)

theorem second_mask_lt {M N row col : Nat} (hrow : row < M) (hcol : col < N) :
    (M + row) * N + col < 2*M*N :=
  row_address_lt (second_row_lt hrow) hcol (Nat.le_refl N)

/-- The two streams occupy disjoint halves of the returned dropout mask. -/
theorem mask_streams_disjoint {M N row other col next : Nat}
    (hrow : row < M) (hcol : col < N) (_hother : other < M) (_hnext : next < N) :
    row * N + col ≠ (M + other) * N + next := by
  have hlo := first_mask_lt hrow hcol
  have hhi : M * N ≤ (M + other) * N + next := by
    rw [Nat.add_mul]
    omega
  omega

end Domain
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd

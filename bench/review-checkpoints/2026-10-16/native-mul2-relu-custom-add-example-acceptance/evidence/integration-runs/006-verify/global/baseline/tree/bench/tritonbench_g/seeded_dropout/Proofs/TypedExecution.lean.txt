import bench.tritonbench_g.seeded_dropout.Proofs.FP
import VeriTile.Triton.Float.ProfiledRewrite

/-! Successful nonempty complete typed execution from empty registers.
The witness uses N=B=1 and PID=0, with arbitrary runtime p and an original seed
in its selected ABI. It preserves the canonical RNG and all eager operations.
The explicit consistency backend is an abstract model, not device numerics. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.TypedExecution
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def stored {α : Type} (storage : Storage) (value : α) : Cell α :=
  match storage with
  | .fp32 | .fp64 => .mk .real value
  | .fp16 => .mk .fp16 value
  | .bf16 => .mk .bf16 value
  | .f8e4 => .mk .f8e4 value
  | .f8e5 => .mk .f8e5 value

def initial {α : Type} [Inhabited α] (storage : Storage) (value : α) : State α where
  mem := fun _ _ => stored storage value
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem one_legal : SourceDomain.LegalBlock 1 := ⟨0, by decide, by decide⟩

def seedState {α : Type} [Inhabited α] (storage : Storage) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) (value : α) : State α :=
  match kind, seed with
  | .i32, seed | .i64, seed => (initial storage value).setReg "seed_arg" .int [] (fun _ => seed)
  | .u32, seed | .u64, seed => (initial storage value).setReg "seed_arg" .nat [] (fun _ => seed)

def prefixState {α : Type} [Inhabited α] (M : Algebra α) (storage : Storage)
    (kind : Random.IntegerType) (seed : SeedCarrier kind) (value : α) : State α :=
  let s := seedState storage kind seed value
  let s := s.setReg "BLOCK_SIZE_arg" .nat [] (fun _ => 1)
  let s := s.setReg "pid" .nat [] (fun _ => 0)
  let s := s.setReg "block_start" .nat [] (fun _ => 0)
  let s := s.setReg "offsets" .nat [1] (fun _ => 0)
  let s := s.setReg "mask" .bool [1] (fun _ => Bool.true)
  match storage with
  | .fp32 => s.setReg "x" .real [1] (fun _ => M.fp32Load value)
  | .fp64 => s.setReg "x" .real [1] (fun _ => value)
  | .fp16 => s.setReg "x" .fp16 [1] (fun _ => value)
  | .bf16 => s.setReg "x" .bf16 [1] (fun _ => value)
  | .f8e4 => s.setReg "x" .f8e4 [1] (fun _ => value)
  | .f8e5 => s.setReg "x" .f8e5 [1] (fun _ => value)

theorem prefix_executes (storage : Storage) (kind : Random.IntegerType)
    (p : ℝ) (seed : SeedCarrier kind) {α : Type} [Inhabited α]
    (M : Algebra α) (value : α) :
    run M ((source storage kind Bool.false "x" "y" 1 p seed 1 one_legal).surfaceBody.take 7)
      (initial storage value) = some (prefixState M storage kind seed value) := by
  cases storage <;> cases kind
  all_goals simp [source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, numeric, natLt, numericLt, bop, initial, stored,
    Region.cast, Random.i32ArangeSize, prefixState, seedState]
  all_goals rfl

def seedOp : Random.IntegerType → Op .int []
  | .i32 | .i64 => .ref .int [] "seed_arg"
  | .u32 | .u64 => .castNatToInt (.ref .nat [] "seed_arg")

def randomTile {α : Type} (M : Algebra α) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) : Values α .real [1] :=
  fun _ => M.fp32Bits ⟨(Random.rand (seedInt kind seed) 0 .bits32 10).bits⟩

theorem prefix_seed (storage : Storage) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    evalOp M (some .fp32) (seedOp kind) (prefixState M storage kind seed value) =
      some (fun _ => seedInt kind seed) := by
  cases storage <;> cases kind
  all_goals simp [evalOp_unfold, seedOp, prefixState, seedState, seedInt]

theorem prefix_offsets (storage : Storage) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    evalOp M (some .fp32) (.castNatToInt (.ref .nat [1] "offsets")) (prefixState M storage kind seed value) =
      some (fun _ => 0) := by
  cases storage
  all_goals simp [evalOp_unfold, prefixState]

theorem prefix_rand (storage : Storage) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) (hseed : kind.inRange (seedInt kind seed) = Bool.true)
    {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    evalOp M (some .fp32) (.rand (RandomDomain.profile kind) (seedOp kind)
      (.castNatToInt (.ref .nat [1] "offsets"))) (prefixState M storage kind seed value) =
      some (randomTile M kind seed) := by
  have hzero : Random.IntegerType.i32.inRange 0 = Bool.true := by decide
  have hvalid : (RandomDomain.profile kind).inputsValid (shape := [1])
      (seedInt kind seed) (fun _ => 0) = Bool.true := by
    simp [RandomDomain.profile, Random.SourceProfile.inputsValid, hseed, hzero]
  rw [evalOp_unfold]
  simp only [prefix_seed, prefix_offsets, bind, Option.bind]
  rw [hvalid]
  rfl

theorem suffix_executes (storage : Storage) (kind : Random.IntegerType)
    (p : ℝ) (seed : SeedCarrier kind) {α : Type} [Inhabited α]
    (M : Algebra α) (value : α) (random : Values α .real [1])
    (compare : Option ComputeDType → FloatDType → α → α → Bool)
    (hcompare : M.compareLt = fun precision dtype => some (compare precision dtype)) :
    ∃ v, run M ((source storage kind Bool.false "x" "y" 1 p seed 1 one_legal).surfaceBody.drop 8)
      ((prefixState M storage kind seed value).setReg "random" .real [1] random) = some v := by
  cases storage <;> cases kind
  all_goals simp [source, ComputeKernel.surfaceBody, run, step, store, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType, Option.bind, hcompare, numeric, numericLt,
    ofFloat, toFloat, Region.cast, prefixState, Random.i32ArangeSize]

theorem original_executes (storage : Storage) (kind : Random.IntegerType)
    (p : ℝ) (seed : SeedCarrier kind) (hseed : kind.inRange (seedInt kind seed) = Bool.true)
    {α : Type} [Inhabited α] (M : Algebra α)
    (compare : Option ComputeDType → FloatDType → α → α → Bool)
    (hcompare : M.compareLt = fun precision dtype => some (compare precision dtype))
    (value : α) :
    ∃ v, FP.Structural.exec M (source storage kind Bool.false "x" "y" 1 p seed 1 one_legal)
      (initial storage value) = some v := by
  have hp := prefix_executes storage kind p seed M value
  have hr := prefix_rand storage kind seed hseed M value
  obtain ⟨v, hv⟩ := suffix_executes storage kind p seed M value
    (randomTile M kind seed) compare hcompare
  refine ⟨v, ?_⟩
  change run M (source storage kind Bool.false "x" "y" 1 p seed 1 one_legal).surfaceBody
    (initial storage value) = some v
  rw [← List.take_append_drop 7 (source storage kind Bool.false "x" "y" 1 p seed 1 one_legal).surfaceBody,
    run_append, hp]
  simp only [Option.bind_some]
  cases storage <;> cases kind
  all_goals
    simp only [seedOp, RandomDomain.profile] at hr
    simp only [source, ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero, Random.i32ArangeSize] at hv ⊢
    simp only [run, step, evalExpr, evalComputeOp] at ⊢
    erw [hr]
    simpa only [run, step, evalExpr, evalComputeOp, bind, Option.bind] using hv

theorem candidate_executes (storage : Storage) (kind : Random.IntegerType)
    (p : ℝ) (seed : SeedCarrier kind) (hseed : kind.inRange (seedInt kind seed) = Bool.true)
    {α : Type} [Inhabited α] (M : Algebra α)
    (compare : Option ComputeDType → FloatDType → α → α → Bool)
    (hcompare : M.compareLt = fun precision dtype => some (compare precision dtype))
    (value : α) :
    ∃ v, FP.Structural.exec M (source storage kind Bool.true "x" "y" 1 p seed 1 one_legal)
      (initial storage value) = some v := by
  obtain ⟨v, he⟩ := original_executes storage kind p seed hseed M compare hcompare value
  exact ⟨v, (FPProof.whole_state storage kind "x" "y" 1 p seed 1 one_legal M
    (initial storage value)).symm.trans he⟩

/-- A concrete abstract backend with ordinary real arithmetic and the finite
fp32 bit decoder. Casts and loads are identities in this consistency model;
unused unary/reduction operations are assigned arbitrary total values. None
of these interpretations is an IEEE rewrite assumption of the public theorem.
The source's fixed generator still produces the actual canonical bit payload. -/
def consistencyModel : Algebra ℝ where
  literal := fun _ _ value => value
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => 0
  unary := fun _ _ value => value
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  cast := fun _ _ _ value => value
  fromNat := fun _ value => value
  fromInt := fun _ value => value
  fp32Bits := fun value => (value.decodeRat.getD 0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

/-- One concrete successful pair in the same default-fp32 engine used by the
public FP relation. All storage/seed profiles and runtime seed/p remain
quantified. Registers start empty and the sole lane is active. -/
theorem nonempty_pair (storage : Storage) (kind : Random.IntegerType)
    (p : ℝ) (seed : SeedCarrier kind)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    ∃ v,
      FP.Structural.exec (FP.ProfiledRewrite.engine consistencyModel)
        (source storage kind Bool.false "x" "y" 1 p seed 1 one_legal)
        (initial storage (1 : ℝ)) = some v ∧
      FP.Structural.exec (FP.ProfiledRewrite.engine consistencyModel)
        (source storage kind Bool.true "x" "y" 1 p seed 1 one_legal)
        (initial storage (1 : ℝ)) = some v := by
  obtain ⟨v, he⟩ := original_executes storage kind p seed hseed
    (FP.ProfiledRewrite.engine consistencyModel) (fun _ _ a b => decide (a < b)) rfl (1 : ℝ)
  exact ⟨v, he, (FPProof.whole_state storage kind "x" "y" 1 p seed 1 one_legal
    (FP.ProfiledRewrite.engine consistencyModel) (initial storage (1 : ℝ))).symm.trans he⟩

end
end VeriTile.Bench.TritonBenchG.SeededDropout.TypedExecution

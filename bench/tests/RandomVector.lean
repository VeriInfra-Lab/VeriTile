import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.Erasure

/-! Same-shaped vector seeds: source profiles, exact canonical generation,
both operand failures, empty tensors, and source precision. -/
namespace VeriTile.Bench.Tests.RandomVector
open VeriTile.Triton

private def body : ComputeKernel → List ComputeStmt
  | .mk _ _ ss => ss

private def profile? (k : ComputeKernel) : Option (Random.SourceProfile × TileShape) :=
  match (body k).reverse.head? with
  | some (.assign _ sh _ (.compute (.alg .fp32 (.randTensor cfg _ _)))) => some (cfg, sh)
  | _ => none

def vectorLoads (seeds offsets : Region .int) : ComputeKernel := triton {
  tl.region seeds = tl.int64, offsets = tl.int32
  lane = tl.arange(0, 8)
  seed = tl.load(seeds + lane)
  offset = tl.load(offsets + lane)
  sample = tl.rand(seed, offset)
}

def vectorInlineLoads (seeds offsets : Region .int) : ComputeKernel := triton {
  tl.region seeds = tl.int32, offsets = tl.int64
  sample = tl.rand(tl.load(seeds + tl.arange(0, 8)), tl.load(offsets + tl.arange(0, 8)))
}

def vectorUnsignedLoads (seeds offsets : Region .nat) : ComputeKernel := triton {
  tl.region seeds = tl.uint64, offsets = tl.uint32
  sample = tl.rand(tl.load(seeds + tl.arange(0, 8)), tl.load(offsets + tl.arange(0, 8)))
}

def vectorRounds (seeds offsets : Region .int) (rounds : Nat) : ComputeKernel := triton {
  tl.region seeds = tl.int64, offsets = tl.int64
  sample = tl.rand(tl.load(seeds + tl.arange(0, 8)), tl.load(offsets + tl.arange(0, 8)),
    n_rounds=$(_root_.VeriTile.Triton.Random.roundsConstexpr rounds))
}

def vectorMasked (seeds offsets : Region .int) (n : Nat) : ComputeKernel := triton {
  tl.region seeds = tl.int64, offsets = tl.int32
  lane = tl.arange(0, 8)
  mask = lane < $((n : Nat))
  seed = tl.load(seeds + lane, mask=mask)
  offset = tl.load(offsets + lane, mask=mask)
  sample = tl.rand(seed, offset)[:, None]
}

def pair_i32_i32 (a : Region .int) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.int32, b = tl.int32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i32_i32_profile (a : Region .int) (b : Region .int) :
    profile? (pair_i32_i32 a b) = some (⟨.i32, .i32, 10⟩, [8]) := rfl
#print axioms pair_i32_i32_profile

def pair_i32_i64 (a : Region .int) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.int32, b = tl.int64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i32_i64_profile (a : Region .int) (b : Region .int) :
    profile? (pair_i32_i64 a b) = some (⟨.i32, .i64, 10⟩, [8]) := rfl
#print axioms pair_i32_i64_profile

def pair_i32_u32 (a : Region .int) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.int32, b = tl.uint32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i32_u32_profile (a : Region .int) (b : Region .nat) :
    profile? (pair_i32_u32 a b) = some (⟨.i32, .u32, 10⟩, [8]) := rfl
#print axioms pair_i32_u32_profile

def pair_i32_u64 (a : Region .int) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.int32, b = tl.uint64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i32_u64_profile (a : Region .int) (b : Region .nat) :
    profile? (pair_i32_u64 a b) = some (⟨.i32, .u64, 10⟩, [8]) := rfl
#print axioms pair_i32_u64_profile

def pair_i64_i32 (a : Region .int) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.int64, b = tl.int32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i64_i32_profile (a : Region .int) (b : Region .int) :
    profile? (pair_i64_i32 a b) = some (⟨.i64, .i32, 10⟩, [8]) := rfl
#print axioms pair_i64_i32_profile

def pair_i64_i64 (a : Region .int) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.int64, b = tl.int64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i64_i64_profile (a : Region .int) (b : Region .int) :
    profile? (pair_i64_i64 a b) = some (⟨.i64, .i64, 10⟩, [8]) := rfl
#print axioms pair_i64_i64_profile

def pair_i64_u32 (a : Region .int) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.int64, b = tl.uint32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i64_u32_profile (a : Region .int) (b : Region .nat) :
    profile? (pair_i64_u32 a b) = some (⟨.i64, .u32, 10⟩, [8]) := rfl
#print axioms pair_i64_u32_profile

def pair_i64_u64 (a : Region .int) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.int64, b = tl.uint64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_i64_u64_profile (a : Region .int) (b : Region .nat) :
    profile? (pair_i64_u64 a b) = some (⟨.i64, .u64, 10⟩, [8]) := rfl
#print axioms pair_i64_u64_profile

def pair_u32_i32 (a : Region .nat) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.uint32, b = tl.int32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u32_i32_profile (a : Region .nat) (b : Region .int) :
    profile? (pair_u32_i32 a b) = some (⟨.u32, .i32, 10⟩, [8]) := rfl
#print axioms pair_u32_i32_profile

def pair_u32_i64 (a : Region .nat) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.uint32, b = tl.int64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u32_i64_profile (a : Region .nat) (b : Region .int) :
    profile? (pair_u32_i64 a b) = some (⟨.u32, .i64, 10⟩, [8]) := rfl
#print axioms pair_u32_i64_profile

def pair_u32_u32 (a : Region .nat) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.uint32, b = tl.uint32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u32_u32_profile (a : Region .nat) (b : Region .nat) :
    profile? (pair_u32_u32 a b) = some (⟨.u32, .u32, 10⟩, [8]) := rfl
#print axioms pair_u32_u32_profile

def pair_u32_u64 (a : Region .nat) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.uint32, b = tl.uint64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u32_u64_profile (a : Region .nat) (b : Region .nat) :
    profile? (pair_u32_u64 a b) = some (⟨.u32, .u64, 10⟩, [8]) := rfl
#print axioms pair_u32_u64_profile

def pair_u64_i32 (a : Region .nat) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.uint64, b = tl.int32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u64_i32_profile (a : Region .nat) (b : Region .int) :
    profile? (pair_u64_i32 a b) = some (⟨.u64, .i32, 10⟩, [8]) := rfl
#print axioms pair_u64_i32_profile

def pair_u64_i64 (a : Region .nat) (b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.uint64, b = tl.int64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u64_i64_profile (a : Region .nat) (b : Region .int) :
    profile? (pair_u64_i64 a b) = some (⟨.u64, .i64, 10⟩, [8]) := rfl
#print axioms pair_u64_i64_profile

def pair_u64_u32 (a : Region .nat) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.uint64, b = tl.uint32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u64_u32_profile (a : Region .nat) (b : Region .nat) :
    profile? (pair_u64_u32 a b) = some (⟨.u64, .u32, 10⟩, [8]) := rfl
#print axioms pair_u64_u32_profile

def pair_u64_u64 (a : Region .nat) (b : Region .nat) : ComputeKernel := triton {
  tl.region a = tl.uint64, b = tl.uint64
  sample = tl.rand(tl.load(a + tl.arange(0, 8)), tl.load(b + tl.arange(0, 8)))
}

theorem pair_u64_u64_profile (a : Region .nat) (b : Region .nat) :
    profile? (pair_u64_u64 a b) = some (⟨.u64, .u64, 10⟩, [8]) := rfl
#print axioms pair_u64_u64_profile

theorem signed_profile (a b : Region .int) :
    profile? (vectorLoads a b) = some (⟨.i64, .i32, 10⟩, [8]) := rfl
theorem signed_inline_profile (a b : Region .int) :
    profile? (vectorInlineLoads a b) = some (⟨.i32, .i64, 10⟩, [8]) := rfl
theorem unsigned_inline_profile (a b : Region .nat) :
    profile? (vectorUnsignedLoads a b) = some (⟨.u64, .u32, 10⟩, [8]) := rfl
theorem symbolic_rounds_profile (a b : Region .int) (rounds : Nat) :
    profile? (vectorRounds a b rounds) = some (⟨.i64, .i64, rounds⟩, [8]) := rfl
theorem zero_rounds_profile (a b : Region .int) :
    profile? (vectorRounds a b 0) = some (⟨.i64, .i64, 0⟩, [8]) := rfl
theorem both_input_regions (a b : Region .int) :
    (vectorLoads a b).inputs.length = 2 := rfl

theorem both_inline_regions (a b : Region .int) :
    (vectorInlineLoads a b).inputs.length = 2 := rfl

def promotedVector (a b : Region .int) : ComputeKernel := triton {
  tl.region a = tl.int64, b = tl.int32
  sample = tl.rand(tl.load(a + tl.arange(0, 8)),
    tl.load(b + tl.arange(0, 8))).to(tl.float64)
}

private def isRandTensor {dtype shape} : Op dtype shape → Bool
  | .randTensor _ _ _ => Bool.true
  | _ => Bool.false

private def isScopedFP32 {dtype shape} : Op dtype shape → Bool
  | .withPrecision .real .fp32 child => isRandTensor child
  | _ => Bool.false

private def isCastScoped {dtype shape} : Op dtype shape → Bool
  | .castFloat .real .real child => isScopedFP32 child
  | _ => Bool.false

private def isFixedFP32InsideFP64 (k : ComputeKernel) : Bool :=
  match (body k).reverse.head? with
  | some (.assign _ _ _ (.compute (.alg .fp64 op))) => isCastScoped op
  | _ => Bool.false

theorem promoted_retains_fp32 (a b : Region .int) :
    isFixedFP32InsideFP64 (promotedVector a b) = Bool.true := rfl

private def maskedSourceLoads (k : ComputeKernel) : Nat :=
  (body k).filterMap (fun s => match s with
    | .assign _ _ _ (.alg (.load .int _ (.mask _) _)) => some ()
    | _ => none) |>.length

theorem masked_loads_keep_unspecified (a b : Region .int) (n : Nat) :
    maskedSourceLoads (vectorMasked a b n) = 2 := rfl

/-- error: tl.rand: vector seeds require the same shape as offsets: shape mismatch -/
#guard_msgs in
def differentShapes : ComputeKernel := triton {
  seeds = tl.arange(0, 8)
  offsets = tl.arange(0, 16)
  sample = tl.rand(seeds, offsets)
}

/-- error: tl.rand: vector seeds require the same shape as offsets: shape mismatch -/
#guard_msgs in
def expandingOffsets (offset : Int) : ComputeKernel := triton {
  seeds = tl.arange(0, 8)
  offsets = $((offset : Int))
  tl.source_integer offsets = tl.int32
  sample = tl.rand(seeds, offsets, n_rounds=0)
}

/-- error: tl.rand: vector seeds require the same shape as offsets: shape mismatch -/
#guard_msgs in
def singletonVectorBroadcast : ComputeKernel := triton {
  seeds = tl.arange(0, 1)
  offsets = tl.arange(0, 8)
  sample = tl.rand(seeds, offsets)
}

/-- error: tl.rand: vector seeds require the same shape as offsets: shape mismatch -/
#guard_msgs in
def twoAxisBroadcast : ComputeKernel := triton {
  seeds = tl.arange(0, 8)[:, None]
  offsets = tl.arange(0, 8)[None, :]
  sample = tl.rand(seeds, offsets)
}

theorem real_success {sh : TileShape} (cfg : Random.SourceProfile)
    (seeds offsets : Op .int sh) (s : BlockState) (vs vo : Tile .int sh)
    (hs : evalOp seeds s = some vs) (ho : evalOp offsets s = some vo)
    (hv : cfg.tensorInputsValid vs.data vo.data = Bool.true) :
    evalOp (.randTensor cfg seeds offsets) s =
      some ⟨fun i => some (Random.randReal (vs.data i) (vo.data i)
        cfg.offsetType.offsetWidth cfg.nRounds)⟩ := by
  simp only [evalOp, hs, ho, bind, Option.bind, hv, ↓reduceIte]

theorem real_seed_failure {sh : TileShape} (cfg : Random.SourceProfile)
    (seeds offsets : Op .int sh) (s : BlockState) (hs : evalOp seeds s = none) :
    evalOp (.randTensor cfg seeds offsets) s = none := by
  simp only [evalOp, hs, bind, Option.bind]

theorem real_offset_failure {sh : TileShape} (cfg : Random.SourceProfile)
    (seeds offsets : Op .int sh) (s : BlockState) (vs : Tile .int sh)
    (hs : evalOp seeds s = some vs) (ho : evalOp offsets s = none) :
    evalOp (.randTensor cfg seeds offsets) s = none := by
  simp only [evalOp, hs, ho, bind, Option.bind]

theorem real_invalid_input {sh : TileShape} (cfg : Random.SourceProfile)
    (seeds offsets : Op .int sh) (s : BlockState) (vs vo : Tile .int sh)
    (hs : evalOp seeds s = some vs) (ho : evalOp offsets s = some vo)
    (hv : cfg.tensorInputsValid vs.data vo.data = Bool.false) :
    evalOp (.randTensor cfg seeds offsets) s = none := by
  simp only [evalOp, hs, ho, bind, Option.bind, hv, Bool.false_eq_true, ↓reduceIte]

theorem empty_valid (cfg : Random.SourceProfile) (vs vo : TileIndex [0] → Int) :
    cfg.tensorInputsValid vs vo = Bool.true := by
  rw [Random.SourceProfile.tensorInputsValid_iff]
  constructor <;> intro i <;> exact Fin.elim0 i.1

theorem empty_rank_two_valid (cfg : Random.SourceProfile) (vs vo : TileIndex [2,0] → Int) :
    cfg.tensorInputsValid vs vo = Bool.true := by
  rw [Random.SourceProfile.tensorInputsValid_iff]
  constructor <;> intro i <;> exact Fin.elim0 i.2.1

theorem rounded_success {sh : TileShape} (R : RoundingModel)
    (cfg : Random.SourceProfile) (seeds offsets : Op .int sh)
    (s : BlockState) (vs vo : Tile .int sh)
    (hs : evalOpR R seeds s = some vs) (ho : evalOpR R offsets s = some vo)
    (hv : cfg.tensorInputsValid vs.data vo.data = Bool.true) :
    evalOpR R (.randTensor cfg seeds offsets) s =
      some ⟨fun i => some (Random.randReal (vs.data i) (vo.data i)
        cfg.offsetType.offsetWidth cfg.nRounds)⟩ := by
  simp only [evalOpR, hs, ho, bind, Option.bind, hv, ↓reduceIte]

-- In contrast, scalar seed validation still happens with empty offset shape.
theorem scalar_empty_invalid :
    (⟨.i32, .i32, 10⟩ : Random.SourceProfile).inputsValid (2^31)
      (fun _ : TileIndex [0] => 0) = Bool.false := by decide

theorem invalid_seed_lane :
    (⟨.i32, .i64, 10⟩ : Random.SourceProfile).tensorInputsValid
      (fun i : TileIndex [2] => if i.1.val = 0 then 0 else 2^31) (fun _ => 0) = Bool.false := by decide

theorem invalid_offset_lane :
    (⟨.i64, .u32, 10⟩ : Random.SourceProfile).tensorInputsValid
      (fun _ : TileIndex [2] => 0) (fun i => if i.1.val = 0 then 0 else -1) = Bool.false := by decide

theorem unsigned_seed_negative :
    (⟨.u64, .i32, 10⟩ : Random.SourceProfile).tensorInputsValid
      (fun _ : TileIndex [1] => -1) (fun _ => 0) = Bool.false := by decide

theorem source_signed_extremes :
    (⟨.i64, .i32, 10⟩ : Random.SourceProfile).tensorInputsValid
      (fun i : TileIndex [2] => if i.1.val = 0 then -(2^63) else 2^63-1)
      (fun i => if i.1.val = 0 then -(2^31) else 2^31-1) = Bool.true := by decide

namespace Typed
open FP.Structural

theorem success {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (cfg : Random.SourceProfile) (seeds offsets : Op .int sh)
    (s : State α) (vs vo : Values α .int sh)
    (hs : evalOp M p seeds s = some vs) (ho : evalOp M p offsets s = some vo)
    (hv : cfg.tensorInputsValid vs vo = Bool.true) :
    evalOp M p (.randTensor cfg seeds offsets) s =
      some (fun i => M.fp32Bits
        ⟨(Random.rand (vs i) (vo i) cfg.offsetType.offsetWidth cfg.nRounds).bits⟩) := by
  rw [evalOp_unfold]
  simp only [hs, ho, bind, Option.bind, hv, ↓reduceIte]

theorem seed_failure {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (cfg : Random.SourceProfile) (seeds offsets : Op .int sh)
    (s : State α) (hs : evalOp M p seeds s = none) :
    evalOp M p (.randTensor cfg seeds offsets) s = none := by
  rw [evalOp_unfold]
  simp only [hs, bind, Option.bind]

theorem offset_failure {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (cfg : Random.SourceProfile) (seeds offsets : Op .int sh)
    (s : State α) (vs : Values α .int sh)
    (hs : evalOp M p seeds s = some vs) (ho : evalOp M p offsets s = none) :
    evalOp M p (.randTensor cfg seeds offsets) s = none := by
  rw [evalOp_unfold]
  simp only [hs, ho, bind, Option.bind]

theorem invalid_input {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (cfg : Random.SourceProfile) (seeds offsets : Op .int sh)
    (s : State α) (vs vo : Values α .int sh)
    (hs : evalOp M p seeds s = some vs) (ho : evalOp M p offsets s = some vo)
    (hv : cfg.tensorInputsValid vs vo = Bool.false) :
    evalOp M p (.randTensor cfg seeds offsets) s = none := by
  rw [evalOp_unfold]
  simp only [hs, ho, bind, Option.bind, hv, Bool.false_eq_true, ↓reduceIte]

theorem from_registers {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    {sh : TileShape} (cfg : Random.SourceProfile) (s : State α) (vs vo : Values α .int sh)
    (hv : cfg.tensorInputsValid vs vo = Bool.true) :
    evalOp M p (.randTensor cfg (.ref .int sh "seeds") (.ref .int sh "offsets"))
      ((s.setReg "seeds" .int sh vs).setReg "offsets" .int sh vo) =
      some (fun i => M.fp32Bits
        ⟨(Random.rand (vs i) (vo i) cfg.offsetType.offsetWidth cfg.nRounds).bits⟩) := by
  apply success
  · simp only [evalOp_unfold, State.setReg]
    simp
  · simp only [evalOp_unfold, State.setReg_same]
  · exact hv

theorem empty_seed_still_evaluated {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (cfg : Random.SourceProfile) (s : State α) :
    evalOp M p (.randTensor cfg (.ref .int [0] "missing") (.full [0] (.constInt 0)))
      {s with regs := fun _ _ _ => none} = none := by
  apply seed_failure
  rw [evalOp_unfold]

theorem empty_offset_still_evaluated {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (cfg : Random.SourceProfile) (s : State α) :
    evalOp M p (.randTensor cfg (.full [0] (.constInt 0)) (.ref .int [0] "missing"))
      {s with regs := fun _ _ _ => none} = none := by
  apply offset_failure (vs := fun _ => 0)
  · simp only [evalOp_unfold, bind, Option.bind, pure, Pure.pure]
  · rw [evalOp_unfold]

end Typed

theorem erasure {sh : TileShape} (cfg : Random.SourceProfile) (seeds offsets : Op .int sh) :
    (Op.randTensor cfg seeds offsets).eraseDType =
      .randTensor cfg seeds.eraseDType offsets.eraseDType := by
  rw [Op.eraseDType.eq_def]

-- Two different source seeds in one tensor must not be scalarized.
theorem distinct_seed_answers :
    (Random.rand 0 0 .bits32).raw = 0x3f4c4fd1 ∧
    (Random.rand 123 0 .bits32).raw = 0x3e091be6 := by decide +kernel

#print axioms signed_profile
#print axioms signed_inline_profile
#print axioms unsigned_inline_profile
#print axioms symbolic_rounds_profile
#print axioms zero_rounds_profile
#print axioms both_input_regions
#print axioms both_inline_regions
#print axioms promoted_retains_fp32
#print axioms masked_loads_keep_unspecified
#print axioms real_success
#print axioms real_seed_failure
#print axioms real_offset_failure
#print axioms real_invalid_input
#print axioms empty_valid
#print axioms empty_rank_two_valid
#print axioms rounded_success
#print axioms scalar_empty_invalid
#print axioms invalid_seed_lane
#print axioms invalid_offset_lane
#print axioms unsigned_seed_negative
#print axioms source_signed_extremes
#print axioms Typed.success
#print axioms Typed.seed_failure
#print axioms Typed.offset_failure
#print axioms Typed.invalid_input
#print axioms Typed.from_registers
#print axioms Typed.empty_seed_still_evaluated
#print axioms Typed.empty_offset_still_evaluated
#print axioms erasure
#print axioms distinct_seed_answers

end VeriTile.Bench.Tests.RandomVector

import bench.tritonbench_g.layer_norm_triton.Proofs.RangeStores
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits

namespace VeriTile.Bench.TritonBenchG.LayerNormLock
open VeriTile.Triton
open FP.Equational (ReductionTree ReductionPlan)
abbrev Word := FP.Value .fp32

def widenStorage : Storage → Word → Word
  | .fp32, x => x
  | .fp16, x => QuantizeKvTransform.MixedBits.widen x
  | .bf16, x => FP.cast .fp32 .ieeeValue (⟨x.bits.truncate 16⟩ : FP.Value .bf16)

def narrowStorage : Storage → Word → Word
  | .fp32, x => x
  | .fp16, x => QuantizeKvTransform.MixedBits.narrow .ieeeValue x
  | .bf16, x => ⟨(FP.cast .bf16 .ieeeValue x).bits.zeroExtend 32⟩

/-- Every used scalar/cast is concrete bits. Sqrt alone remains an explicit
primitive input for forward. Backward and final DWDB never call this field. -/
def software (sqrt32 : Word → Word) : Backend Word where
  zero := FP.Value.zero .fp32
  zeroStored := fun _ => FP.Value.zero .fp32
  one := FP.Value.ofBits .fp32 0x3f800000
  add := FP.add .ieeeValue
  sub := FP.sub .ieeeValue
  mul := FP.mul .ieeeValue
  div := FP.div .ieeeValue
  sqrt := sqrt32
  fromI32 := fun n => FP.round .fp32 .ieeeValue (Rat.ofInt n.toInt)
  cast := fun src dst x => if src = dst then x else narrowStorage dst (widenStorage src x)

/-- Only the unsupported and unused sqrt slot is filled here. The closed
backward and final kernels do not consult it; this is NOT a sqrt implementation. -/
def backwardBits : Backend Word := software (fun _ => FP.Value.qNaN .fp32)

def bitReduce (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a d => FP.add .ieeeValue (bitReduce xs a) (bitReduce xs d)

theorem reduce_software (sqrt32 : Word → Word) (xs : Fin n → Word) (tree : ReductionTree n) :
    reduce (software sqrt32) xs tree = bitReduce xs tree := by
  induction tree with
  | input => rfl
  | zero => rfl
  | add a d ih1 ih2 =>
    change FP.add .ieeeValue (reduce (software sqrt32) xs a) (reduce (software sqrt32) xs d) = _
    rw [ih1, ih2]
    rfl

theorem backward_sqrt_unused (sqrt1 sqrt2 : Word → Word) (c : Config)
    (plan : ReductionPlan c.block) (mem : Memory Word) (undef : Undef Word) (p : Nat) :
    backwardSource (software sqrt1) c plan mem undef p = backwardSource (software sqrt2) c plan mem undef p := by
  rw [backward_value, backward_value]
  simp only [backwardReference, sum, reduce_software]
  rfl

def oneLane : ReductionPlan 1 where
  tree := .input ⟨0, by decide⟩
  padding := 0
  valid := by decide

def twoRows : Config := ⟨2, 1, 1, 1, 1, {}⟩
def zero : Word := FP.Value.zero .fp32
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def big : Word := FP.Value.ofBits .fp32 0x4b800000

def exampleInputs : Memory Word := fun region address =>
  match region with
  | .dy => if address = 0 then big else one
  | .w | .rstd => one
  | _ => zero

def exampleUndef : Undef Word := fun _ _ => FP.Value.qNaN .fp32

def sourceRows : Nat → RowValues Word :=
  backwardSource backwardBits twoRows oneLane exampleInputs exampleUndef

def initialShared : Shared Word := ⟨fun _ => big, fun _ => big, fun _ => big⟩
def orderedSchedule : List Nat := List.replicate 11 0 ++ List.replicate 14 1

def completeRun : List Event × State Word :=
  run backwardBits twoRows sourceRows exampleUndef orderedSchedule
    (initial sourceRows initialShared (fun _ => 0))

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
theorem two_rows_complete_and_round :
    completeRun.2.control.pc 0 = .done ∧ completeRun.2.control.pc 1 = .done ∧
    completeRun.2.control.words 0 = 0 ∧ completeRun.2.control.words 1 = 1 ∧
    completeRun.2.values.mem.db 0 = big ∧ completeRun.2.values.mem.dw 0 = zero ∧
    completeRun.2.values.mem.dx 0 = zero ∧ completeRun.2.values.mem.dx 1 = zero ∧
    completeRun.2.values.mem.dx 2 = big ∧
    completeRun.1.count (.addDB 1 0 0 Bool.true) = 1 ∧
    completeRun.1.count (.addDB 0 0 0 Bool.true) = 0 := by
  decide +kernel

/-- Explicit cast and FP addition counterexamples forbid Real identity models. -/
theorem rounding_and_half_store_are_not_identity :
    FP.add .ieeeValue one big = big ∧
    (narrowStorage .fp16 (FP.Value.ofBits .fp32 0x45001000)).bits = (0x6800#32) ∧
    widenStorage .fp16 (narrowStorage .fp16 (FP.Value.ofBits .fp32 0x45001000)) ≠
      FP.Value.ofBits .fp32 0x45001000 := by
  decide +kernel

def negativeZeroInputs : Memory Word := fun region address =>
  if region = .dy then FP.Value.ofBits .fp32 0x80000000 else exampleInputs region address

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
theorem first_count_zero_store_preserves_negative_zero :
    (run backwardBits twoRows
      (backwardSource backwardBits twoRows oneLane negativeZeroInputs exampleUndef)
      exampleUndef (List.replicate 11 0)
      (initial (backwardSource backwardBits twoRows oneLane negativeZeroInputs exampleUndef)
        initialShared (fun _ => 0))).2.values.mem.db 0 = FP.Value.ofBits .fp32 0x80000000 ∧
    FP.add .ieeeValue zero (FP.Value.ofBits .fp32 0x80000000) = zero := by
  decide +kernel

#axiomsClean backward_sqrt_unused
#axiomsClean two_rows_complete_and_round
#axiomsClean rounding_and_half_store_are_not_identity
#axiomsClean first_count_zero_store_preserves_negative_zero
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock

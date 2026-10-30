import bench.tritonbench_g.spinning_lock_reduction.Proofs.TraceReference

namespace VeriTile.Bench.TritonBenchG.SpinningLock
open VeriTile.Triton

/-- Closed scalar software backend: fp32 nearest-even, subnormals preserved,
canonical NaN. No Real identity-rounding or external callback remains. -/
def fp32 : Backend (FP.Value .fp32) := ⟨FP.Value.zero .fp32, FP.add .ieeeValue⟩

def twoPrograms : Config :=
  ⟨2, 1, 1, 2, 2, 1, 2, 2, 1⟩

def zeroMemory : Memory (FP.Value .fp32) :=
  ⟨fun _ => FP.Value.zero .fp32, fun _ => FP.Value.ofBits .fp32 0x40400000, fun _ => 0⟩

def producerFirst : List Nat := List.replicate 55 1 ++ List.replicate 73 0

def completeRun := run fp32 twoPrograms producerFirst (initial fp32 zeroMemory)

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
theorem nine_rounds_nonempty_execution :
    allDone twoPrograms completeRun.2 = true ∧
    completeRun.2.mem.output 0 = FP.Value.zero .fp32 ∧
    completeRun.2.mem.output 1 = FP.Value.zero .fp32 ∧
    completeRun.2.mem.output 2 = zeroMemory.output 2 ∧
    completeRun.1.count (.add 0 0 1) = 9 ∧
    completeRun.1.count (.publish 1 0 1) = 9 ∧
    completeRun.1.count (.release 1) = 9 ∧
    completeRun.1.count (.output 0 0 0) = 9 ∧
    completeRun.1.count (.output 1 0 1) = 9 := by decide +kernel

/-- A real rounding event, independently of any GPU binding. -/
theorem fp32_add_rounds :
    fp32.add (FP.Value.ofBits .fp32 0x4b800000) (FP.Value.ofBits .fp32 0x3f800000) =
      FP.Value.ofBits .fp32 0x4b800000 := by decide +kernel

/-- Persistent ready locks are allowed as arbitrary machine input. The consumer
observes a stale nonzero P nine times if its publisher is not scheduled. -/
def staleMemory : Memory (FP.Value .fp32) :=
  ⟨fun _ => FP.Value.ofBits .fp32 0x3f800000,
    fun _ => FP.Value.zero .fp32, fun _ => 1⟩

def staleRun := run fp32 twoPrograms (List.replicate 73 0) (initial fp32 staleMemory)

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
theorem lock_reuse_retains_nine_accumulations :
    (staleRun.2.control 0).phase = .done ∧
    staleRun.2.mem.output 0 = FP.Value.ofBits .fp32 0x41100000 ∧
    staleRun.1.count (.add 0 0 1) = 9 ∧
    staleRun.1.count (.poll 1) = 9 := by decide +kernel

/-- One scalar addition keeps IEEE signed-zero behavior; it is not a Real sum. -/
theorem zero_sign_is_not_a_real_identity :
    fp32.add (FP.Value.zero .fp32) (FP.Value.zero .fp32 true) = FP.Value.zero .fp32 ∧
    FP.Value.zero .fp32 ≠ FP.Value.zero .fp32 true := by decide +kernel

def threePrograms : Config := ⟨3, 1, 1, 3, 3, 1, 3, 3, 1⟩

def roundingMemory : Memory (FP.Value .fp32) :=
  ⟨fun a => FP.Value.ofBits .fp32 (if a = 1 then 0x4b800000 else 0x3f800000),
    fun _ => FP.Value.zero .fp32, fun _ => 1⟩

def roundingRun := run fp32 threePrograms (List.replicate 109 0) (initial fp32 roundingMemory)

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
/-- Complete nine-round consumer execution with an actual rounding loss in
EVERY round. The independently exact rational expression is larger by nine.
No producer execution or native memory-order conclusion is claimed here. -/
theorem full_loop_rounding_occurs :
    (roundingRun.2.control 0).phase = .done ∧
    roundingRun.2.mem.output 0 = FP.Value.ofBits .fp32 0x4d100000 ∧
    (roundingRun.2.mem.output 0).toRat? = some (9 * 16777216 : Rat) ∧
    (9 * (16777216 + 1) : Rat) ≠ 9 * 16777216 ∧
    roundingRun.1.count (.add 0 0 1) = 9 ∧
    roundingRun.1.count (.add 0 0 2) = 9 := by decide +kernel

#axiomsClean full_loop_rounding_occurs
#axiomsClean nine_rounds_nonempty_execution
#axiomsClean fp32_add_rounds
#axiomsClean lock_reuse_retains_nine_accumulations
#axiomsClean zero_sign_is_not_a_real_identity
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock

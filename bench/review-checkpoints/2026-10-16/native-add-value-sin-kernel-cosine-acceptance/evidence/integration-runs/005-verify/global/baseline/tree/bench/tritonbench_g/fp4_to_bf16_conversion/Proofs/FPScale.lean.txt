import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPOperations
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPLayout
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow decodeExecution offsetExpr signedExponent intConst
attribute [local simp] decodeExecution_mem decodeExecution_undef decodeExecution_pid
  decodeExecution_n_elements_out decodeExecution_n_elements_s decodeExecution_output

/-- Scale loads occur after decoding and before every store; their masked
values remain arbitrary even when the final partial scale group is active. -/
def scaleLoadExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 28 |>.take 4)
      (decodeExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take]
  simp [run, step, evalExpr, evalComputeOp, FP.Structural.evalOp,
    backend, Fp4ToBf16.FPValue.backend, FP.Structural.numeric, FP.Structural.natLt,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind]
  try rfl

@[local simp] theorem scaleLoadExecution_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleLoadExecution rest P p s).val.regs .nat [p.groups] "s" =
      some (fun j => loadedScale p s j.1.val) := by
  simp [scaleLoadExecution, loadedScale, scaleOffset, scaleCount,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl
@[local simp] theorem scaleLoadExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleLoadExecution rest P p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j => converted (laneNibble p s j)) := by simp [scaleLoadExecution]
@[local simp] theorem scaleLoadExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleLoadExecution rest P p s).val.regs .nat [] "pid" = some (fun _ => s.pids 0) := by simp [scaleLoadExecution]
@[local simp] theorem scaleLoadExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleLoadExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun _ => p.count*2) := by simp [scaleLoadExecution]
@[local simp] theorem scaleLoadExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleLoadExecution rest P p s).val.mem = s.mem :=
  (assignment_run_memory (backend rest P) _ _ _ (by rfl) (scaleLoadExecution rest P p s).property).trans
    (decodeExecution_mem rest P p s)
attribute [local irreducible] scaleLoadExecution

def exponentExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 32 |>.take 1)
      (scaleLoadExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [offsetCode_source]
  simp [offsetCode, run, step, evalExpr, evalComputeOp, Option.bind, Option.map]
  try rfl

@[local simp] theorem exponentExecution_s_offset (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.regs .int [p.groups] "s_offset" =
      some (fun j => signed16 (signed16 (loadedScale p s j.1.val) - 127)) := by simp [exponentExecution]
@[local simp] theorem exponentExecution_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.regs .nat [p.groups] "s" =
      some (fun j => loadedScale p s j.1.val) := by simp [exponentExecution]
@[local simp] theorem exponentExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j => converted (laneNibble p s j)) := by simp [exponentExecution]
@[local simp] theorem exponentExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.regs .nat [] "pid" = some (fun _ => s.pids 0) := by simp [exponentExecution]
@[local simp] theorem exponentExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun _ => p.count*2) := by simp [exponentExecution]
@[local simp] theorem exponentExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (exponentExecution rest P p s).val.mem = s.mem :=
  (assignment_run_memory (backend rest P) _ _ _ (by rfl) (exponentExecution rest P p s).property).trans
    (scaleLoadExecution_mem rest P p s)
attribute [local irreducible] exponentExecution

def powerExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 33 |>.take 1)
      (exponentExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [powerCode_source]
  simp [powerCode, run, step, evalExpr, evalComputeOp, FP.Structural.evalOp,
    int32_values_valid, int32_bmod_values_valid, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.numeric, FP.Structural.ofFloat, FP.Structural.toFloat,
    ComputeDType.eraseDType, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind, Option.map]
  try rfl


@[local simp] theorem powerExecution_s_fp (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.regs .real [p.groups] "s_fp" = some (fun j => P.pow2 (scaleExponent (loadedScale p s j.1.val))) := by
  simp [powerExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem powerExecution_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.regs .nat [p.groups] "s" = some (fun j => loadedScale p s j.1.val) := by
  simp [powerExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem powerExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.regs .bf16 [p.block*2] "output" = some (fun j => converted (laneNibble p s j)) := by
  simp [powerExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem powerExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [powerExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem powerExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [powerExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem powerExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (powerExecution rest P p s).val.mem = s.mem :=
  (assignment_run_memory (backend rest P) _ _ _ (by rfl) (powerExecution rest P p s).property).trans
    (exponentExecution_mem rest P p s)
attribute [local irreducible] powerExecution

def scaleExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 34 |>.take 5)
      (powerExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [scaleCode_source]
  simp [scaleCode, run, step, evalExpr, evalComputeOp, FP.Structural.evalOp,
    backend, Fp4ToBf16.FPValue.backend, FP.Structural.numeric, FP.Structural.natLt,
    FP.Structural.numericEq, FP.Structural.integralMod, FP.Structural.integralFloorDiv,
    FP.Structural.ofFloat, FP.Structural.toFloat, ComputeDType.eraseDType,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind, Option.map,
    IntegerExpr.same_left, IntegerExpr.same_right]
  try rfl

@[local simp] theorem scaleExecution_s_fp (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.regs .bf16 [p.groups] "s_fp" = some (fun j => scaleValue P (loadedScale p s j.1.val)) := by
  simp [scaleExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem scaleExecution_s (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.regs .nat [p.groups] "s" = some (fun j => loadedScale p s j.1.val) := by
  simp [scaleExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem scaleExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.regs .bf16 [p.block*2] "output" = some (fun j => converted (laneNibble p s j)) := by
  simp [scaleExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem scaleExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.regs .nat [] "pid" = some (fun j => s.pids 0) := by
  simp [scaleExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem scaleExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun j => p.count*2) := by
  simp [scaleExecution, scaleExponent, scaleValue, nanScale, embed,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  try rfl

@[local simp] theorem scaleExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (scaleExecution rest P p s).val.mem = s.mem :=
  (assignment_run_memory (backend rest P) _ _ _ (by rfl) (scaleExecution rest P p s).property).trans
    (powerExecution_mem rest P p s)
attribute [local irreducible] scaleExecution
#axiomsClean scaleExecution
#axiomsClean scaleLoadExecution
#axiomsClean exponentExecution
#axiomsClean powerExecution
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue

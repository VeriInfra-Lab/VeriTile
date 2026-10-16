import bench.tritonbench_g.isfinite_kernel.Proofs.Header
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection

/-! Actual source bodies over arbitrary incoming states. Register facts are
provided by the proved source header and retained by every source iteration.
The bodies keep all eight source statements, including both block pointers,
the unspecified masked-load padding, the helper classification, and the cast
on the final Bool store. No assumption about memory, sizes, or strides is
needed merely to execute these bodies in the typed bit model. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedBits TypedSource
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def one32_run (rest : Algebra Float32Bits) (p : Parameters) (pid : Nat)
    (s : State Float32Bits) (h : OneReady pid s) :
    {t : State Float32Bits // FP.Structural.run (engine (backend32 rest))
      (oneBody32 p) s = some t} := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h
  refine ⟨?_, ?_⟩
  swap
  simp [oneBody32, arms, original32, source32,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend32, hp, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

def one64_run (rest : Algebra Float64Bits) (p : Parameters) (pid : Nat)
    (s : State Float64Bits) (h : OneReady pid s) :
    {t : State Float64Bits // FP.Structural.run (engine (backend64 rest))
      (oneBody64 p) s = some t} := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h
  refine ⟨?_, ?_⟩
  swap
  simp [oneBody64, arms, original64, source64,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend64, hp, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

def loop32_run (rest : Algebra Float32Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float32Bits) (h : LoopReady pid numCtas s) :
    {t : State Float32Bits // FP.Structural.run (engine (backend32 rest))
      (loopBody32 p) (s.setReg "j" .nat [] (fun _ => j)) = some t} := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h.1
  refine ⟨?_, ?_⟩
  swap
  simp [loopBody32, loopBody, arms, original32, source32,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend32, hp, h.2, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

def loop64_run (rest : Algebra Float64Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float64Bits) (h : LoopReady pid numCtas s) :
    {t : State Float64Bits // FP.Structural.run (engine (backend64 rest))
      (loopBody64 p) (s.setReg "j" .nat [] (fun _ => j)) = some t} := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h.1
  refine ⟨?_, ?_⟩
  swap
  simp [loopBody64, loopBody, arms, original64, source64,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend64, hp, h.2, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

theorem one32_ready (rest : Algebra Float32Bits) (p : Parameters) (pid : Nat)
    (s : State Float32Bits) (h : OneReady pid s) :
    OneReady pid (one32_run rest p pid s h).val := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h
  simp [OneReady, one32_run, hp]

theorem one64_ready (rest : Algebra Float64Bits) (p : Parameters) (pid : Nat)
    (s : State Float64Bits) (h : OneReady pid s) :
    OneReady pid (one64_run rest p pid s h).val := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h
  simp [OneReady, one64_run, hp]

theorem loop32_ready (rest : Algebra Float32Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float32Bits) (h : LoopReady pid numCtas s) :
    LoopReady pid numCtas (loop32_run rest p pid numCtas j s h).val := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h.1
  simp [LoopReady, OneReady, loop32_run, hp, h.2]

theorem loop64_ready (rest : Algebra Float64Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float64Bits) (h : LoopReady pid numCtas s) :
    LoopReady pid numCtas (loop64_run rest p pid numCtas j s h).val := by
  have hp : s.regs .nat [] "pid" = some (fun _ => pid) := h.1
  simp [LoopReady, OneReady, loop64_run, hp, h.2]

#axiomsClean one32_run
#axiomsClean one64_run
#axiomsClean loop32_run
#axiomsClean loop64_run
#axiomsClean one32_ready
#axiomsClean one64_ready
#axiomsClean loop32_ready
#axiomsClean loop64_ready
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric

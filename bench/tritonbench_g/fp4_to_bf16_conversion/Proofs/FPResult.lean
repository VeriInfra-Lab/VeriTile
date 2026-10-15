import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPScale
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow scaleExecution
attribute [local simp] scaleExecution_s_fp scaleExecution_output scaleExecution_pid
  scaleExecution_n_elements_out scaleExecution_mem

@[simp] theorem linear32 (n : Nat) : n/32*32+n%32=n := by omega

theorem reshape_rows32_any (block n : Nat) (j : TileIndex [n])
    (h : TileShape.product [block,32] = TileShape.product [n]) :
    TileShape.reshapeIndex h j =
      (⟨j.1.val / 32, by have := j.1.isLt; simp only [TileShape.product, Nat.mul_one] at h; omega⟩,
       ⟨j.1.val % 32, Nat.mod_lt _ (by decide)⟩, PUnit.unit) := by
  have hn : n=block*32 := by simpa [TileShape.product] using h.symm
  subst n
  exact reshape_rows32 block j h

def productExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 39 |>.take 4)
      (scaleExecution rest P p s).val = some t} := by
  have hmatrix : TileShape.product [p.block*2] = TileShape.product [p.groups,32] := by
    simp [TileShape.product, Parameters.block, Nat.mul_assoc]
  have hscale : TileShape.product [p.groups] = TileShape.product [p.groups,1] := by simp [TileShape.product]
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take, offsetCode, List.append]
  simp [run, step, evalExpr, evalComputeOp, FP.Structural.evalOp,
    hmatrix, hscale, backend, Fp4ToBf16.FPValue.backend,
    FP.Structural.numeric, FP.Structural.ofFloat, FP.Structural.toFloat, ComputeDType.eraseDType,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind, Option.map]
  try rfl

@[local simp] theorem productExecution_output (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (productExecution rest P p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j => P.mulBF16 (converted (laneNibble p s j))
        (scaleValue P (loadedScale p s (j.1.val/32)))) := by
  simp [productExecution, reshape_vector, reshape_rows32_any,
    TileShape.linearIndex, TileShape.product, Parameters.block, Nat.mul_assoc,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  congr 1
  funext j
  rw [reshape_rows32_any p.groups (p.groups*16*2) j]
  rw [reshape_vector (p.groups*16*2) [p.groups,32]]
  simp [TileShape.linearIndex, TileShape.product, linear32]
  try rfl

@[local simp] theorem productExecution_pid (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (productExecution rest P p s).val.regs .nat [] "pid" = some (fun _ => s.pids 0) := by simp [productExecution]
@[local simp] theorem productExecution_n_elements_out (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (productExecution rest P p s).val.regs .nat [] "n_elements_out" = some (fun _ => p.count*2) := by simp [productExecution]
@[local simp] theorem productExecution_mem (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (productExecution rest P p s).val.mem = s.mem :=
  (assignment_run_memory (backend rest P) _ _ _ (by rfl) (productExecution rest P p s).property).trans
    (scaleExecution_mem rest P p s)
attribute [local irreducible] productExecution

def storeExecution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest P) ((source p).surfaceBody.drop 43 |>.take 4)
      (productExecution rest P p s).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take, offsetCode, List.append]
  simp [run, step, evalExpr, evalComputeOp, FP.Structural.evalOp, store,
    backend, Fp4ToBf16.FPValue.backend, FP.Structural.numeric, FP.Structural.natLt,
    FP.Structural.ofFloat, FP.Structural.toFloat, ComputeDType.eraseDType,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind, Option.map]
  try rfl

/-- Full 47-statement candidate, including both parsed signed16 casts, every scale
load, powif callback, bf16 cast/select/mul, reshapes and masked store. -/
def execution (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.exec (backend rest P) (source p) s = some t} := by
  refine ⟨(storeExecution rest P p s).val, ?_⟩
  change run _
    (((source p).surfaceBody.take 28) ++
      ((source p).surfaceBody.drop 28 |>.take 4) ++
      ((source p).surfaceBody.drop 32 |>.take 1) ++
      ((source p).surfaceBody.drop 33 |>.take 1) ++
      ((source p).surfaceBody.drop 34 |>.take 5) ++
      ((source p).surfaceBody.drop 39 |>.take 4) ++
      ((source p).surfaceBody.drop 43 |>.take 4)) s = _
  simp only [run_append, (decodedPrefix rest P p s).property, decodedPrefix,
    (scaleLoadExecution rest P p s).property, (exponentExecution rest P p s).property,
    (powerExecution rest P p s).property, (scaleExecution rest P p s).property,
    (productExecution rest P p s).property, (storeExecution rest P p s).property, Option.bind_some]

#axiomsClean productExecution
#axiomsClean productExecution_output
#axiomsClean storeExecution
#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue

import bench.tritonbench_g.chunk_linear_attn.NativeMeaning
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable (decode : BitVec 32 → ℝ)

theorem matrix_expansion {m n : Nat} (a : Matrix Term m n) :
    mapMatrix decode a=(fun i => (a i).meaning decode) := rfl
 theorem transpose_expansion {α : Type} {m n : Nat} (a : Matrix α m n) :
    transpose a=(fun i => a (i.2.1,i.1,())) := rfl
 theorem score_expansion {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (scale : ℝ) (a : GradAcc p α) :
    gradScoreGradient M p scale a=(fun i => cast M (if i.2.1.val ≤ i.1.val then mul M (a.score i) (scalar M scale) else zero M)) := rfl

def mapOutput (a : OutputAcc p Term) : OutputAcc p ℝ := ⟨mapMatrix decode a.output,mapMatrix decode a.score⟩
 theorem map_outputNext (p : Parameters) (pid : PID) (m : Memory Term) (kb : Nat) (a : OutputAcc p Term) :
    mapOutput decode (outputNext algebra dotPrimitive p pid m kb a)=
      outputNext Real.algebra Real.dotPrimitive p pid (mapMemory decode m) kb (mapOutput decode a) := by
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapOutput,outputNext,mapMatrix,map_add,map_dot,queryTile,keyTile,forwardHistoryTile,map_load]
 theorem map_outputStates (p : Parameters) (pid : PID) (m : Memory Term) (n : Nat) :
    mapOutput decode (outputStates algebra dotPrimitive p pid m n)=
      outputStates Real.algebra Real.dotPrimitive p pid (mapMemory decode m) n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [matrix_expansion,transpose_expansion,score_expansion,outputStates,map_outputNext,ih]
 theorem map_outputResult (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (a : OutputAcc p Term) :
    mapMatrix decode (outputResult algebra dotPrimitive p pid m scale a)=
      outputResult Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale (mapOutput decode a) := by
  funext i
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapMatrix,outputResult,mapOutput,map_add,map_mul,map_scalar,map_cast,map_if,map_zero,map_dot,valueTile,map_load]
 theorem map_outputWrite (p : Parameters) (pid : PID) (a : Matrix Term p.BT p.BV) (m : Memory Term) :
    mapMemory decode (outputWrite algebra p pid a m)=
      outputWrite Real.algebra p pid (mapMatrix decode a) (mapMemory decode m) := by
  simp only [matrix_expansion,transpose_expansion,score_expansion,outputWrite,map_scatter,map_mk,map_cast,mapMatrix]
 theorem map_outputMemory (p : Parameters) (pid : PID) (input current : Memory Term) (scale : ℝ) :
    mapMemory decode (outputMemory algebra dotPrimitive p pid input current scale)=
      outputMemory Real.algebra Real.dotPrimitive p pid (mapMemory decode input) (mapMemory decode current) scale := by
  simp only [outputMemory,map_outputWrite,map_outputResult,map_outputStates]

def mapGrad (a : GradAcc p Term) : GradAcc p ℝ := ⟨mapMatrix decode a.query,mapMatrix decode a.key,mapMatrix decode a.score⟩
 theorem map_gradNext (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (vb : Nat) (a : GradAcc p Term) :
    mapGrad decode (gradNext algebra dotPrimitive p pid m scale vb a)=
      gradNext Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale vb (mapGrad decode a) := by
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapGrad,gradNext,mapMatrix,map_add,map_mul,map_scalar,map_dot,valueTile,gradHistory,gradHistoryGradient,transpose,map_load]
 theorem map_gradStates (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (n : Nat) :
    mapGrad decode (gradStates algebra dotPrimitive p pid m scale n)=
      gradStates Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [matrix_expansion,transpose_expansion,score_expansion,gradStates,map_gradNext,ih]
 theorem map_gradValue (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (vb : Nat) :
    mapMatrix decode (gradValue algebra dotPrimitive p pid m scale vb)=
      gradValue Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale vb := by
  funext i
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapMatrix,gradValue,gradScore,map_add,map_mul,map_scalar,map_cast,map_if,map_zero,map_dot,
    gradKey,gradQuery,gradHistoryGradient,valueTile,map_load]
 theorem map_gradQueryResult (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (a : GradAcc p Term) :
    mapMatrix decode (gradQueryResult algebra dotPrimitive p pid m scale a)=
      gradQueryResult Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale (mapGrad decode a) := by
  funext i
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapMatrix,mapGrad,gradQueryResult,gradScoreGradient,map_add,map_mul,map_scalar,map_cast,map_if,map_zero,map_dot,gradKey,map_load]
 theorem map_gradKeyResult (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (a : GradAcc p Term) :
    mapMatrix decode (gradKeyResult algebra dotPrimitive p pid m scale a)=
      gradKeyResult Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale (mapGrad decode a) := by
  funext i
  simp only [matrix_expansion,transpose_expansion,score_expansion,mapMatrix,mapGrad,gradKeyResult,gradScoreGradient,map_add,map_mul,map_scalar,map_cast,map_if,map_zero,map_dot,gradQuery,transpose,map_load]
 theorem map_gradValueWrite (p : Parameters) (pid : PID) (heads vb : Nat)
    (a : Matrix Term p.BT p.BV) (m : Memory Term) :
    mapMemory decode (gradValueWrite algebra p pid heads vb a m)=
      gradValueWrite Real.algebra p pid heads vb (mapMatrix decode a) (mapMemory decode m) := by
  simp only [matrix_expansion,transpose_expansion,score_expansion,gradValueWrite,map_scatter,map_mk,map_cast,mapMatrix]
 theorem map_gradKeyWrite (p : Parameters) (pid : PID) (r : RegionName)
    (a : Matrix Term p.BT p.BK) (m : Memory Term) :
    mapMemory decode (gradKeyWrite algebra p pid r a m)=
      gradKeyWrite Real.algebra p pid r (mapMatrix decode a) (mapMemory decode m) := by
  simp only [matrix_expansion,transpose_expansion,score_expansion,gradKeyWrite,map_scatter,map_mk,map_cast,mapMatrix]
 theorem map_gradientMemory (p : Parameters) (pid : PID) (heads : Nat) (input current : Memory Term) (scale : ℝ) :
    mapMemory decode (gradientMemory algebra dotPrimitive p pid heads input current scale)=
      gradientMemory Real.algebra Real.dotPrimitive p pid heads (mapMemory decode input) (mapMemory decode current) scale := by
  simp only [gradientMemory,map_gradKeyWrite,map_gradQueryResult,map_gradKeyResult,map_gradStates]
  congr 2
  apply map_range
  intro vb mem
  simp only [map_gradValueWrite,map_gradValue]

#axiomsClean map_outputMemory
#axiomsClean map_gradientMemory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms

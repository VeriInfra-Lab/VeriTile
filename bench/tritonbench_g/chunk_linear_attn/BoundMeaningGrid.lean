import bench.tritonbench_g.chunk_linear_attn.BoundMeaningOutput
import bench.tritonbench_g.chunk_linear_attn.NativeFP
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeBound
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open NativeTerms
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable (decode : BitVec 32 → ℝ) (word : BitVec 32)

 theorem map_stage (stage : Stage) (p : Parameters) (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory Term) :
    mapMemory decode word (stageMemory algebra dotPrimitive stage p scale seed final pid m)=
      stageMemory Real.algebra Real.dotPrimitive stage p (scaleMeaning decode word scale) seed final pid (mapMemory decode word m) := by
  cases stage <;> simp only [stageMemory,map_stateMemory,map_outputMemory,map_gradientMemory]
 theorem map_grid (stage : Stage) (p : Parameters) (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory Term) :
    mapMemory decode word (gridMemory algebra dotPrimitive stage p scale seed final schedule m)=
      gridMemory Real.algebra Real.dotPrimitive stage p (scaleMeaning decode word scale) seed final schedule (mapMemory decode word m) := by
  exact map_fold (mapMemory decode word) _ _ (fun pid mem => map_stage decode word stage p scale seed final pid mem) schedule m
 theorem map_four (f b : Parameters) (scale : ℝ) (seed final terminal : Bool)
    (sf so sb sg : List PID) (m : Memory Term) :
    mapMemory decode word (modelFour algebra dotPrimitive f b scale seed final terminal sf so sb sg m)=
      modelFour Real.algebra Real.dotPrimitive f b (scaleMeaning decode word scale) seed final terminal sf so sb sg (mapMemory decode word m) := by
  simp only [modelFour,map_grid]
 theorem map_sum (p : Parameters) (selection : LibrarySelection p) (m : Memory Term) (o : Fin (valueCount p)) :
    boundMeaning decode word (sumValue algebra p selection m o)=
      sumValue Real.algebra p selection (mapMemory decode word m) o := by
  unfold sumValue LibrarySum.evaluate
  apply LibrarySum.transport (boundMeaning decode word)
  · exact map_zero decode word
  · intros; rfl
  · intro i; rfl
 theorem map_host (p : Parameters) (selection : LibrarySelection p) (input current : Memory Term) :
    mapMemory decode word (hostSumMemory algebra p selection input current)=
      hostSumMemory Real.algebra p selection (mapMemory decode word input) (mapMemory decode word current) := by
  simp only [hostSumMemory,map_scatter,map_mk,map_cast,map_sum]
 theorem map_complete (c : Real.Call) (m : Memory Term) :
    mapMemory decode word (completeMemory algebra dotPrimitive c.forward c.backward c.scale c.seed c.final c.terminal c.selection m)=
      ({c with scale:=scaleMeaning decode word c.scale} : Real.Call).result (mapMemory decode word m) := by
  rw [Real.Call.result_is_complete]
  simp only [completeMemory,map_host,map_four,Real.Call.forward,Real.Call.backward]

def tagged (c : Real.Call) : Real.Call := {c with scale:=1}
def rawMemory (c : Real.Call) (bits : RegionName → Nat → BitVec 32) : Memory Term :=
  completeMemory algebra dotPrimitive c.forward c.backward 1 c.seed c.final c.terminal c.selection (inputMemory bits)
/-- Binding acts on the separate scale tag before instruction lowering. Thus
actual FP32 -0, finite underflowed scalars and every supplied word survive. -/
def graph (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) : Graph :=
  lowerAt c.backward r (bindScale word ((rawMemory c bits r o).read .real))

theorem graph_real (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    (graph word c bits r o).eval (realISA decode)=
      (({c with scale:=decode word} : Real.Call).result (decodedMemory decode bits) r o).read .real := by
  have hm := congrArg (fun m : Memory ℝ => (m r o).read .real) (map_complete decode word (tagged c) (inputMemory bits))
  unfold graph lowerAt
  split <;> simp only [lowerKey_real,lower_real]
  all_goals simpa only [tagged,scaleMeaning,if_pos rfl] using hm

theorem graph_scalar (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    ScalarGraph (graph word c bits r o) := by
  unfold graph lowerAt
  split
  · exact lowerKey_scalar _ _
  · exact lower_scalar _

theorem graph_fp (I : ISA (BitVec 32)) (p : FP32Profile) (h : PrimitiveLeaves I p)
    (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    (graph word c bits r o).eval I=(graph word c bits r o).eval p.isa :=
  scalar_transport I p h _ (graph_scalar word c bits r o)

theorem bound_scale_word : bindScale word (.literal 1)=.bits word := by simp [bindScale]
theorem bound_zero : bindScale word (.literal 0)=.literal 0 := by simp [bindScale]

#axiomsClean map_complete
#axiomsClean graph_real
#axiomsClean graph_fp
#axiomsClean bound_scale_word
#axiomsClean bound_zero
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeBound

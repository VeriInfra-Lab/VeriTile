import bench.repairs.tritonbench_g.chunk_linear_attn.NativeMeaningOutput
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable (decode : BitVec 32 → ℝ)

 theorem map_stage (stage : Stage) (p : Parameters) (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory Term) :
    mapMemory decode (stageMemory algebra dotPrimitive stage p scale seed final pid m)=
      stageMemory Real.algebra Real.dotPrimitive stage p scale seed final pid (mapMemory decode m) := by
  cases stage <;> simp only [stageMemory,map_stateMemory,map_outputMemory,map_gradientMemory]
 theorem map_grid (stage : Stage) (p : Parameters) (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory Term) :
    mapMemory decode (gridMemory algebra dotPrimitive stage p scale seed final schedule m)=
      gridMemory Real.algebra Real.dotPrimitive stage p scale seed final schedule (mapMemory decode m) := by
  exact map_fold (mapMemory decode) _ _ (fun pid mem => map_stage decode stage p scale seed final pid mem) schedule m
 theorem map_four (f b : Parameters) (scale : ℝ) (seed final terminal : Bool)
    (sf so sb sg : List PID) (m : Memory Term) :
    mapMemory decode (modelFour algebra dotPrimitive f b scale seed final terminal sf so sb sg m)=
      modelFour Real.algebra Real.dotPrimitive f b scale seed final terminal sf so sb sg (mapMemory decode m) := by
  simp only [modelFour,map_grid]
 theorem map_sum (p : Parameters) (selection : LibrarySelection p) (m : Memory Term) (o : Fin (valueCount p)) :
    (sumValue algebra p selection m o).meaning decode=
      sumValue Real.algebra p selection (mapMemory decode m) o := by
  unfold sumValue LibrarySum.evaluate
  apply LibrarySum.transport (Term.meaning decode)
  · rfl
  · intros; rfl
  · intro i; rfl
 theorem map_host (p : Parameters) (selection : LibrarySelection p) (input current : Memory Term) :
    mapMemory decode (hostSumMemory algebra p selection input current)=
      hostSumMemory Real.algebra p selection (mapMemory decode input) (mapMemory decode current) := by
  simp only [hostSumMemory,map_scatter,map_mk,map_cast,map_sum]
 theorem map_complete (c : Real.Call) (m : Memory Term) :
    mapMemory decode (completeMemory algebra dotPrimitive c.forward c.backward c.scale c.seed c.final c.terminal c.selection m)=
      c.result (mapMemory decode m) := by
  rw [c.result_is_complete]
  simp only [completeMemory,map_host,map_four]

def inputMemory (bits : RegionName → Nat → BitVec 32) : Memory Term :=
  fun r o => .mk .real (.bits (bits r o))
def decodedMemory (bits : RegionName → Nat → BitVec 32) : Memory ℝ :=
  mapMemory decode (inputMemory bits)
def rawMemory (c : Real.Call) (bits : RegionName → Nat → BitVec 32) : Memory Term :=
  completeMemory algebra dotPrimitive c.forward c.backward c.scale c.seed c.final c.terminal c.selection (inputMemory bits)
def lowerAt (p : Parameters) (r : RegionName) (e : Term) : Graph :=
  if r="keyGradient" then lowerKey (decide (valueBlocks p=1)) e else lower e
def graph (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) : Graph :=
  lowerAt c.backward r ((rawMemory c bits r o).read .real)

/-- Complete semantic connection: no output equation is assumed. All current-memory
reads, every grid program, all casts, and each library scalar add are mapped. -/
theorem graph_real (c : Real.Call) (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    (graph c bits r o).eval (realISA decode)=(c.result (decodedMemory decode bits) r o).read .real := by
  have hm := congrArg (fun m : Memory ℝ => (m r o).read .real) (map_complete decode c (inputMemory bits))
  unfold graph lowerAt
  split <;> simp only [lowerKey_real,lower_real]
  all_goals exact hm

/-- Numeric interpretation is delegated only to individual scalar ISA operations.
The tensor recurrence, dot length, reduction coverage, and VJP are proved above. -/
theorem graph_transport {α : Type} (f : α → ℝ) (I : ISA α)
    (ports : LocalTransport f I (realISA decode)) (c : Real.Call)
    (bits : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat) :
    f ((graph c bits r o).eval I)=(c.result (decodedMemory decode bits) r o).read .real := by
  rw [transport f I (realISA decode) ports,graph_real]

#axiomsClean map_complete
#axiomsClean graph_real
#axiomsClean graph_transport
#specNonCircular graph avoiding [FP.Structural.exec,runGrid,executeFour,Real.Call.target]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms

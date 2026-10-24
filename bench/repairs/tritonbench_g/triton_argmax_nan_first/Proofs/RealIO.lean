import bench.repairs.tritonbench_g.triton_argmax_nan_first.IO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- The mathematical argument has exactly the logical input extent. -/
def input (c : GlobalConfig) (vals : ChannelValues ℝ [⟨.real,[c.size],X⟩]) (i : Nat) : ℝ :=
  if h : i < c.size then vals 0 (⟨i,h⟩,PUnit.unit) else 0

def mathematical (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (index64 : Bool)
    (vals : ChannelValues ℝ [⟨.real,[c.size],X⟩]) :
    ChannelValues ℝ [⟨.int,[1],Region.cast O⟩] :=
  Fin.cases (fun _ => Int.ofNat
    (Math.argmax c.size coverage.sizePositive (fun i => vals 0 (i,PUnit.unit))).val) (fun i => Fin.elim0 i)

theorem grid_execution (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I : Region .int) (c : GlobalConfig) (index64 : Bool)
    (jobs : List Nat) (s t : State Math.Value)
    (ran : TypedReal.GridRun (TypedReal.backend rest)
      (TypedScratch.firstKernel kind X V I c.size c.block 0 index64) jobs s t) :
    runSchedule (jobs.map (IO.atProgram (TypedReal.backend rest)
      (TypedScratch.firstKernel kind X V I c.size c.block 0 index64))) s = some t := by
  induction ran with
  | nil => rfl
  | cons first rest ih =>
    simp only [List.map_cons, runSchedule, Invocation.run, IO.atProgram_initial]
    change (exec (TypedReal.backend _) (TypedScratch.firstKernel kind X V I c.size c.block 0 index64)
      (TypedReal.atProgram _ _)).bind _ = _
    rw [first]
    exact ih

#axiomsClean grid_execution
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealIO

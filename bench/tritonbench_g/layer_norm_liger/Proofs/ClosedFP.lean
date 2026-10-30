import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardSoftware
import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardSoftware
namespace VeriTile.Bench.Repairs.LayerNormLiger.ClosedFP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (softwareSum withSum)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def forwardReference [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Forward.Parameters) (s : State (FP.Value .fp32)) : State (FP.Value .fp32) :=
  let t := (s.write p.mean (Forward.meanOffset p s) (.mk .fp32 (Forward.bitMean cfg literal p s))).write
    p.rstd (Forward.rstdOffset p s) (.mk .fp32 (Forward.bitRstd cfg literal rsqrt32 p s))
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val<p.columns then
    t.write p.y (Forward.outputOffset p s i) (.mk .fp32 (Forward.bitY cfg literal rsqrt32 p s i)) else t) t

def finish (p : Backward.Parameters) (pid : Nat) (a : Backward.Acc (FP.Value .fp32) p) : State (FP.Value .fp32) :=
  let t := (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val<p.columns then
    t.write p.dw (pid*p.dwStride+i.1.val) (.mk .fp32 (a.dw i)) else t) a.state
  (TileShape.allIndices [p.block]).foldl (fun t i => if i.1.val<p.columns then
    t.write p.db (pid*p.dbStride+i.1.val) (.mk .fp32 (a.db i)) else t) t

def backwardReference [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (p : Backward.Parameters) (s : State (FP.Value .fp32)) :=
  finish p (s.pids 0) (foldRange (Backward.bitUpdate cfg literal p) (Backward.start p s) (Backward.stop p s) 1
    ⟨s,fun _ => FP.Value.zero .fp32,fun _ => FP.Value.zero .fp32⟩)

theorem forward_reference [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Forward.Parameters) (s : State (FP.Value .fp32)) :
    Forward.reference (Forward.software rest cfg literal rsqrt32) (softwareSum cfg) p s =
      forwardReference cfg literal rsqrt32 p s := by
  obtain ⟨hm,hr,hy⟩ := Forward.software_values rest cfg literal rsqrt32 p s
  unfold Forward.reference forwardReference
  simp only [hm,hr,hy]

theorem backward_reference [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Backward.Parameters) (s : State (FP.Value .fp32)) :
    Backward.reference (Forward.software rest cfg literal rsqrt32) (softwareSum cfg) p s =
      backwardReference cfg literal p s := by
  unfold Backward.reference
  rw [Backward.accumulated_software]
  simp [Backward.finish, Backward.stored, finish, backwardReference, Forward.software,
    VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem forward_exec [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Forward.Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (Forward.software rest cfg literal rsqrt32) (softwareSum cfg))
      (Forward.kernel p) s=some t ∧ t.mem=(forwardReference cfg literal rsqrt32 p s).mem := by
  obtain ⟨t,ht,hm⟩ := Forward.complete_run (Forward.software rest cfg literal rsqrt32) (softwareSum cfg) p s
  exact ⟨t,ht,by simpa only [forward_reference] using hm⟩

theorem backward_exec [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Backward.Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (Forward.software rest cfg literal rsqrt32) (softwareSum cfg))
      (Backward.kernel p) s=some t ∧ t.mem=(backwardReference cfg literal p s).mem := by
  obtain ⟨t,ht,hm⟩ := Backward.complete_run (Forward.software rest cfg literal rsqrt32) (softwareSum cfg) p s
  exact ⟨t,ht,by simpa only [backward_reference] using hm⟩

#axiomsClean forward_reference
#axiomsClean backward_reference
#axiomsClean forward_exec
#axiomsClean backward_exec
end
end VeriTile.Bench.Repairs.LayerNormLiger.ClosedFP

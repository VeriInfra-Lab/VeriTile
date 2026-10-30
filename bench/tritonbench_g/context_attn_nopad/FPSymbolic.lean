import bench.tritonbench_g.context_attn_nopad.FPBackend

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPSymbolic
open VeriTile.Triton FP.Structural
open FP.ProfiledRewrite (engine)
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev program (p : Layout) (r : Program.Regions) (scale : ℝ) :=
  Program.kernel p r .fp32 Bool.false scale

abbrev parts (p : Layout) (r : Program.Regions) (scale : ℝ) :=
  Program.parts p r Bool.false scale

theorem source_split (p : Layout) (r : Program.Regions) (scale : ℝ) :
    (program p r scale).surfaceBody =
      (parts p r scale).before ++
      [.forRangeInt "start_n" (parts p r scale).start
        (parts p r scale).stop (parts p r scale).stride
        (parts p r scale).body] ++ (parts p r scale).after := rfl


end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPSymbolic

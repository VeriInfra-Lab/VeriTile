import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.SurfaceSource
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def code (h : Call) := (Surface.source h).surfaceBody
def prelude (h : Call) := (code h).take 3
def active (h : Call) := ((Source.conditional? (code h)
  (fun b => (Source.intRange? b "k").isSome)).map (·.yes)).getD []
def setup (h : Call) := ((Source.intRange? (active h) "k").map (·.before)).getD []
def body (h : Call) := ((Source.intRange? (active h) "k").map (·.body)).getD []
def tail (h : Call) := ((Source.intRange? (active h) "k").map (·.after)).getD []
def guard : Op .bool [] := .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))
def loop (h : Call) : ComputeStmt := .forRangeInt "k" (.constInt 0) (.constInt h.K) (.constNat 16384) (body h)

theorem complete_code (h : Call) : code h=prelude h++[.ifThen guard (active h)] := by
  cases eq : h.storage <;> simp only [code,prelude,active,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.source,eq]
  all_goals rfl

theorem complete_active (h : Call) : active h=setup h++loop h::tail h := by
  cases eq : h.storage <;> simp only [code,prelude,active,setup,body,loop,tail,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.source,eq]
  all_goals rfl

#axiomsClean complete_code
#axiomsClean complete_active
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPProgram

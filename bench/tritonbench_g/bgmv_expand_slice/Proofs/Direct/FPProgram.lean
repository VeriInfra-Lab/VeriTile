import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPProgram
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def primarySource (h : Call) : ComputeKernel :=
  Kernels.original .fp32 .fp16 Bool.true .fp32
    "X" "W" "Y" h.N h.K ⟨"Indices"⟩ h.K 1 (h.N*h.K) h.K 1
    h.columns 1 h.slice 256 h.BK h.N h.even h.add

def code (h : Call) := (primarySource h).surfaceBody
def prelude (h : Call) := (code h).take 3
def active (h : Call) := ((Source.conditional? (code h)
  (fun b => (Source.dynamicRange? b "n").isSome)).map (·.yes)).getD []
def setup (h : Call) := ((Source.dynamicRange? (active h) "n").map (·.before)).getD []
def body (h : Call) := ((Source.dynamicRange? (active h) "n").map (·.body)).getD []
def guard : Op .bool [] := .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))
def loop (h : Call) : ComputeStmt :=
  .forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat 256) (body h)

theorem common_source (h : Call) (primary : Kernels.Primary h) : Kernels.source h=primarySource h := by
  rcases primary with ⟨hi,hw,ho⟩
  simp only [Kernels.source,VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.directSource,primarySource,hi,hw,ho,Kernels.hostCast,
    VeriTile.Bench.TritonBenchG.BgmvExpandSlice.TypedSource.hostCast]

theorem complete_code (h : Call) : code h=prelude h++[.ifThen guard (active h)] := by rfl

theorem complete_active (h : Call) : active h=setup h++[loop h] := by rfl

#axiomsClean common_source
#axiomsClean complete_code
#axiomsClean complete_active
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPProgram

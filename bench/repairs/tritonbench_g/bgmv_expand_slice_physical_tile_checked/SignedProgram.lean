import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SurfaceSource
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.SourceView
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Program
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
/-- Proof view connected to the full raw body by SourceView.source_run/source_safe. -/
def code (h : Call) := SourceView.code h
def prelude (h : Call) := (code h).take 3
def active (h : Call) := SourceView.active h
def setup (h : Call) := (active h).take 7
def body (h : Call) : List ComputeStmt :=
  match (active h).getD 7 (.effectMarker "absent") with
  | .forRangeInt _ _ _ _ body => body
  | _ => []
def guard : Op .bool [] := .boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))
def loop (h : Call) : ComputeStmt :=
  .forRangeInt "n" (.constInt 0) (.ref .int [] "split_n_length") (.constNat 256) (body h)
theorem complete_code (h : Call) : code h=prelude h++[.ifThen guard (active h)] := by rfl
theorem complete_active (h : Call) : active h=setup h++[loop h] := by rfl
theorem signed_length (h : Call) (positive : 0<h.N) :
    ((h.N : Int)+h.N-1)/h.N=1 := by
  have hs : (h.N : Int)+h.N-1=((h.N+h.N-1 : Nat) : Int) := by omega
  rw [hs,←Int.natCast_ediv,Kernels.split_length h positive]
  rfl
#axiomsClean complete_code
#axiomsClean complete_active
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Program

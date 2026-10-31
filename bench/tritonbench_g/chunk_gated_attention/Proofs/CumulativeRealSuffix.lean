import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealPointers
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumExecution
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG
open ChunkGatedAttention
open ChunkCumsumVector.RealStep (pointer inputTile chunkPrefix)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def tileLayout (p : Parameters) (initial : BlockState) : RealCumProgram.Layout :=
  ⟨tileBase p initial,p.timeStride,p.featureStride,p.BT,p.BS,p.BT,p.BS⟩
def localLaunch : RealCumProgram.Launch := ⟨0,1⟩

theorem prepared_input_pointer (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.regs .blockPtr [p.BT,p.BS] "p_s"=
      some (pointer (tileLayout p initial) localLaunch s 0) := by
  rw [active_input_pointer]
  simp [pointer,tileLayout,localLaunch]

theorem prepared_output_pointer (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.regs .blockPtr [p.BT,p.BS] "p_o"=
      some (pointer (tileLayout p initial) localLaunch o 0) := by
  rw [active_output_pointer]
  simp [pointer,tileLayout,localLaunch]

def suffixPrepared (p : Parameters) (s o : RegionName) (initial : BlockState) : BlockState :=
  let q := tileLayout p initial
  let a := initial.readMem s
  let padding := initial.undef s
  let u := (activePreludeExecution p s o initial).val
  let u := u.setReg "b_s" .real [p.BT,p.BS] (inputTile q localLaunch a padding 0)
  u.setReg "b_o" .real [p.BT,p.BS] (RealCumExecution.outputTile q localLaunch a padding 0)

def activeResult (p : Parameters) (s o : RegionName) (initial : BlockState) : BlockState :=
  RealCumExecution.scatter (tileLayout p initial) localLaunch o
    (initial.readMem s) (initial.undef s) 0 (suffixPrepared p s o initial)

theorem suffix_computes (p : Parameters) (s o : RegionName) (initial : BlockState) :
    stepStmts (suffix p s o) (activePreludeExecution p s o initial).val=
      some (activeResult p s o initial) := by
  let q := tileLayout p initial
  rw [suffix_ast]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ChunkCumsumVector.RealStep.load_eval q localLaunch s (initial.readMem s)
      (initial.undef s) 0 _ (by intro off; rw [active_undef])
      (by intro off; simp only [BlockState.readMem,active_memory])
      (prepared_input_pointer p s o initial)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (RealCumExecution.dot_eval q localLaunch
    (initial.readMem s) (initial.undef s) 0 _ (by simpa [q,tileLayout] using active_triangular p s o initial)
    (by simp [q,tileLayout])))]
  erw [stepStmts.cons_some (RealCumExecution.store_eval q localLaunch o
    (initial.readMem s) (initial.undef s) 0 _
    (by simpa [q,tileLayout] using prepared_output_pointer p s o initial)
    (by simp [q,tileLayout])),stepStmts.nil]
  rfl

theorem all_active (p : Parameters) (initial : BlockState) (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.tileActive (tileLayout p initial) localLaunch 0 i := by
  exact ⟨by simpa [RealCumExecution.tileActive,ChunkCumsumVector.RealStep.active,tileLayout,localLaunch]
      using i.1.isLt,
    by simpa [RealCumExecution.tileActive,ChunkCumsumVector.RealStep.active,tileLayout,localLaunch]
      using i.2.1.isLt⟩

theorem active_readback (p : Parameters) (s o : RegionName) (initial : BlockState)
    (hinj : RealCumExecution.NoCollision (tileLayout p initial) localLaunch 0)
    (i : TileIndex [p.BT,p.BS]) :
    (activeResult p s o initial).readMem o
      (RealCumExecution.offset (tileLayout p initial) localLaunch 0 i)=
      RealCumExecution.expected (tileLayout p initial) localLaunch 0 (initial.readMem s) i := by
  let q := tileLayout p initial
  have ha := all_active p initial i
  have hr := BlockState.scatter_readback_prop_masked_nd_of_true (region:=o)
    (suffixPrepared p s o initial)
    (RealCumExecution.offset q localLaunch 0)
    (fun j=>chunkPrefix q localLaunch (initial.readMem s) (initial.undef s) 0 j.1 j.2.1)
    (RealCumExecution.tileActive q localLaunch 0) i ha (fun j hj he=>hinj j i hj ha he)
  exact hr.trans (RealCumExecution.prefix_active q localLaunch (initial.readMem s)
    (initial.undef s) 0 i ha)

theorem active_frame (p : Parameters) (s o : RegionName) (initial : BlockState) :
    BlockState.WriteWithin (RealCumExecution.footprint (tileLayout p initial) localLaunch 0 o)
      initial (activeResult p s o initial) := by
  intro region off hn
  have h := RealCumExecution.scatter_frame (tileLayout p initial) localLaunch o
    (initial.readMem s) (initial.undef s) 0 (suffixPrepared p s o initial) region off hn
  exact (show initial.mem region off=(suffixPrepared p s o initial).mem region off from by
    simp [suffixPrepared,active_memory]).trans h

#axiomsClean suffix_computes
#axiomsClean active_readback
#axiomsClean active_frame
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real

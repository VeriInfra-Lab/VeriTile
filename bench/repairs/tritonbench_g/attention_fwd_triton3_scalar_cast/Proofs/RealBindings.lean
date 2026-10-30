import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.RealIO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxHeartbeats 2000000

theorem source_projection (c : Call) :
    (selectedSource c).kernel.toAlgKernel=c.source := by
  simp only [selectedSource,Kernels.source_eq_inherited,AttentionFwdTriton3.Kernels.real_erasure]

theorem input_binding (c : Call) (v : Inputs c) (s : BlockState)
    (bound : (programIO c).InputBinding s v) (idx : GridIndex c.grid)
    (i : Fin 6) (j : TileIndex (inputShape c i))
    (active : c.io.inputMask i (c.tile idx) (c.hz idx) 0 j) :
    (s.withGridIndex idx).readMem (c.io.inputRegion i)
      (c.io.inputWindow i (c.tile idx) (c.hz idx) 0 j)=
      v i (inputCoordinates c i (idx,j)) := by
  have h := bound i (inputCoordinates c i (idx,j))
  simp only [programIO,AlgorithmProgramIO.InputBinding,AlgorithmMemoryIO.InputBinding,
    Equiv.symm_apply_apply,ChanTy.read] at h
  have hr : c.io.inputRegion i=inputRegion i := by fin_cases i <;> rfl
  exact hr ▸ h active

theorem pins (c : Call) (v : Inputs c) (s : BlockState)
    (bound : (programIO c).InputBinding s v) (idx : GridIndex c.grid) :
    RealIO.Pins c.io (s.withGridIndex idx) (localInputs c v idx) := by
  constructor
  · intro r d active
    exact input_binding c v s bound idx 0 (r,d,PUnit.unit) active
  · intro k d
    exact input_binding c v s bound idx 1 (k,d,PUnit.unit) trivial
  · intro k d
    exact input_binding c v s bound idx 2 (k,d,PUnit.unit) trivial
  · intro read r
    exact input_binding c v s bound idx 3 (r,PUnit.unit) read
  · intro read r
    exact input_binding c v s bound idx 4 (r,PUnit.unit) read
  · intro read r d
    exact input_binding c v s bound idx 5 (r,d,PUnit.unit) read

theorem input_bound (c : Call) (v : Inputs c) (s : BlockState) (bounds : RegionBounds)
    (bounded : (programIO c).InBounds bounds s v) (idx : GridIndex c.grid)
    (i : Fin 6) (j : TileIndex (inputShape c i))
    (active : c.io.inputMask i (c.tile idx) (c.hz idx) 0 j) :
    c.io.inputWindow i (c.tile idx) (c.hz idx) 0 j<bounds (c.io.inputRegion i) := by
  have h := bounded.1 i (inputCoordinates c i (idx,j))
  simp only [programIO,Equiv.symm_apply_apply] at h
  have hr : c.io.inputRegion i=inputRegion i := by fin_cases i <;> rfl
  exact hr ▸ h active

theorem output_bound (c : Call) (v : Inputs c) (s : BlockState) (bounds : RegionBounds)
    (bounded : (programIO c).InBounds bounds s v) (idx : GridIndex c.grid)
    (o : Fin 3) (j : TileIndex (outputShape c o)) (active : c.io.outputMask o) :
    c.io.outputWindowFor o (c.tile idx) (c.hz idx) 0 j<bounds (c.io.outputRegion o) := by
  have h := bounded.2.1 o (outputCoordinates c o (idx,j))
  simp only [programIO,Equiv.symm_apply_apply] at h
  have hr : c.io.outputRegion o=outputRegion o := by fin_cases o <;> rfl
  exact hr ▸ h active

theorem windows (c : Call) (v : Inputs c) (s : BlockState) (bounds : RegionBounds)
    (bounded : (programIO c).InBounds bounds s v) (idx : GridIndex c.grid) :
    RealIO.Bounds c.io (s.withGridIndex idx) bounds := by
  constructor
  · intro j active
    exact input_bound c v s bounds bounded idx 0 j active
  · intro j
    exact input_bound c v s bounds bounded idx 1 j trivial
  · intro j
    exact input_bound c v s bounds bounded idx 2 j trivial
  · intro j
    exact output_bound c v s bounds bounded idx 0 j trivial
  · intro active j
    rcases active with read|write
    · exact input_bound c v s bounds bounded idx 4 j read
    · exact output_bound c v s bounds bounded idx 1 j write
  · intro j
    exact output_bound c v s bounds bounded idx 2 j trivial

theorem safe (c : Call) (valid : Valid c) (v : Inputs c) (s : BlockState)
    (bounds : RegionBounds) (bounded : (programIO c).InBounds bounds s v)
    (idx : GridIndex c.grid) : c.source.TraceSafe bounds (s.withGridIndex idx) := by
  exact RealSource.safe c.BM c.BN c.D c.Z c.finish c.params Bool.false _ bounds
    valid.blockN (evenN c) (RealAddressBounds.trace_bounds c.BM c.BN c.D c.finish
      c.params _ bounds valid.blockN (evenN c)
      (RealContract.address_bounds c.BM c.BN c.D c.Z c.finish c.params Bool.false _ bounds
        (windows c v s bounds bounded idx)))

#axiomsClean pins
#axiomsClean safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

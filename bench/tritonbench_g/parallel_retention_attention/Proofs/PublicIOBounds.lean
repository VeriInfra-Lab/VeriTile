import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicIOLayout
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessForward
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBackward

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (withDot)
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type}

@[simp] theorem forward_pids (s : Shape) (tile : Tile) (state : State (Expr α)) (ax : Nat) :
    ((forwardInvocation α s tile).initial state).pids ax=programIds s tile ax := rfl

@[simp] theorem backward_pids (s : Shape) (tile : Tile) (state : State (Expr α)) (ax : Nat) :
    ((backwardInvocation α s tile).initial state).pids ax=programIds s tile ax := rfl

theorem backward_source (s : Shape) (tile : Tile) (valid : ProgramDomain s 64 tile) :
    (backwardInvocation α s tile).kernel=FPBackward.source (backwardConfig s tile) "dq" := by
  have decoded := grid_decode s 64 tile valid
  simp only [backwardInvocation,backwardKernel,decodedBackwardConfig,programIds,
    decoded.2.1,decoded.2.2,backwardConfig]

theorem forward_bounds (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (xs : ChannelValues α (forwardInputs s))
    (bounds : RegionName → Nat) (bounded : (forwardIO a p s tile).InBounds bounds xs)
    (state : State (Expr α)) :
    PublicAccess.ForwardBounds bounds (config s 128) ((forwardInvocation α s tile).initial state) := by
  have decoded := grid_decode s 128 tile valid
  constructor
  · intro i j hi hj
    have h := bounded.2.1 (0 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DK tile.bh i j<bounds "q" at h
    change tile.bh*(s.T*s.DK)+i*s.DK+j*1<bounds "q"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (1 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨j,hj⟩,⟨i,hi⟩,PUnit.unit) trivial
    change dense s.T s.DK tile.bh j i<bounds "k" at h
    change tile.bh*(s.T*s.DK)+i*1+j*s.DK<bounds "k"
    simpa [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (2 : Fin 3) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DV tile.bh i j<bounds "v" at h
    change tile.bh*(s.T*s.DV)+i*s.DV+j*1<bounds "v"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i hi hj
    change tile.chunk*128+i.1.val<s.T at hi
    change ProgramDomain.gridX s tile%blocks s.DV s.BV*s.BV+i.2.1.val<s.DV at hj
    rw [decoded.2.2] at hj
    change (tile.bh+s.B*s.H*(ProgramDomain.gridX s tile/blocks s.DV s.BV))*(s.T*s.DV)+
      (tile.chunk*128+i.1.val)*s.DV+(ProgramDomain.gridX s tile%blocks s.DV s.BV*s.BV+i.2.1.val)*1<bounds "o"
    rw [decoded.2.1,decoded.2.2]
    have h := bounded.2.2.1 (0 : Fin 1) i ⟨hi,hj⟩
    simpa only [forwardIO,forwardAddress,tileOffset,Nat.mul_one] using h

theorem backward_query_bounds (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (bounds : RegionName → Nat) (bounded : (backwardIO a p s tile).InBounds bounds xs) :
    PublicQueryAccess.Bounds bounds (backwardConfig s tile) "dq" := by
  constructor
  · intro i j hi hj
    have h := bounded.2.1 (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DV tile.bh i j<bounds "g" at h
    change tile.bh*(s.T*s.DV)+i*s.DV+j*1<bounds "g"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DK tile.bh i j<bounds "k" at h
    change tile.bh*(s.T*s.DK)+i*s.DK+j*1<bounds "k"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨j,hj⟩,⟨i,hi⟩,PUnit.unit) trivial
    change dense s.T s.DV tile.bh j i<bounds "v" at h
    change tile.bh*(s.T*s.DV)+i*1+j*s.DV<bounds "v"
    simpa [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using h
  · intro i hi hj
    have h := bounded.2.2.1 (0 : Fin 3) i ⟨hi,hj⟩
    simpa only [backwardIO,backwardWrite,backwardOutputs,Fin.cases_zero,
      keyAddress,tileOffset,backwardConfig,config,Nat.mul_one] using h

theorem backward_key_value_bounds (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (bounds : RegionName → Nat) (bounded : (backwardIO a p s tile).InBounds bounds xs) :
    PublicBackwardAccess.Bounds bounds (backwardConfig s tile) := by
  constructor
  · intro i j hi hj
    have h := bounded.2.1 (0 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨j,hj⟩,⟨i,hi⟩,PUnit.unit) trivial
    change dense s.T s.DK tile.bh j i<bounds "q" at h
    change tile.bh*(s.T*s.DK)+i*1+j*s.DK<bounds "q"
    simpa [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (3 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨j,hj⟩,⟨i,hi⟩,PUnit.unit) trivial
    change dense s.T s.DV tile.bh j i<bounds "g" at h
    change tile.bh*(s.T*s.DV)+i*1+j*s.DV<bounds "g"
    simpa [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (1 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DK tile.bh i j<bounds "k" at h
    change tile.bh*(s.T*s.DK)+i*s.DK+j*1<bounds "k"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i j hi hj
    have h := bounded.2.1 (2 : Fin 4) (⟨tile.bh,valid.bh⟩,⟨i,hi⟩,⟨j,hj⟩,PUnit.unit) trivial
    change dense s.T s.DV tile.bh i j<bounds "v" at h
    change tile.bh*(s.T*s.DV)+i*s.DV+j*1<bounds "v"
    simpa only [dense,Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using h
  · intro i hi hj
    have h := bounded.2.2.1 (1 : Fin 3) i ⟨hi,hj⟩
    simpa only [backwardIO,backwardWrite,backwardOutputs,Fin.cases_succ,Fin.cases_zero,
      keyAddress,tileOffset,backwardConfig,config,Nat.mul_one] using h
  · intro i hi hj
    have h := bounded.2.2.1 (2 : Fin 3) i ⟨hi,hj⟩
    simpa only [backwardIO,backwardWrite,backwardOutputs,Fin.cases_succ,Fin.cases_zero,
      valueAddress,tileOffset,backwardConfig,config,Nat.mul_one] using h

theorem forward_schedule_safe (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (xs : ChannelValues α (forwardInputs s))
    (bounds : RegionName → Nat) (bounded : (forwardIO a p s tile).InBounds bounds xs)
    (state : State (Expr α)) : ScheduleSafe bounds ((forwardIO a p s tile).schedule xs) state := by
  change ScheduleSafe bounds [forwardInvocation α s tile] state
  rw [scheduleSafe_singleton]
  exact PublicAccess.forward_safe algebra dots bounds (config s 128) _
    (forward_bounds a p s tile valid xs bounds bounded state)

theorem backward_schedule_safe (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (bounds : RegionName → Nat) (bounded : (backwardIO a p s tile).InBounds bounds xs)
    (state : State (Expr α)) : ScheduleSafe bounds ((backwardIO a p s tile).schedule xs) state := by
  change ScheduleSafe bounds [backwardInvocation α s tile] state
  rw [scheduleSafe_singleton]
  change TraceSafe (withDot algebra dots) bounds (backwardInvocation α s tile).kernel _
  rw [backward_source s tile valid]
  exact PublicBackwardAccess.backward_safe algebra dots bounds (backwardConfig s tile) "dq" _
    (backward_query_bounds a p s tile valid xs bounds bounded)
    (backward_key_value_bounds a p s tile valid xs bounds bounded)

#axiomsClean forward_schedule_safe
#axiomsClean backward_schedule_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

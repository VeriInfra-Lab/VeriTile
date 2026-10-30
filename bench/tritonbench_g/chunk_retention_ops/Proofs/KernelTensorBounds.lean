import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelTensor
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
namespace G
export VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry
  (StatePID OutputPID GradientPID history_bound final_bound output_bound gradient_bound partial_bound)
end G
noncomputable section

theorem history_bound (v : View c role) (hr : role=.history ∨ role=.dhistory) (pid : PID)
    (hp : G.StatePID (parameters c) pid) (n : Nat) (hn : n<c.chunks) (i : TileIndex [c.BK,c.BV])
    (hi : hActive (parameters c) pid i) : hOffset (parameters c) pid n i<v.tensor.metadata.numel := by
  have h:=G.history_bound (parameters c) pid n i hp hn hi
  rcases hr with rfl|rfl <;>
    simpa only [VeriTile.Torch.TensorMetadata.numel,v.shape,Role.shape,List.prod_cons,List.prod_nil,Nat.mul_one,parameters,Nat.mul_assoc] using h

theorem state_bound (v : View c role) (hr : role=.finalState ∨ role=.initialGradient) (pid : PID)
    (hp : G.StatePID (parameters c) pid) (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i) :
    fOffset (parameters c) pid i<v.tensor.metadata.numel := by
  have h:=G.final_bound (parameters c) pid i hp hi
  rcases hr with rfl|rfl <;>
    simpa only [VeriTile.Torch.TensorMetadata.numel,v.shape,Role.shape,List.prod_cons,List.prod_nil,Nat.mul_one,parameters,Nat.mul_assoc] using h

theorem output_bound (v : View c .output) (pid : PID) (hp : G.OutputPID (parameters c) pid)
    (i : TileIndex [64,c.BV]) (hi : outputActive (parameters c) pid i) :
    outputOffset (parameters c) pid i<v.tensor.metadata.numel := by
  have h:=G.output_bound (parameters c) pid i hp hi
  simpa only [VeriTile.Torch.TensorMetadata.numel,v.shape,Role.shape,List.prod_cons,List.prod_nil,Nat.mul_one,parameters,Nat.mul_assoc] using h

theorem gradient_bound (v : View c role) (hr : role=.queryGradient ∨ role=.keyGradient) (pid : PID)
    (hp : G.GradientPID (parameters c) pid) (i : TileIndex [64,c.BK])
    (hi : KernelGradient.gradKeyActive (parameters c) pid i) :
    KernelGradient.gradKeyOffset (parameters c) pid i<v.tensor.metadata.numel := by
  have h:=G.gradient_bound (parameters c) pid i hp hi
  rcases hr with rfl|rfl <;>
    simpa only [VeriTile.Torch.TensorMetadata.numel,v.shape,Role.shape,List.prod_cons,List.prod_nil,Nat.mul_one,parameters,Nat.mul_assoc] using h

theorem partial_bound (v : View c .valuePartials) (pid : PID) (hp : G.GradientPID (parameters c) pid)
    (vb : Nat) (i : TileIndex [64,c.BV]) (hi : KernelGradient.gradValueActive (parameters c) pid vb i) :
    KernelGradient.gradValueOffset (parameters c) pid (c.B*c.H) vb i<v.tensor.metadata.numel := by
  have h:=G.partial_bound (parameters c) pid vb i hp hi
  simpa only [VeriTile.Torch.TensorMetadata.numel,v.shape,Role.shape,List.prod_cons,List.prod_nil,Nat.mul_one,
    parameters,Nat.mul_assoc,VeriTile.Bench.Repairs.ChunkLinearAttn.keyBlocks,Shape.keyBlocks] using h

#axiomsClean history_bound
#axiomsClean state_bound
#axiomsClean output_bound
#axiomsClean gradient_bound
#axiomsClean partial_bound
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor

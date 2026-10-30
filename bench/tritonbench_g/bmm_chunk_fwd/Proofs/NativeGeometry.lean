import bench.tritonbench_g.bmm_chunk_fwd.NativeIO
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000

@[simp] theorem coordinate_batch (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    (coordinates p c pid).batch=Host.p1 pid := rfl
@[simp] theorem coordinate_chunk (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    (coordinates p c pid).chunk=Host.p2 pid/p.shape.groups := rfl
@[simp] theorem coordinate_head (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    (coordinates p c pid).head=Host.p2 pid%p.shape.groups := by
  exact Nat.mod_eq_sub_div_mul.symm
@[simp] theorem coordinate_row (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) (i : Fin c.M) :
    SourcePointers.row c (coordinates p c pid) i=RealViews.row c.M c.N (parameters p).parameters (Host.p0 pid) i := rfl
@[simp] theorem coordinate_column (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) (j : Fin c.N) :
    SourcePointers.col c (coordinates p c pid) j=RealViews.column c.N (parameters p).parameters (Host.p0 pid) j := rfl
@[simp] theorem coordinate_limit (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    SourcePointers.limit (parameters p) (coordinates p c pid)=RealViews.length (parameters p).parameters (Host.p2 pid) := rfl
@[simp] theorem coordinate_active (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    NativeReference.activeBlock (parameters p) c (coordinates p c pid)=
      RealViews.activeTile c.M c.N (parameters p).parameters (Host.p0 pid) := rfl

theorem a_address_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (q : Nat) (i : Fin c.M) (e : Fin c.K)
    (active : SourceAccess.aMask (parameters p) c (coordinates p c pid) q i e=Bool.true) :
    SourcePointers.aOffset (parameters p) c (coordinates p c pid) q i e<dataExtent p := by
  have mask : SourcePointers.row c (coordinates p c pid) i<SourcePointers.limit (parameters p) (coordinates p c pid) ∧
      e.val<p.shape.K-q*c.K := by simpa [SourceAccess.aMask] using active
  have k : q*c.K+e.val<p.shape.K := by omega
  unfold SourcePointers.aOffset FPGrid.inputOffset
  rw [coordinate_batch,coordinate_chunk,coordinate_head,coordinate_row]
  exact BmmIO.data_bound p _ _ _ _ (Host.p1_lt pid)
    (Host.local_to_sequence (parameters p) _ _ mask.1)
    (Nat.mod_lt _ (BmmIO.groups_positive p c pid)) k

theorem b_address_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (q : Nat) (e : Fin c.K) (j : Fin c.N)
    (active : SourceAccess.bMask (parameters p) c (coordinates p c pid) q e j=Bool.true) :
    SourcePointers.bOffset (parameters p) c (coordinates p c pid) q e j<dataExtent p := by
  have mask : e.val<p.shape.K-q*c.K ∧
      SourcePointers.col c (coordinates p c pid) j<SourcePointers.limit (parameters p) (coordinates p c pid) := by
    simpa [SourceAccess.bMask] using active
  have k : q*c.K+e.val<p.shape.K := by omega
  unfold SourcePointers.bOffset FPGrid.inputOffset
  rw [coordinate_batch,coordinate_chunk,coordinate_head,coordinate_column]
  exact BmmIO.data_bound p _ _ _ _ (Host.p1_lt pid)
    (Host.local_to_sequence (parameters p) _ _ mask.2)
    (Nat.mod_lt _ (BmmIO.groups_positive p c pid)) k

theorem sequence_address_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (lane : Nat) (active : lane<SourcePointers.limit (parameters p) (coordinates p c pid)) :
    SourcePointers.seqOffset (parameters p) (coordinates p c pid) lane<seqExtent p := by
  have bound := BmmIO.sequence_bound p (Host.p1 pid)
    (Host.p2 pid/p.shape.groups*p.shape.chunk+lane) (Host.p1_lt pid)
    (Host.local_to_sequence (parameters p) _ _ active)
  convert bound using 1
  simp only [SourcePointers.seqOffset,SourcePointers.seqBase,coordinate_batch,coordinate_chunk,parameters]
  ring

theorem output_address (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (i : Fin c.M) (j : Fin c.N) :
    SourcePointers.outOffset (parameters p) c (coordinates p c pid) i j=
      Host.offset (parameters p) c pid (i,j,PUnit.unit) := by
  unfold SourcePointers.outOffset Host.offset RealViews.outputOffset
  simp only [coordinate_batch,coordinate_chunk,coordinate_head,coordinate_row,coordinate_column,
    Host.Call.parameters,parameters,RealViews.chunk,RealViews.head]
  rw [←Nat.mod_eq_sub_div_mul]
  ring

theorem output_address_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (activeBlock : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true)
    (i : Fin c.M) (j : Fin c.N)
    (active : SourceAccess.outMask (parameters p) c (coordinates p c pid) i j=Bool.true) :
    SourcePointers.outOffset (parameters p) c (coordinates p c pid) i j<p.shape.outputSize := by
  have mask : SourcePointers.row c (coordinates p c pid) i<p.shape.chunk ∧
      SourcePointers.col c (coordinates p c pid) j<p.shape.chunk := by
    simpa [SourceAccess.outMask] using active
  rw [output_address]
  exact Host.output_bound (parameters p) c pid (i,j,PUnit.unit) ⟨activeBlock,mask⟩

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native

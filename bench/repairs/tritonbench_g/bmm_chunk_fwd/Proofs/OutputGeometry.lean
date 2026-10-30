import bench.repairs.tritonbench_g.bmm_chunk_fwd.IO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section

/-- Every logical output lane is represented, including partial chunks. -/
theorem output_coordinates (p : Parameters) (off : Nat) (bound : off<p.shape.outputSize) :
    outputBatch p off<p.shape.batch ∧ outputChunk p off<p.shape.chunks ∧
    outputGroup p off<p.shape.groups ∧ outputRow p off<p.shape.chunk ∧ outputColumn p off<p.shape.chunk := by
  have size : 0<p.shape.outputSize := Nat.zero_lt_of_lt bound
  have chunk : 0<p.shape.chunk := by
    by_contra h
    have z : p.shape.chunk=0 := by omega
    simp [Host.Shape.outputSize,z] at size
  have groups : 0<p.shape.groups := by
    by_contra h
    have z : p.shape.groups=0 := by omega
    simp [Host.Shape.outputSize,z] at size
  have chunks : 0<p.shape.chunks := by
    by_contra h
    have z : p.shape.chunks=0 := by omega
    simp [Host.Shape.outputSize,z] at size
  have d0 : off/p.shape.chunk<p.shape.batch*(p.shape.chunks*p.shape.groups)*p.shape.chunk :=
    (Nat.div_lt_iff_lt_mul chunk).mpr bound
  have d1 : off/p.shape.chunk/p.shape.chunk<p.shape.batch*(p.shape.chunks*p.shape.groups) :=
    (Nat.div_lt_iff_lt_mul chunk).mpr d0
  have d2 : off/p.shape.chunk/p.shape.chunk/p.shape.groups<p.shape.batch*p.shape.chunks :=
    (Nat.div_lt_iff_lt_mul groups).mpr (by simpa only [Nat.mul_assoc] using d1)
  exact ⟨(Nat.div_lt_iff_lt_mul chunks).mpr d2,Nat.mod_lt _ chunks,Nat.mod_lt _ groups,
    Nat.mod_lt _ chunk,Nat.mod_lt _ chunk⟩

theorem output_offset (p : Parameters) (off : Nat) :
    Host.returnedOffset (parameters p) (outputBatch p off) (outputChunk p off)
      (outputGroup p off) (outputRow p off) (outputColumn p off)=off := by
  unfold Host.returnedOffset outputBatch outputChunk outputGroup outputRow outputColumn
  change ((off/p.shape.chunk/p.shape.chunk/p.shape.groups/p.shape.chunks*p.shape.chunks+
    off/p.shape.chunk/p.shape.chunk/p.shape.groups%p.shape.chunks)*p.shape.groups+
    off/p.shape.chunk/p.shape.chunk%p.shape.groups)*p.shape.chunk*p.shape.chunk+
    off/p.shape.chunk%p.shape.chunk*p.shape.chunk+off%p.shape.chunk=off
  rw [Nat.div_add_mod',Nat.div_add_mod']
  rw [←Nat.add_mul,Nat.div_add_mod',Nat.div_add_mod']

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO

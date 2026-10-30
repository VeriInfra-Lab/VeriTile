import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ReturnedMemory

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.ReturnCopies
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory RawStorage HostStorage ReturnedMemory
noncomputable section
set_option maxHeartbeats 6000000

theorem history_after_copy (I : ScalarNative.Primitives) (r : Admission.Inner) (checked : r.check=Bool.true)
    (before : Memory) (p : InnerPrepared r before) (scratch : State NativeGraph.Expr)
    (i : Nat) (hi : i < innerCount r) :
    read (copyFinal r (innerGrid I r before p scratch) (HostData.finalWord I r before scratch))
      (p.history.byteBase+4*HostData.historyOffset r i)=HostData.historyWord I r before scratch i := by
  cases target : r.final with
  | none=>simp only [copyFinal,target];exact history_source_read I r checked before p scratch i hi
  | some tensor=>
    simp only [copyFinal,target]
    have shape:=(r.final_valid checked tensor target).2.2.2.2.2.2.2.2
    have fresh:=p.historyFresh tensor (by simp [Admission.Inner.tensors,target])
    have separate : Disjoint tensor.byteBase (finalCount r) p.history.byteBase (historyCapacity r) := by
      simpa [RawStorage.Disjoint,shape,finalCount,Nat.mul_assoc,or_comm] using fresh
    rw [stores_read_frame _ _ _ _ _ (disjoint_cell separate (HostCoordinates.history_bound r checked i hi))]
    exact history_source_read I r checked before p scratch i hi

/-- On the ordinary contiguous sharing branch the source-coordinate map is
the identity. The modeled return copy therefore changes no byte. -/
theorem inner_shared (I : ScalarNative.Primitives) (r : Admission.Inner) (checked : r.check=Bool.true)
    (before : Memory) (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr)
    (shared : finish.place.tensor.byteBase=p.history.byteBase) :
    innerBytes I r before p finish scratch=
      copyFinal r (innerGrid I r before p scratch) (HostData.finalWord I r before scratch) := by
  apply stores_identity
  intro i hi
  have load:=history_after_copy I r checked before p scratch i hi
  rw [finish.direct shared i hi] at load
  rw [shared]
  exact load.symm

theorem prefix_shared (I : ScalarNative.Primitives) (r : Admission.Prefix) (checked : r.check=Bool.true)
    (BS : Nat) (before : Memory) (p : PrefixPrepared r before) (finish : PrefixFinish r before p)
    (scratch : State NativeGraph.Expr) (shared : finish.place.tensor.byteBase=p.output.byteBase) :
    prefixBytes I r BS before p finish scratch=prefixGrid I r BS before p scratch := by
  apply stores_identity
  intro i hi
  have load:=prefix_source_read I r checked BS before p scratch i hi
  rw [finish.direct shared i hi] at load
  rw [shared]
  exact load.symm

#axiomsClean history_after_copy
#axiomsClean inner_shared
#axiomsClean prefix_shared
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.ReturnCopies

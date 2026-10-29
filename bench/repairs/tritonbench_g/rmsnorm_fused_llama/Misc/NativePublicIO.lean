import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.NativeReference
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.NativePublicIO

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.NativePublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (X Y W : RegionName) (rows N : Nat) (epsilon : Native.Word)
    (hN : 0<N) (hNB : N≤16384) (xy : X≠Y) (wy : W≠Y) :
    (NativeIO.original R X Y W rows N epsilon).Observes (fun _ _ => Native.eval I S)
      (Reference.native I R S X Y W rows N epsilon) := by
  intro a _ bounds hb initial hin
  refine ⟨safe _ Bool.false X Y W rows N 16384 _ hN hNB a bounds hb initial,?_⟩
  obtain ⟨final,ran,memory⟩ := RuntimeGrid.complete (Native.algebra R) X Y W rows N 16384
    (.leaf "epsilon" 0 epsilon) initial hN hNB rows
  obtain ⟨values,exterior⟩ := RuntimeGrid.graph_result (Native.algebra R) X Y W N 16384
    (.leaf "epsilon" 0 epsilon) initial hNB xy wy rows
  refine ⟨final,(run_eq _ Bool.false _ _ _ _ _ _ _ _ _).trans ran,?_,?_⟩
  · intro o j _
    fin_cases o
    obtain ⟨r,k,⟨⟩⟩ := j
    change mapCell (Native.eval I S) (mapCell id (final.mem Y (r.val*N+k.val)))=_
    rw [memory,values r.val r.isLt k.val k.isLt]
    change Cell.mk .real (Native.eval I S (RuntimeRow.value (Native.algebra R) initial X W
      r.val N 16384 (.leaf "epsilon" 0 epsilon) k.val))=_
    rw [RuntimeReference.source_value R initial X W r.val N 16384 (.leaf "epsilon" 0 epsilon)
      (Reference.rowInput (a (0 : Fin 2)) r) (Reference.weightInput (a (1 : Fin 2)))
      (fun j hj => by
        simpa [RuntimeRow.read,Reference.rowInput,hj,NativeIO.original,NativeIO.grid,mapValue]
          using hin (0 : Fin 2) (r,⟨j,hj⟩,PUnit.unit) trivial)
      (fun j hj => by
        simpa [RuntimeRow.read,Reference.weightInput,hj,NativeIO.original,NativeIO.grid,mapValue]
          using hin (1 : Fin 2) (⟨j,hj⟩,PUnit.unit) trivial) k.val k.isLt]
    rfl
  · apply frame _ Bool.false X Y W rows N 16384 _ hN a initial final
    intro r off outside
    rw [memory]
    exact exterior r off outside

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.NativePublicIO

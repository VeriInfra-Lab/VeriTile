import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.ProducerPairIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIndices

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndicesIO
open VeriTile Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
variable {α : Type} [Inhabited α]

theorem invocation_run (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (s : State α) :
    (PublicIO.Producer.invocation M integer8 p r).run s=(invocation M integer8 p r).run s := by
  apply ProducerIndices.execution
  rw [PublicIO.Producer.initial_eq]
  exact r.isLt

theorem invocation_safe (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (bounds : RegionName→Nat) (s : State α) :
    (PublicIO.Producer.invocation M integer8 p r).Safe bounds s ↔ (invocation M integer8 p r).Safe bounds s := by
  apply ProducerIndices.safety
  rw [PublicIO.Producer.initial_eq]
  exact r.isLt

theorem schedule_run (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (order : List (Fin p.rows)) (s : State α) :
    runSchedule (order.map (PublicIO.Producer.invocation M integer8 p)) s=
      runSchedule (order.map (invocation M integer8 p)) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons r rest ih=>simp only [List.map_cons,runSchedule,invocation_run,ih]

theorem schedule_safe (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (order : List (Fin p.rows)) (bounds : RegionName→Nat) (s : State α) :
    ScheduleSafe bounds (order.map (PublicIO.Producer.invocation M integer8 p)) s ↔
      ScheduleSafe bounds (order.map (invocation M integer8 p)) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons r rest ih=>simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run,ih]

local instance : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

theorem equivalent (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra Word32) (order : Producer.Schedule p) :
    (PublicIO.Producer.nativeIO f p selection rest order).Equivalent
      (variant (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order) := by
  intro xs _ bounds hb initial hin
  obtain ⟨safe,final,ran,out,frame⟩:=PublicIO.Producer.native_implements f p selection covers rest order xs trivial bounds hb initial hin
  refine ⟨safe,?_,final,final,ran,?_,fun _ _ _=>rfl,frame,frame⟩
  · exact (schedule_safe _ _ p order.order bounds initial).mp safe
  · exact (schedule_run _ _ p order.order initial).symm.trans ran

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndicesIO

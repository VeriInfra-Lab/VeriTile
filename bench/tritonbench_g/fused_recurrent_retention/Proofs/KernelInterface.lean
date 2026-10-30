import bench.tritonbench_g.fused_recurrent_retention.IO

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelInterface
open VeriTile.Triton FP.Structural
noncomputable section

def run [Inhabited α] (mode : Quoted.Mode) (a : Arithmetic α) (s : Shape)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (initial : State (Staged.Node α)) : Option (State (Staged.Node α)) :=
  Quoted.gridExec mode a s (buffers s incoming finalGradient) p.plans seed final
    (Quoted.gridPrograms s) initial

/-- Independent recurrence for every actual kernel output (including the
partial sums before library reduction), and every untouched tagged cell. -/
def Post [Inhabited α] (mode : Quoted.Mode) (a : Arithmetic α) (s : Shape)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (initial result : State (Staged.Node α)) : Prop :=
  (∀ kind i,Quoted.writtenRange mode s final (Quoted.region kind) i →
    Quoted.readWord result (Quoted.region kind) i=
      partialValue a s p.plans (Quoted.inputs (layout s incoming finalGradient) initial) seed final kind i) ∧
  (∀ region i,¬Quoted.writtenRange mode s final region i → result.mem region i=initial.mem region i)

theorem complete [Inhabited α] (mode : Quoted.Mode) (a : Arithmetic α) (s : Shape)
    (shape : s.Admitted) (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat)
    (seed final : Bool) (initial : State (Staged.Node α)) :
    ∃ result,run mode a s p incoming finalGradient seed final initial=some result ∧
      Post mode a s p incoming finalGradient seed final initial result := by
  have domain : Quoted.StoreDomain s (buffers s incoming finalGradient):=⟨shape,rfl,rfl⟩
  obtain ⟨result,executed,memory⟩:=Quoted.grid_execution_memory mode a s
    (buffers s incoming finalGradient) p initial seed final domain
  refine ⟨result,executed,?_,?_⟩
  · intro kind i inside
    exact (Quoted.grid_read mode a s (buffers s incoming finalGradient) p seed final initial result
      memory kind i inside).trans (sourceCell_partial a s p.plans _ seed final kind i)
  · intro region i outside
    rw [memory,if_neg outside]

/-- The adapter retains exactly the source layout of every admitted Tensor
application. Storage, capacity, and device fields are not kernel inputs. -/
theorem layout_correspondence (s : Shape) (b : InputBuffers) (domain : Quoted.StoreDomain s b) :
    layout s b.do_.strides b.finalGrad.strides=Quoted.layout s b := by
  simp only [layout,Quoted.layout,buffers,descriptor,domain.qDense,domain.vDense]

#axiomsClean complete
#axiomsClean layout_correspondence
#specNonCircular Post avoiding [run,Quoted.gridExec,sourceCell,programO,programQ,programK,programV]
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelInterface

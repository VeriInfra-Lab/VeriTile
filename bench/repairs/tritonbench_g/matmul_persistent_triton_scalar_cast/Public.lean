import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Physical
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Pointer

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Public
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open Core TensorBinding
noncomputable section
set_option maxHeartbeats 7000000
variable {α : Type} [Inhabited α]

/-- The checked wrapper's post-allocation branches. The positive branch runs
the entire readable quote on each actual persistent program identifier. -/
def invoke (P : Source.Arithmetic α α) (p : Parameters) (args : Arguments)
    (initial : State α) (entry : Memory α) : Option (ActualTensor × Memory α) :=
  if p.problem.M=0 ∨ p.problem.N=0 then some (args.output,entry)
  else if p.problem.K=0 then
    some (args.output,Host.zeroMemory P.inputZero p.problem args.descriptors entry)
  else (executeGrid P p initial).map (fun final =>
    (args.output,Physical.materialize p args.descriptors entry final))

theorem target_zero (P : Source.Arithmetic α α) (p : Problem) (b : Buffers)
    (entry : Memory α) (hk : p.K=0) (zero : P.cast p.profile P.accumulatorZero=P.inputZero)
    (row column : Nat) : Targets.result P p b entry row column=P.inputZero := by
  have tiles : p.kTiles=0 := by
    cases hp : p.profile <;> simp [Problem.kTiles,hk,hp,Storage.blockK,
      VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath.blocks]
  simpa only [Targets.result,tiles,Targets.accReference] using zero

theorem complete (P : Source.Arithmetic α α) (p : Parameters) (args : Arguments)
    (initial : State α) (entry : Memory α) (layout : Layout p.problem args.descriptors)
    (strides : Strides p args.descriptors) (bindings : Physical.Bindings p args.descriptors initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero) (sms : 0<p.problem.SMS) :
    ∃ after,invoke P p args initial entry=some (args.output,after) ∧
      (∀ row<p.problem.M,∀ column<p.problem.N,
        logicalValue after args.output (row*p.problem.N+column)=
          Targets.result P p.problem args.descriptors entry row column) ∧
      PhysicalFrame p.problem args.descriptors entry after ∧
      (∀ i,after (args.a.storage,i)=entry (args.a.storage,i)) ∧
      (∀ i,after (args.b.storage,i)=entry (args.b.storage,i)) := by
  have finish : ∀ after,invoke P p args initial entry=some (args.output,after) →
      (∀ row<p.problem.M,∀ column<p.problem.N,
        logicalValue after args.output (row*p.problem.N+column)=
          Targets.result P p.problem args.descriptors entry row column) →
      PhysicalFrame p.problem args.descriptors entry after →
      ∃ after,invoke P p args initial entry=some (args.output,after) ∧
        (∀ row<p.problem.M,∀ column<p.problem.N,
          logicalValue after args.output (row*p.problem.N+column)=
            Targets.result P p.problem args.descriptors entry row column) ∧
        PhysicalFrame p.problem args.descriptors entry after ∧
        (∀ i,after (args.a.storage,i)=entry (args.a.storage,i)) ∧
        (∀ i,after (args.b.storage,i)=entry (args.b.storage,i)) := by
    intro after ran values frame
    have inputs := input_storage_frame p.problem args.descriptors entry after frame layout.aSeparate layout.bSeparate
    exact ⟨after,ran,values,frame,by simpa [args.aLaws.storage] using inputs.1,
      by simpa [args.bLaws.storage] using inputs.2⟩
  by_cases empty : p.problem.M=0 ∨ p.problem.N=0
  · apply finish entry (by simp [invoke,empty])
    · intro row hr column hc
      rcases empty with hm|hn <;> omega
    · intro point outside; rfl
  by_cases hk : p.problem.K=0
  · apply finish (Host.zeroMemory P.inputZero p.problem args.descriptors entry) (by simp [invoke,empty,hk])
    · intro row hr column hc
      rw [← layout.cShape.2,logical_read _ _ _ args.outputLaws row column
        (layout.cShape.1 ▸ hr) (layout.cShape.2 ▸ hc)]
      rw [target_zero P p.problem args.descriptors entry hk zero]
      exact if_pos ⟨row,hr,column,hc,rfl⟩
    · intro point outside
      apply if_neg
      rintro ⟨row,hr,column,hc,equal⟩
      exact outside row hr column hc equal
  · obtain ⟨final,ran,post⟩ := gridExecution P p args.descriptors initial entry layout strides bindings.inputs zero sms
    apply finish (Physical.materialize p args.descriptors entry final)
      (by simp [invoke,empty,hk,ran])
    · intro row hr column hc
      exact Physical.tensor_read P p args entry initial final layout strides post row column hr hc
    · exact Physical.materialize_frame P p args.descriptors entry initial final strides bindings post

#axiomsClean complete
#specNonCircular Physical.Bindings avoiding [invoke,Targets.result]
#specNonCircular TensorBinding.Laws avoiding [invoke,Targets.result]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Public

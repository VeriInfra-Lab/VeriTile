import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelExecution
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000

/-- Independent full causal output and complete VJP; initial and terminal
state options follow the source flags. No execution or grid model appears. -/
structure RealResult (c : Real.Call) (before after : Memory ℝ) : Prop where
  output : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (v : Fin c.dims.V),
    after "output" ((h.val*c.dims.T+t.val)*c.dims.V+v.val)=
      .mk .real (Real.output c.scale (Real.inputs c.forward h.val before c.seed) t v)
  query : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (k : Fin c.dims.K),
    after "queryGradient" ((h.val*c.dims.T+t.val)*c.dims.K+k.val)=
      .mk .real ((c.target h.val before).q t k)
  key : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (k : Fin c.dims.K),
    after "keyGradient" ((h.val*c.dims.T+t.val)*c.dims.K+k.val)=
      .mk .real ((c.target h.val before).k t k)
  value : ∀ (h : Fin (c.dims.B*c.dims.H)) (t : Fin c.dims.T) (v : Fin c.dims.V),
    after "valueGradient" ((h.val*c.dims.T+t.val)*c.dims.V+v.val)=
      .mk .real ((c.target h.val before).v t v)
  final : c.final=Bool.true → ∀ (h : Fin (c.dims.B*c.dims.H)) (k : Fin c.dims.K) (v : Fin c.dims.V),
    after "final" ((h.val*c.dims.K+k.val)*c.dims.V+v.val)=
      .mk .real (Real.finalState (Real.inputs c.forward h.val before c.seed) k v)
  initial : c.seed=Bool.true → ∀ (h : Fin (c.dims.B*c.dims.H)) (k : Fin c.dims.K) (v : Fin c.dims.V),
    after "initialGradient" ((h.val*c.dims.K+k.val)*c.dims.V+v.val)=
      .mk .real ((c.target h.val before).initial k v)
  vjp : ∀ h : Fin (c.dims.B*c.dims.H), Real.IsVJP c.scale
    (Real.inputs c.backward h.val before c.seed)
    (fun t v => Real.incoming c.backward h.val before t.val v.val)
    (fun k v => Real.endpoint c.backward h.val before Bool.true c.terminal k.val v.val)
    (c.target h.val before)
  inputs : StateInputs after before

theorem real_model (c : Real.Call) (valid : Host.Valid c.dims) (before : Memory ℝ) :
    RealResult c before (c.result before) := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro h t v; exact c.output_read valid before h.val h.isLt t v
  · intro h t k; exact c.query_read valid before h.val h.isLt t k
  · intro h t k; exact c.key_read valid before h.val h.isLt t k
  · intro h t v; exact c.value_read valid before h.val h.isLt t v
  · intro flag h k v; exact c.final_read valid before flag h.val h.isLt k v
  · intro flag h k v; exact c.initial_read valid before flag h.val h.isLt k v
  · intro h; exact c.target_vjp before h.val
  · exact (sum_data Real.algebra c.backward c.selection (c.bg before) (c.bg before)).trans (c.bg_data before)

theorem real_complete (c : Real.Call) (valid : Host.Valid c.dims) (before : Memory ℝ) :
    ∃ after,run Real.algebra Real.dotPrimitive c before=some after ∧ RealResult c before after := by
  refine ⟨c.result before,?_,real_model c valid before⟩
  exact source_complete Real.algebra Real.dotPrimitive c before

#axiomsClean real_complete
#specNonCircular RealResult avoiding [run,runGrid,executeFour,gridMemory,stageMemory,
  modelFour,completeMemory,Real.Call.result]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelExecution

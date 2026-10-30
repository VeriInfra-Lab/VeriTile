import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.GridOrder

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source
noncomputable section
set_option maxHeartbeats 5000000

theorem run_reads {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) (point : Address) (read : reads b point) :
    run ops p b entry point=entry point := by
  apply source_frame ops p b layout entry point
  intro row hr column hc same
  have storage := congrArg Prod.fst same
  change b.C.storage=point.1 at storage
  rcases read with a|a
  · exact layout.aSeparate (storage.trans a)
  · exact layout.bSeparate (storage.trans a)

/-- Every invocation overwrites all C values from unchanged A/B. Therefore
the one configured autotune candidate can be replayed without an input reset. -/
theorem run_idempotent {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) : run ops p b (run ops p b entry)=run ops p b entry := by
  funext point
  by_cases active : ∃ row<p.M,∃ column<p.N,b.C.address row column=point
  · obtain ⟨row,hr,column,hc,rfl⟩ := active
    obtain ⟨tile,ht,lane,ha,hrow,hcolumn⟩ := output_covered p row column hr hc
    have address : outputAddress p b tile lane=b.C.address row column := by
      rw [output_address p b tile lane ha,hrow,hcolumn]
    rw [← address,source_output ops p b layout _ tile ht lane ha,
      source_output ops p b layout entry tile ht lane ha]
    rw [accumulator_stable ops p b entry (run ops p b entry)
      (fun point read => run_reads ops p b layout entry point read)]
  · have outside : ∀ row<p.M,∀ column<p.N,b.C.address row column≠point := by
      intro row hr column hc same
      exact active ⟨row,hr,column,hc,same⟩
    rw [source_frame ops p b layout (run ops p b entry) point outside]

def replay {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers) : Nat → Memory α → Memory α
  | 0,memory => memory
  | count+1,memory => replay ops p b count (run ops p b memory)

theorem replay_positive {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (count : Nat) (entry : Memory α) :
    replay ops p b (count+1) entry=run ops p b entry := by
  induction count generalizing entry with
  | zero => rfl
  | succ count ih =>
    change replay ops p b (count+1) (run ops p b entry)=_
    rw [ih,run_idempotent ops p b layout entry]

#axiomsClean replay_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source

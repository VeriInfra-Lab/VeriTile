import bench.tritonbench_g.chunk_retention.Sources.Loop

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

theorem SumTree.eval_congr (ops : Ops α) (tree : SumTree) (f g : Nat→α)
    (same : ∀ i∈tree.leaves, f i=g i) : tree.eval ops f=tree.eval ops g := by
  induction tree with
  | zero => rfl
  | leaf i => exact same i (by simp [SumTree.leaves])
  | add a b ia ib =>
    simp only [SumTree.eval]
    rw [ia (fun i hi => same i (by simp [SumTree.leaves,hi])),
      ib (fun i hi => same i (by simp [SumTree.leaves,hi]))]

theorem SumTree.leaf_bound (tree : SumTree) (n i : Nat) (complete : tree.Complete n)
    (hi : i∈tree.leaves) : i<n := List.mem_range.mp (complete.mem_iff.mp hi)

namespace Source
variable (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree) (flag : Bool)

theorem run_output (entry : Heap α) (h r j : Nat) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    run ops c scale schedule flag entry ⟨.output,h,0,r,j⟩=
      Reference.output ops c (inputsOfHeap entry) scale h r j := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule flag entry) _=_
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (by simp [Phase.writes]),
    afterQKV,execute_frame ops c scale schedule .qkv _ (qkv_footprint c) _ _ (by simp [Phase.writes]),
    afterDH,execute_frame ops c scale schedule .dh _ (dh_footprint c) _ _ (by simp [Phase.writes])]
  exact o_value ops c scale schedule flag entry h r j hh hr hj

theorem run_dq (entry : Heap α) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    run ops c scale schedule flag entry ⟨.dq,h,0,r,e⟩=
      Reference.dq ops c (inputsOfHeap entry) scale h r e := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule flag entry) _=_
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (by simp [Phase.writes])]
  exact dq_value ops c scale schedule flag entry h r e hh hr he

theorem run_dk (entry : Heap α) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    run ops c scale schedule flag entry ⟨.dk,h,0,r,e⟩=
      Reference.dk ops c (inputsOfHeap entry) scale h r e := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule flag entry) _=_
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (by simp [Phase.writes])]
  exact dk_value ops c scale schedule flag entry h r e hh hr he

theorem run_dv (entry : Heap α) (complete : schedule.Complete c.keyBlocks)
    (h r j : Nat) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    run ops c scale schedule flag entry ⟨.dv,h,0,r,j⟩=
      Reference.dv ops c (inputsOfHeap entry) scale schedule h r j := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule flag entry) _=_
  rw [execute_spec ops c scale schedule .sum _ (sum_footprint c),if_pos (sum_cover c h r j hh hr hj)]
  simp only [rhs,bodyRHS,project,Phase.reads,List.mem_cons_self,↓reduceIte,Reference.dv]
  apply SumTree.eval_congr
  intro block hb
  exact part_value ops c scale schedule flag entry block h r j
    (schedule.leaf_bound c.keyBlocks block complete hb) hh hr hj

theorem run_history (entry : Heap α) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    run ops c scale schedule flag entry ⟨.history,h,t,e,j⟩=
      ops.cast (Reference.history ops c (inputsOfHeap entry) h t e j) := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule flag entry) _=_
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (by simp [Phase.writes]),
    afterQKV,execute_frame ops c scale schedule .qkv _ (qkv_footprint c) _ _ (by simp [Phase.writes]),
    dh_preserves_h]
  exact h_value ops c scale schedule flag entry h t e j hh ht he hj

theorem run_final (entry : Heap α) (h e j : Nat) (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    run ops c scale schedule Bool.true entry ⟨.finalState,h,0,e,j⟩=
      Reference.finalState ops c (inputsOfHeap entry) h e j := by
  change execute ops c scale schedule .sum (eventsSum c) (afterQKV ops c scale schedule Bool.true entry) _=_
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (by simp [Phase.writes]),
    afterQKV,execute_frame ops c scale schedule .qkv _ (qkv_footprint c) _ _ (by simp [Phase.writes]),
    afterDH,execute_frame ops c scale schedule .dh _ (dh_footprint c) _ _ (by simp [Phase.writes]),
    afterO,execute_frame ops c scale schedule .o _ (o_footprint c) _ _ (by simp [Phase.writes])]
  exact final_value ops c scale schedule entry h e j hh he hj

theorem run_inputs (entry : Heap α) (point : Address) (hp : point.region∈inputRegions) :
    run ops c scale schedule flag entry point=entry point := by
  unfold run
  rw [execute_frame ops c scale schedule .sum _ (sum_footprint c) _ _ (input_separate .sum _ hp),
    execute_frame ops c scale schedule .qkv _ (qkv_footprint c) _ _ (input_separate .qkv _ hp),
    execute_frame ops c scale schedule .dh _ (dh_footprint c) _ _ (input_separate .dh _ hp),
    execute_frame ops c scale schedule .o _ (o_footprint c) _ _ (input_separate .o _ hp),
    execute_frame ops c scale schedule .h _ (h_footprint c flag) _ _ (input_separate .h _ hp)]

def footprint (c : Shape) (flag : Bool) : List Address :=
  eventsH c flag++eventsO c++eventsDH c++eventsQKV c++eventsSum c

theorem run_frame (entry : Heap α) (point : Address) (hp : point∉footprint c flag) :
    run ops c scale schedule flag entry point=entry point := by
  simp only [footprint,List.mem_append,not_or] at hp
  unfold run
  rw [execute_spec ops c scale schedule .sum _ (sum_footprint c),if_neg hp.2,
    execute_spec ops c scale schedule .qkv _ (qkv_footprint c),if_neg hp.1.2,
    execute_spec ops c scale schedule .dh _ (dh_footprint c),if_neg hp.1.1.2,
    execute_spec ops c scale schedule .o _ (o_footprint c),if_neg hp.1.1.1.2,
    execute_spec ops c scale schedule .h _ (h_footprint c flag),if_neg hp.1.1.1.1]
end Source

#axiomsClean Source.run_output
#axiomsClean Source.run_dq
#axiomsClean Source.run_dk
#axiomsClean Source.run_dv
#axiomsClean Source.run_final
#axiomsClean Source.run_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001

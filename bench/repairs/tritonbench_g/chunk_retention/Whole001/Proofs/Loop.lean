import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Grid

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

theorem masked_congr (ops : Ops α) (rows cols : Nat) (x y : Tile α)
    (h : ∀ i j, i<rows → j<cols → x i j=y i j) :
    masked ops rows cols x=masked ops rows cols y := by
  funext i j
  unfold masked
  split_ifs with hij
  · exact h i j hij.1 hij.2
  · rfl

namespace Source
variable (ops : Ops α) (c : Shape) (scale : α) (schedule : SumTree) (flag : Bool)

def afterH (entry : Heap α) := execute ops c scale schedule .h (eventsH c flag) entry
def afterO (entry : Heap α) := execute ops c scale schedule .o (eventsO c) (afterH ops c scale schedule flag entry)
def afterDH (entry : Heap α) := execute ops c scale schedule .dh (eventsDH c) (afterO ops c scale schedule flag entry)
def afterQKV (entry : Heap α) := execute ops c scale schedule .qkv (eventsQKV c) (afterDH ops c scale schedule flag entry)

theorem input_separate (p : Phase) (r : Region) (hr : r∈inputRegions) : r∉p.writes := by
  cases p <;> cases r <;> simp_all [Phase.writes,inputRegions]

theorem execute_inputs (p : Phase) (events : List Address)
    (footprint : ∀ a∈events, a.region∈p.writes) (entry : Heap α) :
    inputsOfHeap (execute ops c scale schedule p events entry)=inputsOfHeap entry := by
  have h (tag : Region) (ht : tag∈inputRegions) (head row col : Nat) :=
    execute_frame ops c scale schedule p events footprint entry ⟨tag,head,0,row,col⟩ (input_separate p tag ht)
  simp only [inputsOfHeap]
  congr 1 <;> funext head row col <;> apply h <;> simp [inputRegions]

theorem h_inputs (entry : Heap α) : inputsOfHeap (afterH ops c scale schedule flag entry)=inputsOfHeap entry :=
  execute_inputs ops c scale schedule .h _ (h_footprint c flag) entry
theorem o_inputs (entry : Heap α) : inputsOfHeap (afterO ops c scale schedule flag entry)=inputsOfHeap entry := by
  rw [afterO,execute_inputs ops c scale schedule .o _ (o_footprint c),h_inputs]
theorem dh_inputs (entry : Heap α) : inputsOfHeap (afterDH ops c scale schedule flag entry)=inputsOfHeap entry := by
  rw [afterDH,execute_inputs ops c scale schedule .dh _ (dh_footprint c),o_inputs]
theorem qkv_inputs (entry : Heap α) : inputsOfHeap (afterQKV ops c scale schedule flag entry)=inputsOfHeap entry := by
  rw [afterQKV,execute_inputs ops c scale schedule .qkv _ (qkv_footprint c),dh_inputs]

theorem project_inputs (p : Phase) (hp : p≠.sum) (entry : Heap α) :
    inputsOfHeap (project ops p entry)=inputsOfHeap entry := by
  cases p <;> simp_all [inputsOfHeap,project,Phase.reads,inputRegions]

theorem h_value (entry : Heap α) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    afterH ops c scale schedule flag entry ⟨.history,h,t,e,j⟩=
      ops.cast (Reference.history ops c (inputsOfHeap entry) h t e j) := by
  rw [afterH,execute_spec ops c scale schedule .h _ (h_footprint c flag),
    if_pos (history_cover c flag h t e j hh ht he hj)]
  simp only [rhs,bodyRHS,project_inputs ops .h (by decide),↓reduceIte]

theorem h_tile (entry : Heap α) (h t : Nat) (hh : h<c.heads) (ht : t<c.chunks) :
    masked ops c.K c.V (fun e j => afterH ops c scale schedule flag entry ⟨.history,h,t,e,j⟩)=
      masked ops c.K c.V (fun e j => ops.cast (Reference.history ops c (inputsOfHeap entry) h t e j)) := by
  apply masked_congr
  exact fun e j he hj => h_value ops c scale schedule flag entry h t e j hh ht he hj

theorem final_value (entry : Heap α) (h e j : Nat)
    (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    afterH ops c scale schedule Bool.true entry ⟨.finalState,h,0,e,j⟩=
      Reference.finalState ops c (inputsOfHeap entry) h e j := by
  rw [afterH,execute_spec ops c scale schedule .h _ (h_footprint c Bool.true),
    if_pos (final_cover c h e j hh he hj)]
  simp only [rhs,bodyRHS,project_inputs ops .h (by decide)]
  rfl

theorem o_value (entry : Heap α) (h r j : Nat) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    afterO ops c scale schedule flag entry ⟨.output,h,0,r,j⟩=
      Reference.output ops c (inputsOfHeap entry) scale h r j := by
  rw [afterO,execute_spec ops c scale schedule .o _ (o_footprint c),
    if_pos (output_cover c h r j hh hr hj)]
  simp only [rhs,bodyRHS,project_inputs ops .o (by decide),h_inputs]
  simp only [project,Phase.reads,List.mem_cons_self,↓reduceIte]
  unfold Reference.output Body.output Body.outputCarry
  rw [h_tile ops c scale schedule flag entry h (r/64) hh (c.chunk_of_row r hr)]

theorem dh_value (entry : Heap α) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    afterDH ops c scale schedule flag entry ⟨.dhistory,h,t,e,j⟩=
      ops.cast (Reference.dhistory ops c (inputsOfHeap entry) scale h t e j) := by
  rw [afterDH,execute_spec ops c scale schedule .dh _ (dh_footprint c),
    if_pos (dhistory_cover c h t e j hh ht he hj)]
  simp only [rhs,bodyRHS,project_inputs ops .dh (by decide),o_inputs]

theorem dh_tile (entry : Heap α) (h t : Nat) (hh : h<c.heads) (ht : t<c.chunks) :
    masked ops c.K c.V (fun e j => afterDH ops c scale schedule flag entry ⟨.dhistory,h,t,e,j⟩)=
      masked ops c.K c.V (fun e j => ops.cast (Reference.dhistory ops c (inputsOfHeap entry) scale h t e j)) := by
  apply masked_congr
  exact fun e j he hj => dh_value ops c scale schedule flag entry h t e j hh ht he hj

theorem dh_preserves_h (entry : Heap α) (h t e j : Nat) :
    afterDH ops c scale schedule flag entry ⟨.history,h,t,e,j⟩=
      afterH ops c scale schedule flag entry ⟨.history,h,t,e,j⟩ := by
  rw [afterDH,execute_frame ops c scale schedule .dh _ (dh_footprint c) _ _ (by simp [Phase.writes]),
    afterO,execute_frame ops c scale schedule .o _ (o_footprint c) _ _ (by simp [Phase.writes])]

theorem dq_value (entry : Heap α) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    afterQKV ops c scale schedule flag entry ⟨.dq,h,0,r,e⟩=
      Reference.dq ops c (inputsOfHeap entry) scale h r e := by
  rw [afterQKV,execute_spec ops c scale schedule .qkv _ (qkv_footprint c),
    if_pos (dq_cover c h r e hh hr he)]
  simp only [rhs,bodyRHS,project_inputs ops .qkv (by decide),dh_inputs,↓reduceIte]
  simp only [project,Phase.reads,List.mem_cons_self,↓reduceIte,dh_preserves_h]
  unfold Reference.dq Body.dq
  rw [h_tile ops c scale schedule flag entry h (r/64) hh (c.chunk_of_row r hr)]

theorem dk_value (entry : Heap α) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    afterQKV ops c scale schedule flag entry ⟨.dk,h,0,r,e⟩=
      Reference.dk ops c (inputsOfHeap entry) scale h r e := by
  rw [afterQKV,execute_spec ops c scale schedule .qkv _ (qkv_footprint c),
    if_pos (dk_cover c h r e hh hr he)]
  simp only [rhs,bodyRHS,project_inputs ops .qkv (by decide),dh_inputs]
  simp only [reduceCtorEq,↓reduceIte,project,Phase.reads,List.mem_cons,or_true,true_or,false_or]
  unfold Reference.dk Body.dk
  rw [dh_tile ops c scale schedule flag entry h (r/64) hh (c.chunk_of_row r hr)]

theorem part_value (entry : Heap α) (b h r j : Nat)
    (hb : b<c.keyBlocks) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    afterQKV ops c scale schedule flag entry ⟨.dvPart,h,b,r,j⟩=
      Reference.dvPart ops c (inputsOfHeap entry) scale b h r j := by
  rw [afterQKV,execute_spec ops c scale schedule .qkv _ (qkv_footprint c),
    if_pos (partial_cover c b h r j hb hh hr hj)]
  simp only [rhs,bodyRHS,project_inputs ops .qkv (by decide),dh_inputs]
  simp only [reduceCtorEq,↓reduceIte,project,Phase.reads,List.mem_cons,or_true,true_or,false_or]
  unfold Reference.dvPart Body.dvPartial
  rw [dh_tile ops c scale schedule flag entry h (r/64) hh (c.chunk_of_row r hr)]
end Source

#axiomsClean Source.o_value
#axiomsClean Source.dq_value
#axiomsClean Source.dk_value
#axiomsClean Source.part_value
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001

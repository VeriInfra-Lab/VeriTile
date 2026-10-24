import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Physical
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Separate

def joined (before later : Heap α) : Inputs α :=
  {inputsOfHeap later with seed := (inputsOfHeap before).seed}

/-- Only saved Q/K/V and history are required at backward. The later incoming
and terminal cotangents are unrestricted, and seed storage need not survive. -/
structure Saved (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (flag : Bool)
    (before later : Heap α) : Prop where
  q : (inputsOfHeap later).q=(inputsOfHeap before).q
  k : (inputsOfHeap later).k=(inputsOfHeap before).k
  v : (inputsOfHeap later).v=(inputsOfHeap before).v
  history : ∀ h<c.heads,∀ t<c.chunks,∀ e<c.K,∀ j<c.V,
    later ⟨.history,h,t,e,j⟩=Source.afterO ops c scale tree flag before ⟨.history,h,t,e,j⟩

def backwardH (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (later : Heap α) :=
  Source.execute ops c scale tree .dh (Source.eventsDH c) later

def backwardQKV (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (later : Heap α) :=
  Source.execute ops c scale tree .qkv (Source.eventsQKV c) (backwardH ops c scale tree later)

def backward (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (later : Heap α) :=
  Source.execute ops c scale tree .sum (Source.eventsSum c) (backwardQKV ops c scale tree later)

def forwardFootprint (c : Shape) (flag : Bool) := Source.eventsH c flag++Source.eventsO c
def backwardFootprint (c : Shape) := Source.eventsDH c++Source.eventsQKV c++Source.eventsSum c

theorem history_joined (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (flag : Bool)
    (before later : Heap α) (saved : Saved ops c scale tree flag before later) (h n : Nat) :
    Reference.history ops c (joined before later) h n=Reference.history ops c (inputsOfHeap before) h n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Reference.history]
    rw [ih]
    funext e j
    simp only [Body.nextH,kAt,vAt,masked,joined,saved.k,saved.v]

theorem reverse_joined (ops : Ops α) (c : Shape) (scale : α)
    (before later : Heap α) (h n : Nat) :
    Reference.reverseCarry ops c (joined before later) scale h n=
      Reference.reverseCarry ops c (inputsOfHeap later) scale h n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Reference.reverseCarry]
    rw [ih]
    rfl

theorem saved_tile (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (flag : Bool)
    (before later : Heap α) (saved : Saved ops c scale tree flag before later)
    (h t : Nat) (hh : h<c.heads) (ht : t<c.chunks) :
    masked ops c.K c.V (fun e j=>backwardH ops c scale tree later ⟨.history,h,t,e,j⟩)=
      masked ops c.K c.V (fun e j=>ops.cast (Reference.history ops c (joined before later) h t e j)) := by
  apply masked_congr
  intro e j he hj
  rw [backwardH,Source.execute_frame ops c scale tree .dh _ (Source.dh_footprint c) _ _ (by simp [Phase.writes]),
    saved.history h hh t ht e he j hj,Source.afterO,
    Source.execute_frame ops c scale tree .o _ (Source.o_footprint c) _ _ (by simp [Phase.writes]),
    Source.h_value ops c scale tree flag before h t e j hh ht he hj,
    history_joined ops c scale tree flag before later saved]

theorem backwardH_inputs (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (later : Heap α) :
    inputsOfHeap (backwardH ops c scale tree later)=inputsOfHeap later :=
  Source.execute_inputs ops c scale tree .dh _ (Source.dh_footprint c) later

theorem dh_tile (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (before later : Heap α) (h t : Nat) (hh : h<c.heads) (ht : t<c.chunks) :
    masked ops c.K c.V (fun e j=>backwardH ops c scale tree later ⟨.dhistory,h,t,e,j⟩)=
      masked ops c.K c.V (fun e j=>ops.cast (Reference.dhistory ops c (joined before later) scale h t e j)) := by
  apply masked_congr
  intro e j he hj
  rw [backwardH,Source.execute_spec ops c scale tree .dh _ (Source.dh_footprint c),
    if_pos (Source.dhistory_cover c h t e j hh ht he hj)]
  simp only [Source.rhs,Source.bodyRHS,Source.project_inputs ops .dh (by decide)]
  simp only [Reference.dhistory,reverse_joined,↓reduceIte]

 theorem query (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (flag : Bool)
    (before later : Heap α) (saved : Saved ops c scale tree flag before later)
    (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    backwardQKV ops c scale tree later ⟨.dq,h,0,r,e⟩=
      Reference.dq ops c (joined before later) scale h r e := by
  rw [backwardQKV,Source.execute_spec ops c scale tree .qkv _ (Source.qkv_footprint c),
    if_pos (Source.dq_cover c h r e hh hr he)]
  simp only [Source.rhs,Source.bodyRHS,Source.project_inputs ops .qkv (by decide),backwardH_inputs,↓reduceIte]
  simp only [project,Phase.reads,List.mem_cons_self,↓reduceIte]
  unfold Reference.dq Body.dq
  rw [saved_tile ops c scale tree flag before later saved h (r/64) hh (c.chunk_of_row r hr)]
  rfl

theorem key (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (before later : Heap α) (h r e : Nat) (hh : h<c.heads) (hr : r<c.T) (he : e<c.K) :
    backwardQKV ops c scale tree later ⟨.dk,h,0,r,e⟩=
      Reference.dk ops c (joined before later) scale h r e := by
  rw [backwardQKV,Source.execute_spec ops c scale tree .qkv _ (Source.qkv_footprint c),
    if_pos (Source.dk_cover c h r e hh hr he)]
  simp only [Source.rhs,Source.bodyRHS,Source.project_inputs ops .qkv (by decide),backwardH_inputs]
  simp only [reduceCtorEq,↓reduceIte,project,Phase.reads,List.mem_cons,or_true,true_or,false_or]
  unfold Reference.dk Body.dk
  rw [dh_tile ops c scale tree before later h (r/64) hh (c.chunk_of_row r hr)]
  rfl

theorem value_partial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (before later : Heap α) (block h r j : Nat)
    (hb : block<c.keyBlocks) (hh : h<c.heads) (hr : r<c.T) (hj : j<c.V) :
    backwardQKV ops c scale tree later ⟨.dvPart,h,block,r,j⟩=
      Reference.dvPart ops c (joined before later) scale block h r j := by
  rw [backwardQKV,Source.execute_spec ops c scale tree .qkv _ (Source.qkv_footprint c),
    if_pos (Source.partial_cover c block h r j hb hh hr hj)]
  simp only [Source.rhs,Source.bodyRHS,Source.project_inputs ops .qkv (by decide),backwardH_inputs]
  simp only [reduceCtorEq,↓reduceIte,project,Phase.reads,List.mem_cons,or_true,true_or,false_or]
  unfold Reference.dvPart Body.dvPartial
  rw [dh_tile ops c scale tree before later h (r/64) hh (c.chunk_of_row r hr)]
  rfl

structure BackwardPost (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (tree : SumTree) (after : Heap α) : Prop where
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,after ⟨.dq,h,0,t,e⟩=Reference.dq ops c xs scale h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,after ⟨.dk,h,0,t,e⟩=Reference.dk ops c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,after ⟨.dv,h,0,t,j⟩=Reference.dv ops c xs scale tree h t j

theorem backward_post (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (flag : Bool)
    (before later : Heap α) (saved : Saved ops c scale tree flag before later) :
    BackwardPost ops c (joined before later) scale tree (backward ops c scale tree later) := by
  refine ⟨?_,?_,?_⟩
  · intro h hh t ht e he
    rw [backward,Source.execute_frame ops c scale tree .sum _ (Source.sum_footprint c) _ _ (by simp [Phase.writes])]
    exact query ops c scale tree flag before later saved h t e hh ht he
  · intro h hh t ht e he
    rw [backward,Source.execute_frame ops c scale tree .sum _ (Source.sum_footprint c) _ _ (by simp [Phase.writes])]
    exact key ops c scale tree before later h t e hh ht he
  · intro h hh t ht j hj
    rw [backward,Source.execute_spec ops c scale tree .sum _ (Source.sum_footprint c),
      if_pos (Source.sum_cover c h t j hh ht hj)]
    simp only [Source.rhs,Source.bodyRHS,project,Phase.reads,List.mem_cons_self,↓reduceIte,Reference.dv]
    apply SumTree.eval_congr
    intro block hb
    exact value_partial ops c scale tree before later block h t j
      (tree.leaf_bound c.keyBlocks block complete hb) hh ht hj

theorem forward_frame (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (flag : Bool)
    (before : Heap α) (point : Address) (outside : point∉forwardFootprint c flag) :
    Source.afterO ops c scale tree flag before point=before point := by
  simp only [forwardFootprint,List.mem_append,not_or] at outside
  rw [Source.afterO,Source.execute_spec ops c scale tree .o _ (Source.o_footprint c),if_neg outside.2,
    Source.afterH,Source.execute_spec ops c scale tree .h _ (Source.h_footprint c flag),if_neg outside.1]

theorem backward_frame (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (later : Heap α) (point : Address) (outside : point∉backwardFootprint c) :
    backward ops c scale tree later point=later point := by
  simp only [backwardFootprint,List.mem_append,not_or] at outside
  rw [backward,Source.execute_spec ops c scale tree .sum _ (Source.sum_footprint c),if_neg outside.2,
    backwardQKV,Source.execute_spec ops c scale tree .qkv _ (Source.qkv_footprint c),if_neg outside.1.2,
    backwardH,Source.execute_spec ops c scale tree .dh _ (Source.dh_footprint c),if_neg outside.1.1]

#axiomsClean history_joined
#axiomsClean backward_post
#axiomsClean forward_frame
#axiomsClean backward_frame
end Separate
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001


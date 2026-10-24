import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Separate
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
namespace Separate

structure ForwardPost (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (flag : Bool) (after : Heap α) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,after ⟨.output,h,0,t,j⟩=Reference.output ops c xs scale h t j
  finalState : flag=Bool.true → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    after ⟨.finalState,h,0,e,j⟩=Reference.finalState ops c xs h e j

theorem forward_post (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (flag : Bool) (before : Heap α) :
    ForwardPost ops c (inputsOfHeap before) scale flag (Source.afterO ops c scale tree flag before) := by
  refine ⟨?_,?_⟩
  · intro h hh t ht j hj
    exact Source.o_value ops c scale tree flag before h t j hh ht hj
  · intro hf h hh e he j hj
    subst flag
    rw [Source.afterO,Source.execute_frame ops c scale tree .o _ (Source.o_footprint c) _ _ (by simp [Phase.writes])]
    exact Source.final_value ops c scale tree before h e j hh he hj

structure RealForwardPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (flag : Bool) (after : Heap ℝ) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.output,h,0,t,j⟩=MathTarget.output c xs scale Bool.true h t j
  finalState : flag=Bool.true → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    after ⟨.finalState,h,0,e,j⟩=MathTarget.finalState c xs Bool.true h e j

structure RealBackwardPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (after : Heap ℝ) : Prop where
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dq,h,0,t,e⟩=MathTarget.queryGradient c xs scale Bool.true h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dk,h,0,t,e⟩=MathTarget.keyGradient c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.dv,h,0,t,j⟩=MathTarget.valueGradient c xs scale h t j

theorem forward_real (c : Shape) (scale : ℝ) (tree : SumTree) (flag : Bool) (before : Heap ℝ) :
    RealForwardPost c (inputsOfHeap before) scale flag (Source.afterO realOps c scale tree flag before) := by
  have post:=forward_post realOps c scale tree flag before
  refine ⟨?_,?_⟩
  · intro h hh t ht j hj
    exact (post.output h hh t ht j hj).trans (RealScalar.reference_output c _ scale h t j ht hj)
  · intro hf h hh e he j hj
    exact (post.finalState hf h hh e he j hj).trans (RealScalar.reference_final c _ h e j he hj)

theorem backward_real (c : Shape) (scale : ℝ) (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (flag : Bool) (before later : Heap ℝ) (saved : Saved realOps c scale tree flag before later) :
    RealBackwardPost c (joined before later) scale (backward realOps c scale tree later) := by
  have post:=backward_post realOps c scale tree complete flag before later saved
  refine ⟨?_,?_,?_⟩
  · intro h hh t ht e he
    exact (post.query h hh t ht e he).trans (RealScalar.reference_query c _ scale h t e ht he)
  · intro h hh t ht e he
    exact (post.key h hh t ht e he).trans (RealScalar.reference_key c _ scale h t e ht he)
  · intro h hh t ht j hj
    exact (post.value h hh t ht j hj).trans (RealScalar.reference_value c _ scale tree complete h t j ht hj)

def forwardPhysical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (flag : Bool) (before : Memory α) : Memory α :=
  Source.executePhysical ops c b scale tree .o (Source.eventsO c)
    (Source.executePhysical ops c b scale tree .h (Source.eventsH c flag) before)

def backwardPhysical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (later : Memory α) : Memory α :=
  Source.executePhysical ops c b scale tree .sum (Source.eventsSum c)
    (Source.executePhysical ops c b scale tree .qkv (Source.eventsQKV c)
      (Source.executePhysical ops c b scale tree .dh (Source.eventsDH c) later))

theorem forward_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (flag : Bool) (before : Memory α) :
    b.view ops.zero (forwardPhysical ops c b scale tree flag before)=
      Source.afterO ops c scale tree flag (b.view ops.zero before) := by
  unfold forwardPhysical Source.afterO Source.afterH
  rw [Source.execute_physical ops c b scale tree .o _ (Source.o_valid c)
      (fun a ha=>Source.write_not_input .o a (Source.o_footprint c a ha)),
    Source.execute_physical ops c b scale tree .h _ (Source.h_valid c flag)
      (fun a ha=>Source.write_not_input .h a (Source.h_footprint c flag a ha))]

theorem backward_physical (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (later : Memory α) :
    b.view ops.zero (backwardPhysical ops c b scale tree later)=
      backward ops c scale tree (b.view ops.zero later) := by
  unfold backwardPhysical backward backwardQKV backwardH
  rw [Source.execute_physical ops c b scale tree .sum _ (Source.sum_valid c)
      (fun a ha=>Source.write_not_input .sum a (Source.sum_footprint c a ha)),
    Source.execute_physical ops c b scale tree .qkv _ (Source.qkv_valid c)
      (fun a ha=>Source.write_not_input .qkv a (Source.qkv_footprint c a ha)),
    Source.execute_physical ops c b scale tree .dh _ (Source.dh_valid c)
      (fun a ha=>Source.write_not_input .dh a (Source.dh_footprint c a ha))]

theorem forward_physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (flag : Bool) (before : Memory α) (point : Nat)
    (outside : ∀ a∈forwardFootprint c flag,point≠b.address a) :
    forwardPhysical ops c b scale tree flag before point=before point := by
  unfold forwardPhysical
  rw [Source.physical_frame ops c b scale tree .o _ _ point (fun a ha=>outside a (by simp [forwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .h _ _ point (fun a ha=>outside a (by simp [forwardFootprint,ha]))]

theorem backward_physical_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (later : Memory α) (point : Nat)
    (outside : ∀ a∈backwardFootprint c,point≠b.address a) :
    backwardPhysical ops c b scale tree later point=later point := by
  unfold backwardPhysical
  rw [Source.physical_frame ops c b scale tree .sum _ _ point (fun a ha=>outside a (by simp [backwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .qkv _ _ point (fun a ha=>outside a (by simp [backwardFootprint,ha])),
    Source.physical_frame ops c b scale tree .dh _ _ point (fun a ha=>outside a (by simp [backwardFootprint,ha]))]

#axiomsClean forward_post
#axiomsClean backward_real
#axiomsClean forward_physical
#axiomsClean backward_physical
#axiomsClean forward_physical_frame
#axiomsClean backward_physical_frame
end Separate
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001

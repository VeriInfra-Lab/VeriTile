import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Execution
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Affine

namespace VeriTile.Bench.Repairs.LayerNormTriton.Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open Storage
noncomputable section
open scoped Classical

theorem final_congr (ops : Ops α) (plan : FinalPlan) (groups : Nat) (a b : Nat → α)
    (h : ∀ i,i<groups → a i=b i) (column : Nat) :
    finalValue ops plan groups a column=finalValue ops plan groups b column := by
  have he : finalLane ops plan groups a=finalLane ops plan groups b := by
    funext lane
    unfold finalLane
    congr 1
    funext acc iteration
    by_cases hi : iteration*32+lane.val<groups
    · simp only [hi,if_true,h _ hi]
    · simp only [hi,if_false]
  unfold finalValue
  rw [he]

theorem final_scratch_values (ops : Ops α) (plan : Plan N B) (last : FinalPlan)
    (api : Memory.API α Store Byte) (p : BackwardBuffers M N G inputs)
    (q : FinalBuffers N inputs) (x w dy r mu : Dense) (m : Storage.World Store)
    (first : E.Schedule 3) (covered : ∀ k i,i<(p.buffers.output k).count → (k,i)∈first.order)
    (i : Fin (q.buffers.output 0).count) :
    let after := p.buffers.run api (groupValues ops plan api p x w dy r mu m) first m
    finalValues ops last api q (min G M) (p.buffers.output 1) (p.buffers.output 2) after 0 i=
      finalValue ops last (min G M)
        (fun group => (groupValue ops plan M G group (E.rows x N m api) (E.rows dy N m api) (E.saved api r mu m) i.val).1) i.val := by
  apply final_congr
  intro group hg
  exact (group_readback ops plan api p x w dy r mu m first covered group i.val hg (by simpa [q.dwCount] using i.isLt)).1

theorem final_scratch_bias_values (ops : Ops α) (plan : Plan N B) (last : FinalPlan)
    (api : Memory.API α Store Byte) (p : BackwardBuffers M N G inputs)
    (q : FinalBuffers N inputs) (x w dy r mu : Dense) (m : Storage.World Store)
    (first : E.Schedule 3) (covered : ∀ k i,i<(p.buffers.output k).count → (k,i)∈first.order)
    (i : Fin (q.buffers.output 1).count) :
    let after := p.buffers.run api (groupValues ops plan api p x w dy r mu m) first m
    finalValues ops last api q (min G M) (p.buffers.output 1) (p.buffers.output 2) after 1 i=
      finalValue ops last (min G M)
        (fun group => (groupValue ops plan M G group (E.rows x N m api) (E.rows dy N m api) (E.saved api r mu m) i.val).2) i.val := by
  apply final_congr
  intro group hg
  exact (group_readback ops plan api p x w dy r mu m first covered group i.val hg (by simpa [q.dbCount] using i.isLt)).2

def twoLaunch (ops : Ops α) (plan : Plan N B) (last : FinalPlan) (api : Memory.API α Store Byte)
    (p : BackwardBuffers M N G inputs) (q : FinalBuffers N inputs)
    (x w dy r mu : Dense) (m : Storage.World Store) (first : E.Schedule 3) (second : E.Schedule 2) :=
  let after := p.buffers.run api (groupValues ops plan api p x w dy r mu m) first m
  q.buffers.run api (finalValues ops last api q (min G M) (p.buffers.output 1) (p.buffers.output 2) after) second after

theorem twoLaunch_frame (ops : Ops α) (plan : Plan N B) (last : FinalPlan) (api : Memory.API α Store Byte)
    (p : BackwardBuffers M N G inputs) (q : FinalBuffers N inputs)
    (x w dy r mu : Dense) (m : Storage.World Store) (first : E.Schedule 3) (second : E.Schedule 2)
    (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api (twoLaunch ops plan last api p q x w dy r mu m first second) key=Storage.observe api m key := by
  unfold twoLaunch
  exact (run_frame q.buffers api _ second _ key hk).trans (run_frame p.buffers api _ first m key hk)

theorem twoLaunch_dw_real (plan : Plan N B) (last : FinalPlan)
    (p : BackwardBuffers M N G inputs) (q : FinalBuffers N inputs) (hg : 0<G)
    (x w dy r mu : Dense) (m : Storage.World (Nat → ℝ)) (first : E.Schedule 3) (second : E.Schedule 2)
    (hfirst : ∀ k i,i<(p.buffers.output k).count → (k,i)∈first.order)
    (hsecond : ∀ k i,i<(q.buffers.output k).count → (k,i)∈second.order)
    (i : Fin (q.buffers.output 0).count) :
    twoLaunch RealProof.ops plan last Memory.realAPI p q x w dy r mu m first second q.buffers.owner
      ((q.buffers.output 0).address i.val)=
      RealProof.Spec.weightGradient N M (E.rows x N m Memory.realAPI) (E.rows dy N m Memory.realAPI)
        (E.saved Memory.realAPI r mu m) i.val := by
  have hs := run_readback q.buffers Memory.realAPI
    (finalValues RealProof.ops last Memory.realAPI q (min G M) (p.buffers.output 1) (p.buffers.output 2)
      (p.buffers.run Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m) first m))
    second hsecond (p.buffers.run Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m) first m) 0 i
  change twoLaunch RealProof.ops plan last Memory.realAPI p q x w dy r mu m first second
    q.buffers.owner ((q.buffers.output 0).address i.val) = _ at hs
  rw [hs,final_scratch_values RealProof.ops plan last Memory.realAPI p q x w dy r mu m first hfirst i]
  exact RealProof.weightGradient_real plan last M G hg _ _ _ _ (by simpa [q.dwCount] using i.isLt)

theorem twoLaunch_db_real (plan : Plan N B) (last : FinalPlan)
    (p : BackwardBuffers M N G inputs) (q : FinalBuffers N inputs) (hg : 0<G)
    (x w dy r mu : Dense) (m : Storage.World (Nat → ℝ)) (first : E.Schedule 3) (second : E.Schedule 2)
    (hfirst : ∀ k i,i<(p.buffers.output k).count → (k,i)∈first.order)
    (hsecond : ∀ k i,i<(q.buffers.output k).count → (k,i)∈second.order)
    (i : Fin (q.buffers.output 1).count) :
    twoLaunch RealProof.ops plan last Memory.realAPI p q x w dy r mu m first second q.buffers.owner
      ((q.buffers.output 1).address i.val)=
      RealProof.Spec.biasGradient N M (E.rows dy N m Memory.realAPI) i.val := by
  have hs := run_readback q.buffers Memory.realAPI
    (finalValues RealProof.ops last Memory.realAPI q (min G M) (p.buffers.output 1) (p.buffers.output 2)
      (p.buffers.run Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m) first m))
    second hsecond (p.buffers.run Memory.realAPI (groupValues RealProof.ops plan Memory.realAPI p x w dy r mu m) first m) 1 i
  change twoLaunch RealProof.ops plan last Memory.realAPI p q x w dy r mu m first second
    q.buffers.owner ((q.buffers.output 1).address i.val) = _ at hs
  rw [hs,final_scratch_bias_values RealProof.ops plan last Memory.realAPI p q x w dy r mu m first hfirst i]
  exact RealProof.biasGradient_real plan last M G hg _ _ _ _ (by simpa [q.dbCount] using i.isLt)

#axiomsClean final_scratch_values
#axiomsClean twoLaunch_frame
#axiomsClean twoLaunch_dw_real
#axiomsClean twoLaunch_db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Execution

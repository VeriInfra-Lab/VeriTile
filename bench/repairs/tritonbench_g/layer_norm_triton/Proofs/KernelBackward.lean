import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelForward

/-! Composition of the independent grouped and final native graphs. The
second graph reads the actual scratch memory written by the first graph.
Constructed coverage replaces all former runtime-success receipts. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelBackward
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open Storage KernelMemory
noncomputable section
open scoped Classical
variable {x w dy r mu : Input}

def outputs (c : KernelContract.Backward M N x w dy r mu) : Fin 3 → Dense
  | ⟨0,_⟩ => c.first.buffers.output 0
  | ⟨1,_⟩ => c.last.buffers.output 0
  | ⟨2,_⟩ => c.last.buffers.output 1

def initialStats (api : Memory.API α Store Byte) (neg : α → α)
    (r mu : Input) (m : World Store) (row : Nat) : Reuse.Statistics α :=
  ⟨inputValue api neg mu m row,inputValue api neg r m row⟩

theorem initialStats_load (api : Memory.API α Store Byte) (neg : α → α)
    (r mu : Input) (m : World Store) :
    initialStats api neg r mu m=Execution.E.saved api r.dense mu.dense m := by
  funext row
  simp only [initialStats,inputValue_load]
  rfl

theorem row_load (api : Memory.API α Store Byte) (neg : α → α) (x : Input)
    (m : World Store) (N : Nat) :
    KernelForward.row api neg x m N=Execution.E.rows x.dense N m api := by
  funext row col
  exact inputValue_load api neg x m _

theorem vector_load (api : Memory.API α Store Byte) (neg : α → α) (x : Input)
    (m : World Store) : inputValue api neg x m=Execution.E.vector x.dense m api := by
  funext i
  exact inputValue_load api neg x m i

def nativeValues (ops : Ops α) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store) :
    ∀ k,Fin (outputs c k).count → α
  | ⟨0,_⟩,i => backwardValue ops p.base (initialStats api neg r mu m (i.val/N))
      (KernelForward.row api neg x m N (i.val/N)) (KernelForward.row api neg dy m N (i.val/N))
      (inputValue api neg w m) (i.val%N)
  | ⟨1,_⟩,i => finalValue ops p.final (min (Geometry.groups N) M)
      (fun group => (groupValue ops p.base M (Geometry.groups N) group
        (KernelForward.row api neg x m N) (KernelForward.row api neg dy m N)
        (initialStats api neg r mu m) i.val).1) i.val
  | ⟨2,_⟩,i => finalValue ops p.final (min (Geometry.groups N) M)
      (fun group => (groupValue ops p.base M (Geometry.groups N) group
        (KernelForward.row api neg x m N) (KernelForward.row api neg dy m N)
        (initialStats api neg r mu m) i.val).2) i.val

def run (ops : Ops α) (p : KernelGraph.BackwardPlan N B) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store) : World Store :=
  Execution.twoLaunch ops p.base p.final api c.first c.last x.dense w.dense dy.dense r.dense mu.dense m
    (schedule c.first.buffers) (schedule c.last.buffers)

def final (ops : Ops α) (p : KernelGraph.BackwardPlan N B) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store) : World Store :=
  if KernelContract.launched M N then run ops p api c m else m

def exec (ops : Ops α) (p : KernelGraph.BackwardPlan N B) (api : Memory.API α Store Byte)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store) : Except Unit (World Store) :=
  .ok (final ops p api c m)

theorem run_first_read (ops : Ops α) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store)
    (k : Fin 3) (i : Fin (c.first.buffers.output k).count) :
    logical api neg (run ops p api c m) (c.first.buffers.output k).tensor i.val=
      Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m k i := by
  let firstValues := Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m
  let after := KernelMemory.run c.first.buffers api firstValues m
  have owned : (c.first.buffers.output k).owner=c.last.buffers.owner :=
    (c.first.buffers.owned k).trans ((c.first.buffers.owned 0).symm.trans c.sameOwner)
  rw [dense_logical api neg _ _ _ i.isLt]
  apply (Execution.run_input c.last.buffers api
    (Execution.finalValues ops p.final api c.last (min (Geometry.groups N) M)
      (c.first.buffers.output 1) (c.first.buffers.output 2) after)
    (schedule c.last.buffers) after (c.first.buffers.output k) owned (fun j => c.preserveScratch j k) i).trans
  rw [c.first.buffers.owned k]
  exact Execution.run_readback c.first.buffers api firstValues
    (schedule c.first.buffers) (schedule_covered c.first.buffers) m k i

theorem run_read (ops : Ops α) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store)
    (k : Fin 3) (i : Fin (outputs c k).count) :
    logical api neg (run ops p api c m) (outputs c k).tensor i.val=nativeValues ops p api neg c m k i := by
  let after := KernelMemory.run c.first.buffers api
    (Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m) m
  fin_cases k
  · simp only [outputs] at i ⊢
    rw [dense_logical api neg _ _ _ i.isLt]
    have preserved := Execution.run_input c.last.buffers api
      (Execution.finalValues ops p.final api c.last (min (Geometry.groups N) M)
        (c.first.buffers.output 1) (c.first.buffers.output 2)
        (KernelMemory.run c.first.buffers api
          (Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m) m))
      (schedule c.last.buffers) after (c.first.buffers.output 0) c.sameOwner c.preserveDX i
    apply preserved.trans
    rw [c.first.buffers.owned 0]
    have read := Execution.run_readback c.first.buffers api
      (Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m)
      (schedule c.first.buffers) (schedule_covered c.first.buffers) m 0 i
    apply read.trans
    simp only [Execution.groupValues,nativeValues,initialStats_load,row_load,vector_load]
  · simp only [outputs] at i ⊢
    rw [dense_logical api neg _ _ _ i.isLt,c.last.buffers.owned 0]
    have read := Execution.run_readback c.last.buffers api
      (Execution.finalValues ops p.final api c.last (min (Geometry.groups N) M)
        (c.first.buffers.output 1) (c.first.buffers.output 2)
        (KernelMemory.run c.first.buffers api
          (Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m) m))
      (schedule c.last.buffers) (schedule_covered c.last.buffers) after 0 i
    apply read.trans
    have h := Execution.final_scratch_values ops p.base p.final api c.first c.last
      x.dense w.dense dy.dense r.dense mu.dense m (schedule c.first.buffers) (schedule_covered c.first.buffers) i
    apply h.trans
    simp only [nativeValues,initialStats_load,row_load]
  · simp only [outputs] at i ⊢
    rw [dense_logical api neg _ _ _ i.isLt,c.last.buffers.owned 1]
    have read := Execution.run_readback c.last.buffers api
      (Execution.finalValues ops p.final api c.last (min (Geometry.groups N) M)
        (c.first.buffers.output 1) (c.first.buffers.output 2)
        (KernelMemory.run c.first.buffers api
          (Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m) m))
      (schedule c.last.buffers) (schedule_covered c.last.buffers) after 1 i
    apply read.trans
    have h := Execution.final_scratch_bias_values ops p.base p.final api c.first c.last
      x.dense w.dense dy.dense r.dense mu.dense m (schedule c.first.buffers) (schedule_covered c.first.buffers) i
    apply h.trans
    simp only [nativeValues,initialStats_load,row_load]

def realValues (c : KernelContract.Backward M N x w dy r mu) (m : World (Nat → ℝ)) :
    ∀ k,Fin (outputs c k).count → ℝ
  | ⟨0,_⟩,i => RealProof.Spec.gradient N
      (KernelForward.row Memory.realAPI Neg.neg x m N (i.val/N))
      (KernelForward.row Memory.realAPI Neg.neg dy m N (i.val/N)) (inputValue Memory.realAPI Neg.neg w m)
      (initialStats Memory.realAPI Neg.neg r mu m (i.val/N)).mean
      (initialStats Memory.realAPI Neg.neg r mu m (i.val/N)).inverse (i.val%N)
  | ⟨1,_⟩,i => RealProof.Spec.weightGradient N M
      (KernelForward.row Memory.realAPI Neg.neg x m N) (KernelForward.row Memory.realAPI Neg.neg dy m N)
      (initialStats Memory.realAPI Neg.neg r mu m) i.val
  | ⟨2,_⟩,i => RealProof.Spec.biasGradient N M (KernelForward.row Memory.realAPI Neg.neg dy m N) i.val

theorem nativeValues_real (p : KernelGraph.BackwardPlan N B)
    (c : KernelContract.Backward M N x w dy r mu) (m : World (Nat → ℝ)) :
    nativeValues RealProof.ops p Memory.realAPI Neg.neg c m=realValues c m := by
  funext k i
  fin_cases k
  · exact RealProof.backward_real p.base _ _ _ _ _
  · exact RealProof.weightGradient_real p.base p.final M (Geometry.groups N) (Geometry.groups_bounds N).1
      _ _ _ _ (by simpa only [outputs,c.last.dwCount] using i.isLt)
  · exact RealProof.biasGradient_real p.base p.final M (Geometry.groups N) (Geometry.groups_bounds N).1
      _ _ _ _ (by simpa only [outputs,c.last.dbCount] using i.isLt)

def Outside (c : KernelContract.Backward M N x w dy r mu) (key : Key) : Prop :=
  KernelMemory.Outside c.first.buffers key ∧ KernelMemory.Outside c.last.buffers key

theorem run_outside (ops : Ops α) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API α Store Byte) (c : KernelContract.Backward M N x w dy r mu)
    (m : World Store) (key : Key) (outside : Outside c key) :
    observe api (run ops p api c m) key=observe api m key := by
  exact (KernelMemory.run_outside c.last.buffers api _ _ key outside.2).trans
    (KernelMemory.run_outside c.first.buffers api _ m key outside.1)

theorem executes (ops : Ops α) (p : KernelGraph.BackwardPlan N B)
    (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Backward M N x w dy r mu) (m : World Store) :
    ∃ result,exec ops p api c m=.ok result ∧ result=final ops p api c m ∧
      (∀ k i,logical api neg result (outputs c k).tensor i.val=
        if KernelContract.launched M N then nativeValues ops p api neg c m k i
        else logical api neg m (outputs c k).tensor i.val) ∧
      (KernelContract.launched M N=Bool.false → result=m) ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.first.buffers.output k).tensor i.val=
          Execution.groupValues ops p.base api c.first x.dense w.dense dy.dense r.dense mu.dense m k i) ∧
      (∀ key,Outside c key → observe api result key=observe api m key) ∧
      (∀ key,caller [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor] key →
        observe api result key=observe api m key) := by
  refine ⟨final ops p api c m,rfl,rfl,?_,?_,?_,?_,?_⟩
  · intro k i
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [final,h,if_true]
      exact run_read ops p api neg c m k i
    · have hf : KernelContract.launched M N=Bool.false := Bool.eq_false_iff.mpr h
      simp only [final,hf,Bool.false_eq_true,if_false]
  · intro hf; simp only [final,hf,Bool.false_eq_true,if_false]
  · intro ht k i
    simp only [final,ht,if_true]
    exact run_first_read ops p api neg c m k i
  · intro key hk
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [final,h,if_true]
      exact run_outside ops p api c m key hk
    · simp [final,h]
  · intro key hk
    by_cases h : KernelContract.launched M N=Bool.true
    · simp only [final,h,if_true]
      exact Execution.twoLaunch_frame ops p.base p.final api c.first c.last
        x.dense w.dense dy.dense r.dense mu.dense m (schedule c.first.buffers) (schedule c.last.buffers) key hk
    · simp [final,h]

#axiomsClean run_read
#axiomsClean run_first_read
#axiomsClean nativeValues_real
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelBackward

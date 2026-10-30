import bench.repairs.tritonbench_g.block_sparse_attn.Kernels
import bench.repairs.tritonbench_g.flash_attn.Proofs.Physical
namespace VeriTile.Bench.Repairs.BlockSparseAttn
set_option maxHeartbeats 4000000

def Readonly (initial current : Memory α) : Prop :=
  ∀ r a,r≠.o → current r a=initial r a

theorem row_readonly (x : Layout) (p : Ops α) (initial current : Memory α) (scale : α)
    (w : Work) (same : Readonly initial current) :
    rowValue x p current scale w=rowValue x p initial scale w := by
  have hq : current .q=initial .q := funext (fun a=>same .q a (by decide))
  have hk : current .k=initial .k := funext (fun a=>same .k a (by decide))
  have hv : current .v=initial .v := funext (fun a=>same .v a (by decide))
  have hscore : score x p current scale=score x p initial scale := by
    funext b h row visit j
    simp only [score,scoreDot,dot,qLoad,kLoad,hq,hk]
  have hvalue : vLoad x p current=vLoad x p initial := by
    funext b h row col
    simp only [vLoad,hv]
  have eq : blockStep x p current scale=blockStep x p initial scale := by
    funext b h row col acc visit
    simp only [blockStep,hscore,hvalue]
  simp only [rowValue,eq]

theorem step_readonly (x : Layout) (p : Ops α) (scale : α) (w : Work)
    (initial current : Memory α) (same : Readonly initial current) :
    Readonly initial (step x p scale w current) := by
  intro r a ho
  simp [step,FlashAttn.store,ho,same r a ho]

theorem execute_readonly (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (initial current : Memory α) (same : Readonly initial current) :
    Readonly initial (execute x p scale ws current) := by
  induction ws generalizing current with
  | nil=>exact same
  | cons w ws ih=>exact ih _ (step_readonly x p scale w initial current same)

theorem execute_frame (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (m : Memory α) (r : FlashAttn.Region) (a : Nat)
    (outside : ∀ w∈ws,(r,a)≠(.o,w.target x.shape)) : execute x p scale ws m r a=m r a := by
  induction ws generalizing m with
  | nil=>rfl
  | cons w ws ih=>
    change execute x p scale ws (step x p scale w m) r a=_
    rw [ih _ (fun v hv=>outside v (by simp [hv]))]
    simp [step,FlashAttn.store,outside w (by simp)]

theorem written_value (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (initial current : Memory α) (same : Readonly initial current) (w : Work)
    (member : w∈ws) (unique : ∀ v∈ws,v.target x.shape=w.target x.shape → v=w) :
    execute x p scale ws current .o (w.target x.shape)=rowValue x p initial scale w := by
  induction ws generalizing current with
  | nil=>simp at member
  | cons v ws ih=>
    change execute x p scale ws (step x p scale v current) _ _=_
    by_cases tail : w∈ws
    · exact ih _ (step_readonly x p scale v initial current same) tail
        (fun u hu=>unique u (by simp [hu]))
    · have eq : v=w := by simpa [tail,eq_comm] using member
      subst v
      rw [execute_frame]
      · simp [step,FlashAttn.store,row_readonly x p initial current scale w same]
      · intro u hu he
        have hh : w.target x.shape=u.target x.shape := Prod.mk.inj he |>.2
        exact tail ((unique u (by simp [hu]) hh.symm) ▸ hu)

abbrev WorkIndex (s : Shape) := Fin s.batches × Fin s.heads × Fin s.length × Fin s.dim
def toWork (s : Shape) (i : WorkIndex s) : Work := ⟨i.1,i.2.1,i.2.2.1,i.2.2.2⟩
noncomputable def schedule (s : Shape) : List Work :=
  (Finset.univ : Finset (WorkIndex s)).toList.map (toWork s)

theorem schedule_mem (s : Shape) (w : Work) : w∈schedule s ↔ w.active s := by
  simp only [schedule,List.mem_map,Finset.mem_toList,Finset.mem_univ,true_and]
  constructor
  · rintro ⟨i,rfl⟩;exact ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.isLt⟩
  · rintro ⟨hb,hh,hn,hd⟩
    exact ⟨(⟨w.batch,hb⟩,⟨w.head,hh⟩,⟨w.row,hn⟩,⟨w.col,hd⟩),rfl⟩

theorem toWork_injective (s : Shape) : Function.Injective (toWork s) := by
  rintro ⟨a,b,c,d⟩ ⟨e,f,g,h⟩ he
  simp only [toWork,Work.mk.injEq] at he
  have aeq : a=e := Fin.ext he.1
  have beq : b=f := Fin.ext he.2.1
  have ceq : c=g := Fin.ext he.2.2.1
  have deq : d=h := Fin.ext he.2.2.2
  simp [aeq,beq,ceq,deq]

theorem schedule_nodup (s : Shape) : (schedule s).Nodup :=
  (List.nodup_map_iff (toWork_injective s)).mpr (Finset.nodup_toList _)

def Complete (s : Shape) (ws : List Work) : Prop := ws.Perm (schedule s)
def footprint (s : Shape) (r : FlashAttn.Region) (a : Nat) : Prop :=
  ∃ w : Work,w.active s ∧ (r,a)=(.o,w.target s)

theorem call_value (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (complete : Complete x.shape ws) (m : Memory α) (w : Work) (active : w.active x.shape) :
    execute x p scale ws m .o (w.target x.shape)=rowValue x p m scale w := by
  apply written_value x p scale ws m m (fun _ _ _=>rfl) w
  · exact complete.mem_iff.mpr ((schedule_mem x.shape w).mpr active)
  · intro v hv he
    exact target_injective x.shape v w ((schedule_mem x.shape v).mp (complete.mem_iff.mp hv)) active he

theorem call_frame (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (complete : Complete x.shape ws) (m : Memory α) (r : FlashAttn.Region) (a : Nat)
    (outside : ¬footprint x.shape r a) : execute x p scale ws m r a=m r a := by
  apply execute_frame
  intro w hw he
  exact outside ⟨w,(schedule_mem x.shape w).mp (complete.mem_iff.mp hw),he⟩

#axiomsClean row_readonly
#axiomsClean execute_readonly
#axiomsClean written_value
#axiomsClean schedule_mem
#axiomsClean schedule_nodup
#axiomsClean call_value
#axiomsClean call_frame
end VeriTile.Bench.Repairs.BlockSparseAttn

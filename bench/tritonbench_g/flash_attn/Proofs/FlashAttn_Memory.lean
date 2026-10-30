import bench.tritonbench_g.flash_attn.Proofs.Kernels
namespace VeriTile.Bench.Repairs.FlashAttn
set_option maxHeartbeats 4000000

def Readonly (initial current : Memory α) : Prop :=
  ∀ r a,r≠.o → r≠.l → current r a=initial r a

theorem result_readonly (x : Layout) (p : Ops α) (initial current : Memory α)
    (scale : α) (w : Work) (h : Readonly initial current) :
    w.result x p current scale=w.result x p initial scale := by
  have hq : current .q=initial .q := funext (fun a=>h .q a (by decide) (by decide))
  have hk : current .k=initial .k := funext (fun a=>h .k a (by decide) (by decide))
  have hv : current .v=initial .v := funext (fun a=>h .v a (by decide) (by decide))
  have stepEq : blockStep x p current scale=blockStep x p initial scale := by
    funext b h row col acc block
    simp only [blockStep,score,dot,query,key,value,logical,hq,hk,hv]
  cases w <;> simp only [Work.result,outputValue,logValue,accumulated,stepEq]

theorem step_readonly (x : Layout) (p : Ops α) (scale : α) (w : Work)
    (initial current : Memory α) (h : Readonly initial current) :
    Readonly initial (step x p scale w current) := by
  intro r a ho hl
  cases w <;> simp [step,store,Work.target,ho,hl,h r a ho hl]

theorem execute_readonly (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (initial current : Memory α) (h : Readonly initial current) :
    Readonly initial (execute x p scale ws current) := by
  induction ws generalizing current with
  | nil => exact h
  | cons w ws ih => exact ih _ (step_readonly x p scale w initial current h)

theorem step_frame (x : Layout) (p : Ops α) (scale : α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (h : (r,a)≠w.target x) :
    step x p scale w m r a=m r a := by simp [step,store,h]

theorem execute_frame (x : Layout) (p : Ops α) (scale : α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (h : ∀ w∈ws,(r,a)≠w.target x) :
    execute x p scale ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change execute x p scale ws (step x p scale w m) r a=_
    rw [ih _ (fun v hv=>h v (by simp [hv]))]
    exact step_frame x p scale w m r a (h w (by simp))

/-- Pure last-writer interpretation; every written value uses initial input memory.
It is derived from current-memory execution below, not assumed at the call boundary. -/
def initialWrites (x : Layout) (p : Ops α) (scale : α) (initial : Memory α)
    (ws : List Work) (target : Region × Nat) (seed : α) : α :=
  ws.foldl (fun old w=>if target=w.target x then w.result x p initial scale else old) seed

theorem execute_initialWrites (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (initial current : Memory α) (same : Readonly initial current) (r : Region) (a : Nat) :
    execute x p scale ws current r a=initialWrites x p scale initial ws (r,a) (current r a) := by
  induction ws generalizing current with
  | nil => rfl
  | cons w ws ih =>
    change execute x p scale ws (step x p scale w current) r a=_
    rw [ih _ (step_readonly x p scale w initial current same)]
    simp only [initialWrites,List.foldl_cons,step,store,result_readonly x p initial current scale w same]

theorem written_value (x : Layout) (p : Ops α) (scale : α) (ws : List Work)
    (initial current : Memory α) (same : Readonly initial current) (w : Work)
    (member : w∈ws) (unique : ∀ v∈ws,v.target x=w.target x → v=w) :
    execute x p scale ws current (w.target x).1 (w.target x).2=w.result x p initial scale := by
  induction ws generalizing current with
  | nil => simp at member
  | cons v ws ih =>
    change execute x p scale ws (step x p scale v current) _ _=_
    by_cases tail : w∈ws
    · exact ih _ (step_readonly x p scale v initial current same) tail
        (fun u hu=>unique u (by simp [hu]))
    · have eq : v=w := by simpa [tail,eq_comm] using member
      subst v
      rw [execute_frame]
      · simp [step,store,result_readonly x p initial current scale w same]
      · intro u hu he
        have uw:=unique u (by simp [hu]) he.symm
        exact tail (uw ▸ hu)

#axiomsClean result_readonly
#axiomsClean execute_readonly
#axiomsClean execute_frame
#axiomsClean execute_initialWrites
#axiomsClean written_value
end VeriTile.Bench.Repairs.FlashAttn

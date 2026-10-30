import bench.repairs.tritonbench_g.streamk_matmul.Proofs.ByteExecution

namespace VeriTile.Bench.Repairs.StreamK.Checked.ValueMemory
open TensorBinding Whole001 Source
noncomputable section
set_option maxHeartbeats 4000000

def render (read : ValueRead α) (base : Nat) (values : Fin n → α) : ValueRead α := fun address =>
  if h : base≤address ∧ address<base+4*n ∧ (address-base)%4=0 then
    values ⟨(address-base)/4,by omega⟩ else read address

theorem render_at (read : ValueRead α) (base : Nat) (values : Fin n → α) (i : Fin n) :
    render read base values (base+4*i.val)=values i := by
  have hi := i.isLt
  have inside : base≤base+4*i.val ∧ base+4*i.val<base+4*n ∧ (base+4*i.val-base)%4=0 := by omega
  simp only [render,dif_pos inside]
  congr 1
  apply Fin.ext
  simp only [Fin.val_mk]
  omega

theorem initial (read : ValueRead α) (base : Nat) (values : Fin n → α)
    (bound : ∀ i,values i=read (base+4*i.val)) : render read base values=read := by
  funext address
  unfold render
  split
  next inside =>
    rw [bound]
    congr 1
    simp only [Fin.val_mk]
    omega
  next => rfl

theorem update (read : ValueRead α) (base : Nat) (values : Fin n → α) (i : Fin n) (value : α) :
    render read base (Function.update values i value)=Function.update (render read base values) (base+4*i.val) value := by
  funext address
  by_cases hit : address=base+4*i.val
  · subst address
    rw [render_at,Function.update_self,Function.update_self]
  · rw [Function.update_of_ne hit]
    unfold render
    split
    next inside =>
      have ne : (⟨(address-base)/4,by omega⟩ : Fin n)≠i := by
        intro same
        have hv := congrArg Fin.val same
        change (address-base)/4=i.val at hv
        apply hit
        omega
      rw [Function.update_of_ne ne]
    next => rfl

theorem frame (read : ValueRead α) (base : Nat) (values : Fin n → α) (address : Nat)
    (outside : address<base ∨ base+4*n≤address) : render read base values address=read address := by
  have miss : ¬(base≤address ∧ address<base+4*n ∧ (address-base)%4=0) := by omega
  simp only [render,dif_neg miss]

theorem update_other (read : ValueRead α) (base other : Nat) (values : Fin n → α) (i : Fin m) (value : α)
    (separate : base+4*n≤other ∨ other+4*m≤base) :
    render (Function.update read (other+4*i.val) value) base values=
      Function.update (render read base values) (other+4*i.val) value := by
  funext address
  have hi := i.isLt
  by_cases hit : address=other+4*i.val
  · subst address
    rw [Function.update_self,frame _ _ _ _ (by omega),Function.update_self]
  · rw [Function.update_of_ne hit]
    unfold render
    split
    · rfl
    · rw [Function.update_of_ne hit]

def rendered (read : ValueRead α) (v : ByteExecution.Layout p) (s : State α p exterior) : ValueRead α :=
  render (if v.scratchActive then render read v.scratchBase s.scratch else read) v.outputBase s.output

abbrev Machine (α : Type) (p : Parameters) (exterior : Nat) := State α p exterior × ValueRead α

def step (v : ByteExecution.Layout p) (add : α → α → α) (e : Event α p) (t : Machine α p exterior) : Machine α p exterior :=
  match e with
  | .storeOutput i value => (Memory.step add e t.1,Function.update t.2 (v.outputBase+4*i.val) (value t.1.inputA t.1.inputB))
  | .storeScratch i value => (Memory.step add e t.1,Function.update t.2 (v.scratchBase+4*i.val) (value t.1.inputA t.1.inputB))
  | .atomic i j => (Memory.step add e t.1,Function.update t.2 (v.outputBase+4*i.val)
      (add (t.2 (v.outputBase+4*i.val)) (t.2 (v.scratchBase+4*j.val))))

def Allowed (v : ByteExecution.Layout p) : Event α p → Prop
  | .storeOutput _ _ => True
  | .storeScratch _ _ | .atomic _ _ => v.scratchActive=Bool.true

theorem step_render (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v) (add : α → α → α)
    (e : Event α p) (allowed : Allowed v e) (read : ValueRead α) (s : State α p exterior) :
    step v add e (s,rendered read v s)=(Memory.step add e s,rendered read v (Memory.step add e s)) := by
  cases e with
  | storeOutput i value =>
    simp only [step,Memory.step,rendered,update]
  | storeScratch i value =>
    have active : v.scratchActive=Bool.true := allowed
    have separate : v.outputBase+4*(p.h.m*p.h.n)≤v.scratchBase ∨
        v.scratchBase+4*(p.h.programs*p.h.bm*p.h.bn)≤v.outputBase := by
      simpa [ByteMemory.Separate,ByteExecution.cView,ByteExecution.sView,active] using laws.separate
    simp only [step,Memory.step,rendered,active,ite_true,update,update_other _ _ _ _ i _ separate]
  | atomic i j =>
    have active : v.scratchActive=Bool.true := allowed
    have separate : v.outputBase+4*(p.h.m*p.h.n)≤v.scratchBase ∨
        v.scratchBase+4*(p.h.programs*p.h.bm*p.h.bn)≤v.outputBase := by
      simpa [ByteMemory.Separate,ByteExecution.cView,ByteExecution.sView,active] using laws.separate
    have hj := j.isLt
    have sr : render (render read v.scratchBase s.scratch) v.outputBase s.output (v.scratchBase+4*j.val)=s.scratch j := by
      rw [frame _ _ _ _ (by omega),render_at]
    simp only [step,Memory.step,rendered,active,ite_true,render_at,sr,update]

def run (v : ByteExecution.Layout p) (add : α → α → α) (events : List (Event α p)) : Machine α p exterior → Machine α p exterior :=
  events.foldl (fun current event => step v add event current)

theorem run_render (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v) (add : α → α → α)
    (events : List (Event α p)) (allowed : ∀ e∈events,Allowed v e) (read : ValueRead α) (s : State α p exterior) :
    run v add events (s,rendered read v s)=(Memory.run add events s,rendered read v (Memory.run add events s)) := by
  induction events generalizing s with
  | nil => rfl
  | cons e tail ih =>
    simp only [run,List.foldl_cons,step_render v laws add e (allowed e (by simp))]
    exact ih (fun x hx => allowed x (by simp [hx])) _

def call (v : ByteExecution.Layout p) (ops : Instructions α)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) (s : Machine α p exterior) : Machine α p exterior :=
  if p.h.m=0 ∨ p.h.n=0 then s else
    run v ops.atomicAdd (Schedule.reorder (fullWave p ops) last)
      (run v ops.atomicAdd ((reduce.events.map Prod.snd).map fun pair => .atomic pair.1 pair.2)
        (run v ops.atomicAdd (Schedule.reorder (firstWave p ops) first) s))

theorem call_render (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v) (ops : Instructions α)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) (read : ValueRead α) (s : State α p exterior) :
    call v ops first reduce last (s,rendered read v s)=
      let final := Schedule.call p ops first reduce last s
      (final,rendered read v final) := by
  unfold call Schedule.call
  split
  · rfl
  · rw [run_render v laws _ _ (fun e he => ByteExecution.first_allowed v ops e ((Schedule.reorder_perm _ first).mem_iff.mp he)),
      run_render v laws _ _ (ByteExecution.reduce_allowed v reduce),
      run_render v laws _ _ (fun e he => ByteExecution.full_allowed v ops e ((Schedule.reorder_perm _ last).mem_iff.mp he))]

#axiomsClean call_render
#axiomsClean initial
#axiomsClean step_render
#axiomsClean run_render
end
end VeriTile.Bench.Repairs.StreamK.Checked.ValueMemory

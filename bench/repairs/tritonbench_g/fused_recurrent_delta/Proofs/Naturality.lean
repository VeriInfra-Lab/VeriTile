import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Naturality
noncomputable section
set_option maxHeartbeats 15000000
set_option maxRecDepth 20000
structure Hom {α β : Type} (p : Ops α) (q : Ops β) (F : α → β) : Prop where
  zero : F p.zero=q.zero
  one : F p.one=q.one
  add : ∀ a b,F (p.add a b)=q.add (F a) (F b)
  sub : ∀ a b,F (p.sub a b)=q.sub (F a) (F b)
  mul : ∀ a b,F (p.mul a b)=q.mul (F a) (F b)
  cut : ∀ a,F (p.cut a)=q.cut (F a)
def mem {α β : Type} (F : α → β) (m : Memory α) : Memory β := fun a => F (m a)
def machine {α β : Type} (F : α → β) (a : Machine α) : Machine β :=
  ⟨fun i j => F (a.h i j),mem F a.mem⟩
variable {α β : Type} {p : Ops α} {q : Ops β} {F : α → β} (h : Hom p q F)
include h
theorem tree (tr : Tree) (x : Nat → α) : F (tr.eval p x)=tr.eval q (fun i => F (x i)) := by
  induction tr <;> simp_all [Tree.eval,h.zero,h.add]
theorem row (m : Memory α) (r : Region) (plane t lo hi : Nat) (v : Nat → α) :
    mem F (rowStore p m r plane t lo hi v)=rowStore q (mem F m) r plane t lo hi (fun j => F (v j)) := by
  funext a
  simp only [mem,rowStore,write]
  split <;> simp [h.cut]
theorem state (s : Shape) (m : Memory α) (r : Region) (b iv BV : Nat) (v : Nat → Nat → α) :
    mem F (stateStore s p m r b iv BV v)=stateStore s q (mem F m) r b iv BV (fun i j => F (v i j)) := by
  funext a
  simp only [mem,stateStore,write]
  split <;> simp [h.cut]
theorem load (m : Memory α) (r : Region) (b t j : Nat) (active : Prop) [Decidable active] :
    F (FusedRecurrentDelta.load p m r b t j active)=FusedRecurrentDelta.load q (mem F m) r b t j active := by
  unfold FusedRecurrentDelta.load
  split <;> simp [mem,h.zero]
theorem key (s : Shape) (m : Memory α) (r : Region) (b t j : Nat) :
    F (keyLoad s p m r b t j)=keyLoad s q (mem F m) r b t j := load h _ _ _ _ _ _
theorem val (s : Shape) (m : Memory α) (r : Region) (b t iv BV j : Nat) :
    F (valLoad s p m r b t iv BV j)=valLoad s q (mem F m) r b t iv BV j := load h _ _ _ _ _ _
theorem beta (s : Shape) (f : Flags) (m : Memory α) (b t iv BV j : Nat) :
    F (betaLoad s f p m b t iv BV j)=betaLoad s f q (mem F m) b t iv BV j := by
  unfold betaLoad
  split <;> simp [val h,mem]
theorem seed (s : Shape) (m : Memory α) (r : Region) (use : Bool) (b iv BV i j : Nat) :
    F (FusedRecurrentDelta.seed s p m r use b iv BV i j)=FusedRecurrentDelta.seed s q (mem F m) r use b iv BV i j := by
  unfold FusedRecurrentDelta.seed
  split <;> simp [h.add,h.zero,load h]

theorem forward_step (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv t : Nat) (st : Machine α) :
    machine F (forwardStep s f p g scale b iv t st)=forwardStep s f q g (F scale) b iv t (machine F st) := by
  dsimp only [forwardStep,machine]
  simp only [row h,h.add,h.mul,h.sub,tree h,key h,val h,beta h]
theorem reverse_step (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv t : Nat) (st : Machine α) :
    machine F (reverseStep s f p g scale b iv t st)=reverseStep s f q g (F scale) b iv t (machine F st) := by
  dsimp only [reverseStep,machine]
  cases hf:f.vectorBeta <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true,row h,h.add,h.mul,h.sub,tree h,key h,val h,beta h]
theorem recompute_step (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv t : Nat) (st : Machine α) :
    machine F (recomputeStep s f p g scale b iv t st)=recomputeStep s f q g (F scale) b iv t (machine F st) := by
  dsimp only [recomputeStep,machine]
  simp only [row h,h.add,h.mul,h.sub,tree h,key h,val h,beta h]

theorem forward_loop (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv n : Nat) (st : Machine α) :
    machine F (forwardLoop s f p g scale b iv n st)=forwardLoop s f q g (F scale) b iv n (machine F st) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [forwardLoop,forward_step h,ih];rfl
theorem reverse_loop (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv n : Nat) (st : Machine α) :
    machine F (reverseLoop s f p g scale b iv n st)=reverseLoop s f q g (F scale) b iv n (machine F st) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [reverseLoop,reverse_step h,ih];rfl
theorem recompute_loop (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv n : Nat) (st : Machine α) :
    machine F (recomputeLoop s f p g scale b iv n st)=recomputeLoop s f q g (F scale) b iv n (machine F st) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [recomputeLoop,recompute_step h,ih];rfl

theorem seed_machine (s : Shape) (m : Memory α) (r : Region) (use : Bool) (b iv BV : Nat) :
    machine F ⟨FusedRecurrentDelta.seed s p m r use b iv BV,m⟩=
      ⟨FusedRecurrentDelta.seed s q (mem F m) r use b iv BV,mem F m⟩ := by
  unfold machine
  congr 1
  funext i j
  exact seed h _ _ _ _ _ _ _ _ _
theorem forward_program (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv : Nat) (m : Memory α) :
    mem F (forwardProgram s f p g scale b iv m)=forwardProgram s f q g (F scale) b iv (mem F m) := by
  have hh:=forward_loop h s f g scale b iv s.T ⟨FusedRecurrentDelta.seed s p m .h0 f.initial b iv s.FBV,m⟩
  rw [seed_machine h] at hh
  unfold forwardProgram
  split
  · rw [state h]
    have hm:=congrArg Machine.mem hh
    have hv:=congrArg Machine.h hh
    dsimp only [machine] at hm hv
    rw [hm,hv]
  · exact congrArg Machine.mem hh

theorem backward_program (s : Shape) (f : Flags) (g : Trees) (scale : α) (b iv : Nat) (m : Memory α) :
    mem F (backwardProgram s f p g scale b iv m)=backwardProgram s f q g (F scale) b iv (mem F m) := by
  have hr:=reverse_loop h s f g scale b iv s.T ⟨FusedRecurrentDelta.seed s p m .dht f.terminalGrad b iv s.BBV,m⟩
  rw [seed_machine h] at hr
  let mid:=if f.initialGrad then stateStore s p (reverseLoop s f p g scale b iv s.T ⟨FusedRecurrentDelta.seed s p m .dht f.terminalGrad b iv s.BBV,m⟩).mem .dh0 b iv s.BBV
    (reverseLoop s f p g scale b iv s.T ⟨FusedRecurrentDelta.seed s p m .dht f.terminalGrad b iv s.BBV,m⟩).h
    else (reverseLoop s f p g scale b iv s.T ⟨FusedRecurrentDelta.seed s p m .dht f.terminalGrad b iv s.BBV,m⟩).mem
  have hm : mem F mid=(if f.initialGrad then stateStore s q
      (reverseLoop s f q g (F scale) b iv s.T ⟨FusedRecurrentDelta.seed s q (mem F m) .dht f.terminalGrad b iv s.BBV,mem F m⟩).mem .dh0 b iv s.BBV
      (reverseLoop s f q g (F scale) b iv s.T ⟨FusedRecurrentDelta.seed s q (mem F m) .dht f.terminalGrad b iv s.BBV,mem F m⟩).h
    else (reverseLoop s f q g (F scale) b iv s.T ⟨FusedRecurrentDelta.seed s q (mem F m) .dht f.terminalGrad b iv s.BBV,mem F m⟩).mem) := by
    dsimp only [mid]
    split
    · rw [state h]
      have hm:=congrArg Machine.mem hr
      have hv:=congrArg Machine.h hr
      dsimp only [machine] at hm hv
      rw [hm,hv]
    · exact congrArg Machine.mem hr
  have hc:=recompute_loop h s f g scale b iv s.T ⟨FusedRecurrentDelta.seed s p mid .h0 f.initial b iv s.BBV,mid⟩
  rw [seed_machine h,hm] at hc
  exact congrArg Machine.mem hc

omit h in
theorem fold (step : (Nat×Nat) → Memory α → Memory α) (step' : (Nat×Nat) → Memory β → Memory β)
    (hf : ∀ job m,mem F (step job m)=step' job (mem F m)) (jobs : List (Nat×Nat)) (m : Memory α) :
    mem F (jobs.foldl (fun m j => step j m) m)=jobs.foldl (fun m j => step' j m) (mem F m) := by
  induction jobs generalizing m with
  | nil => rfl
  | cons j js ih => simpa only [List.foldl_cons,hf] using ih (step j m)
theorem forward_grid (s : Shape) (f : Flags) (g : Trees) (scale : α) (m : Memory α) :
    mem F (forwardGrid s f p g scale m)=forwardGrid s f q g (F scale) (mem F m) :=
  fold _ _ (fun j m => forward_program h s f g scale j.1 j.2 m) _ m
theorem backward_grid (s : Shape) (f : Flags) (g : Trees) (scale : α) (m : Memory α) :
    mem F (backwardGrid s f p g scale m)=backwardGrid s f q g (F scale) (mem F m) :=
  fold _ _ (fun j m => backward_program h s f g scale j.1 j.2 m) _ m

theorem prepare_beta (s : Shape) (f : Flags) (m : Memory α) :
    mem F (prepareBeta s f p m)=prepareBeta s f q (mem F m) := by
  funext a
  cases hf:f.defaultBeta <;> simp [mem,prepareBeta,hf,write,apply_ite,h.one] <;> split <;> simp_all
theorem prepare_dbeta (s : Shape) (f : Flags) (m : Memory α) :
    mem F (prepareDbeta s f p m)=prepareDbeta s f q (mem F m) := by
  funext a
  cases hf:f.vectorBeta <;> simp [mem,prepareDbeta,hf,write,apply_ite,h.zero] <;> split <;> simp_all
theorem host (s : Shape) (f : Flags) (g : Trees) (m : Memory α) :
    mem F (hostReduce s f p g m)=hostReduce s f q g (mem F m) := by
  funext a
  rcases a with ⟨r,b,t,v⟩
  cases hf:f.vectorBeta <;> cases r <;> simp [mem,hostReduce,hf,write,apply_ite,h.cut,h.zero,tree h] <;> split <;> simp_all

theorem forward_call (s : Shape) (f : Flags) (g : Trees) (scale : α) (m : Memory α) :
    mem F (forwardCall s f p g scale m)=forwardCall s f q g (F scale) (mem F m) := by
  unfold forwardCall
  rw [forward_grid h,prepare_beta h]
theorem backward_call (s : Shape) (f : Flags) (g : Trees) (scale : α) (m : Memory α) :
    mem F (backwardCall s f p g scale m)=backwardCall s f q g (F scale) (mem F m) := by
  unfold backwardCall
  rw [host h,backward_grid h,prepare_dbeta h]

#axiomsClean forward_call
#axiomsClean backward_call
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Naturality

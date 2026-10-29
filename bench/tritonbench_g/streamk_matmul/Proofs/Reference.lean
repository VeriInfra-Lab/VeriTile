import bench.tritonbench_g.streamk_matmul.Proofs.Control

/-! Independent initial-memory expression reference. It contains no program
counter, scheduler, numeric accumulator or source execution. Dot/add/cast/RMW
nodes retain the exact operation order. Every read is substituted with the
expression currently stored at its address, so arbitrary data aliases are
preserved. Events contain addresses/names only, never supplied numeric values. -/
namespace VeriTile.Bench.TritonBenchG.StreamK
inductive Recipe where
 | initial (address : Address)
 | zero
 | add (a b : Recipe)
 | half (v : Recipe)
 | output (v : Recipe)
 | atomicAdd (a b : Recipe)
 | dot (m n k : Nat) (a : Fin (m*k)→Recipe) (b : Fin (k*n)→Recipe) (lane : Fin (m*n))

def denote (b : Backend V) (initial : Address→V) : Recipe→V
 | .initial a => initial a
 | .zero => b.zero
 | .add a d => b.add (denote b initial a) (denote b initial d)
 | .half a => b.half (denote b initial a)
 | .output a => b.output (denote b initial a)
 | .atomicAdd a d => b.atomicAdd (denote b initial a) (denote b initial d)
 | .dot m n k a d i => b.dot m n k (fun j=>denote b initial (a j)) (fun j=>denote b initial (d j)) i

structure Reference where
 data : Address→Recipe
 locks : Address→I32
 regs : Nat→String→Nat→Recipe

def referenceInitial (locks : Address→I32) : Reference :=
 ⟨Recipe.initial,locks,fun _ _ _=>.zero⟩

def observe (c : Config) (e : Event) (r : Reference) : Reference :=
 match e with
 | .silent => r
 | .reset p n => {r with regs:=put r.regs p (put (r.regs p) n (fun _=>.zero))}
 | .load p n i a => {r with regs:=put r.regs p (put (r.regs p) n (put (r.regs p n) i (r.data a)))}
 | .mac p n a d =>
   {r with regs:=put r.regs p (put (r.regs p) n (fun i=>if h:i<c.bm*c.bn then
     .add (r.regs p n i) (.dot c.bm c.bn c.bk (fun j=>r.regs p a j) (fun j=>r.regs p d j) ⟨i,h⟩)
     else r.regs p n i))}
 | .half p n v => {r with regs:=put r.regs p (put (r.regs p) n (fun i=>.half (r.regs p v i)))}
 | .store p n i a => {r with data:=put r.data a (.output (r.regs p n i))}
 | .atomicAdd p n i a => {r with data:=put r.data a (.atomicAdd (r.data a) (.output (r.regs p n i)))}
 | .exchange a v => {r with locks:=put r.locks a v}
 | .cas a cmp rep => {r with locks:=put r.locks a (if r.locks a==cmp then rep else r.locks a)}

def replay (c : Config) (events : List Event) (r : Reference) : Reference :=
 events.foldl (fun r e=>observe c e r) r

structure Related (b : Backend V) (mem : Address→V) (s : State V) (r : Reference) : Prop where
 data : ∀ a, s.mem.data a=denote b mem (r.data a)
 locks : s.mem.locks=r.locks
 regs : ∀ p n i, s.regs p n i=denote b mem (r.regs p n i)

theorem related_initial (source : List Command) (b : Backend V) (mem : Memory V) :
 Related b mem.data (initial source b mem) (referenceInitial mem.locks) := by
 exact ⟨fun _=>rfl,rfl,fun _ _ _=>rfl⟩

theorem execute_related (c : Config) (b : Backend V) (mem : Address→V)
 (e : Event) (s : State V) (r : Reference) (h : Related b mem s r) :
 Related b mem (execute c b e s) (observe c e r) := by
 rcases h with ⟨hd,hl,hr⟩
 cases e with
 | silent => exact ⟨hd,hl,hr⟩
 | reset p n =>
   refine ⟨hd,hl,?_⟩
   intro q v i
   by_cases hp:q=p <;> by_cases hn:v=n <;> simp [execute,observe,put,hp,hn,denote,hr]
 | load p n i a =>
   refine ⟨hd,hl,?_⟩
   intro q v j
   by_cases hp:q=p <;> by_cases hn:v=n <;> by_cases hi:j=i <;>
     simp [execute,observe,put,hp,hn,hi,denote,hr,hd]
 | mac p n a d =>
   refine ⟨hd,hl,?_⟩
   intro q v i
   by_cases hp:q=p <;> by_cases hn:v=n <;> by_cases hi:i<c.bm*c.bn <;>
     simp [execute,observe,put,hp,hn,hi,denote,hr]
 | half p n v =>
   refine ⟨hd,hl,?_⟩
   intro q w i
   by_cases hp:q=p <;> by_cases hn:w=n <;> simp [execute,observe,put,hp,hn,denote,hr]
 | store p n i a =>
   refine ⟨?_,hl,hr⟩
   intro x; by_cases ha:x=a <;> simp [execute,observe,put,ha,denote,hr,hd]
 | atomicAdd p n i a =>
   refine ⟨?_,hl,hr⟩
   intro x; by_cases ha:x=a <;> simp [execute,observe,put,ha,denote,hr,hd]
 | exchange a v => exact ⟨hd,by simp only [execute,observe,hl],hr⟩
 | cas a cmp rep => exact ⟨hd,by simp only [execute,observe,hl],hr⟩

theorem step_related (c : Config) (b : Backend V) (mem : Address→V)
 (grid p : Nat) (s : State V) (r : Reference) (h : Related b mem s r) :
 Related b mem (step c b grid p s).2 (observe c (step c b grid p s).1 r) := by
 have he:=execute_related c b mem (decode c grid p s.mem.locks (s.control p)).1 s r h
 exact ⟨he.data,he.locks,he.regs⟩

theorem run_related (c : Config) (b : Backend V) (mem : Address→V)
 (grid : Nat) (schedule : List Nat) (s : State V) (r : Reference) (h : Related b mem s r) :
 Related b mem (run c b grid schedule s).2 (replay c (run c b grid schedule s).1 r) := by
 induction schedule generalizing s r with
 | nil => exact h
 | cons p ps ih =>
   simpa only [run,replay,List.foldl_cons] using
     ih (step c b grid p s).2 (observe c (step c b grid p s).1 r) (step_related c b mem grid p s r h)

/-- Every address absent from actual store/RMW events keeps its exact initial
bits. No non-alias premise or hypothesized final frame is needed. -/
theorem replay_frame (c : Config) (events : List Event) (r : Reference) (a : Address)
 (h : ∀ e∈events, writeAddress e≠some a) : (replay c events r).data a=r.data a := by
 induction events generalizing r with
 | nil => rfl
 | cons e es ih =>
   have ht:∀ e∈es,writeAddress e≠some a:=fun e he=>h e (List.mem_cons_of_mem _ he)
   change (replay c es (observe c e r)).data a=_
   rw [ih _ ht]
   have he:=h e (List.mem_cons_self ..)
   cases e <;> simp_all [writeAddress,observe,put,ne_comm]

def lockWriteAddress : Event→Option Address
 | .exchange a _ | .cas a _ _ => some a
 | _ => none

theorem replay_lock_frame (c : Config) (events : List Event) (r : Reference) (a : Address)
 (h : ∀ e∈events, lockWriteAddress e≠some a) : (replay c events r).locks a=r.locks a := by
 induction events generalizing r with
 | nil => rfl
 | cons e es ih =>
   have ht:∀ e∈es,lockWriteAddress e≠some a:=fun e he=>h e (List.mem_cons_of_mem _ he)
   change (replay c es (observe c e r)).locks a=_
   rw [ih _ ht]
   have he:=h e (List.mem_cons_self ..)
   cases e <;> simp_all [lockWriteAddress,observe,put,ne_comm]

#axiomsClean replay_lock_frame
#axiomsClean execute_related
#axiomsClean run_related
#axiomsClean replay_frame
#specNonCircular replay avoiding [run, step, execute]
end VeriTile.Bench.TritonBenchG.StreamK

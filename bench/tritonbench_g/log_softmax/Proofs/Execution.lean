import bench.tritonbench_g.log_softmax.Proofs.Address

/-! Concrete four-byte writes of the source graph. The dynamic machine reads
its current physical memory; separation proves agreement with initial inputs.
Any finite prefix, including one stopped by native failure, has the same byte
frame. No primitive API field asserts a whole-kernel result. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Execution
open Address
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical

def Disjoint (a : Dense p) (b : Dense q) : Prop :=
  a.base+p*4≤b.base ∨ b.base+q*4≤a.base

theorem disjoint_words (a : Dense p) (b : Dense q) (h : Disjoint a b)
    (i j : Nat) (hi : i<p) (hj : j<q) : Apart (a.address i) (b.address j) := by
  unfold Disjoint at h
  unfold Apart Dense.address
  omega

structure Launch (α : Type) (M N K C : Nat) where
  direction : Direction
  ops : Ops α
  plan : Plan N C
  first : Dense (M*N*K)
  second : Dense (M*N*K)
  output : Dense (M*N*K)
  firstApart : Disjoint output first
  secondApart : Disjoint output second

def rowRead (api : API α Store Byte) (a : Dense (M*N*K)) (m : Store)
    (q : Coord M N K) (j : Nat) : α :=
  api.read m (a.address (offset N K q.1.val (safe j N) q.2.2.val))

theorem rowRead_bound (q : Coord M N K) (j : Nat) :
    offset N K q.1.val (safe j N) q.2.2.val<M*N*K :=
  VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
    q.1.isLt (safe_bound (Nat.zero_lt_of_lt q.2.1.isLt)) q.2.2.isLt

def scalar (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (m : Store) (q : Coord M N K) : α :=
  value launch.direction launch.ops launch.plan (rowRead api launch.first m q)
    (rowRead api launch.second m q) q.2.1.val

def event (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (m : Store) (q : Coord M N K) : Write α :=
  ⟨launch.output.address (linear q),scalar api launch m q⟩

def step (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (m : Store) (q : Coord M N K) : Store :=
  api.write m (launch.output.address (linear q)) (scalar api launch m q)

def execute (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (m : Store) : Store :=
  order.foldl (step api launch) m

def stableInputs (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (before after : Store) : Prop :=
  (∀ q j,rowRead api launch.first after q j=rowRead api launch.first before q j) ∧
  (∀ q j,rowRead api launch.second after q j=rowRead api launch.second before q j)

theorem step_inputs (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (m : Store) (p : Coord M N K) : stableInputs api launch m (step api launch m p) := by
  constructor
  · intro q j
    apply api.read_apart
    exact disjoint_words _ _ launch.firstApart _ _ (linear_bound p) (rowRead_bound q j)
  · intro q j
    apply api.read_apart
    exact disjoint_words _ _ launch.secondApart _ _ (linear_bound p) (rowRead_bound q j)

theorem scalar_stable (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (before after : Store) (h : stableInputs api launch before after) (q : Coord M N K) :
    scalar api launch after q=scalar api launch before q := by
  unfold scalar
  rw [funext (h.1 q),funext (h.2 q)]

theorem inputs_trans (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (a b c : Store) (h : stableInputs api launch a b) (k : stableInputs api launch b c) :
    stableInputs api launch a c := by
  exact ⟨fun q j=>(k.1 q j).trans (h.1 q j),fun q j=>(k.2 q j).trans (h.2 q j)⟩

theorem execute_inputs (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (m : Store) : stableInputs api launch m (execute api launch order m) := by
  induction order generalizing m with
  | nil => exact ⟨fun _ _=>rfl,fun _ _=>rfl⟩
  | cons p ps ih =>
    exact inputs_trans api launch m _ _ (step_inputs api launch m p) (ih _)

theorem execute_static_aux (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (initial m : Store) (stable : stableInputs api launch initial m) :
    execute api launch order m=run api (order.map (event api launch initial)) m := by
  induction order generalizing m with
  | nil => rfl
  | cons p ps ih =>
    change execute api launch ps (step api launch m p)=
      run api (ps.map (event api launch initial)) (api.write m _ _)
    have valueEq:=scalar_stable api launch initial m stable p
    have next:=inputs_trans api launch initial m _ stable (step_inputs api launch m p)
    rw [ih _ next]
    simp only [step,event,valueEq]

theorem execute_static (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (m : Store) :
    execute api launch order m=run api (order.map (event api launch m)) m :=
  execute_static_aux api launch order m m ⟨fun _ _=>rfl,fun _ _=>rfl⟩

theorem execute_readback (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (m : Store) (q : Coord M N K) (present : q∈order) :
    api.read (execute api launch order m) (launch.output.address (linear q))=scalar api launch m q := by
  rw [execute_static]
  apply run_readback
  · exact ⟨event api launch m q,List.mem_map.mpr ⟨q,present,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨p,hp,rfl⟩:=List.mem_map.mp he
    by_cases eq : p=q
    · subst p
      exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (coordinate_apart launch.output eq)

theorem execute_frame (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (m : Store) (i : Nat) (outside : ¬launch.output.contains i) :
    api.byte (execute api launch order m) i=api.byte m i := by
  rw [execute_static]
  apply run_byte_frame
  intro e he
  obtain ⟨q,hq,rfl⟩:=List.mem_map.mp he
  have hq:=linear_bound q
  simp only [event,Dense.contains,Dense.address] at *
  omega

/-- The native prefix is actual writes; a failed callback's after-state is
retained separately. Ordinary callback frame is local to that one callback. -/
structure Stop (Store : Type) where
  completed : Nat
  afterCallback : Store → Store
  failure : Option Nat

def stopped (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (stop : Stop Store) (m : Store) : Store :=
  stop.afterCallback (execute api launch (order.take stop.completed) m)

theorem stopped_frame (api : API α Store Byte) (launch : Launch (α:=α) M N K C)
    (order : List (Coord M N K)) (stop : Stop Store) (m : Store) (i : Nat)
    (outside : ¬launch.output.contains i)
    (localFrame : ∀ s,api.byte (stop.afterCallback s) i=api.byte s i) :
    api.byte (stopped api launch order stop m) i=api.byte m i := by
  unfold stopped
  rw [localFrame,execute_frame api launch _ m i outside]

#axiomsClean execute_static
#axiomsClean execute_readback
#axiomsClean execute_frame
#axiomsClean stopped_frame
end
end VeriTile.Bench.Repairs.LogSoftmax.Execution

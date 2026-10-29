import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Closed

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Wrapper
open VeriTile.Triton FP.Structural
open Checked Physical
noncomputable section

def roleOf (r : RegionName) : Option Role :=
  if r="x" then some .x else if r="g" then some .g else if r="h0" then some .h0
  else if r="o" then some .o else if r="ht" then some .ht else if r="do" then some .incoming
  else if r="dht" then some .dht else if r="dx" then some .dx else if r="dg" then some .dg else none

theorem role_name (r : Role) : roleOf (name r)=some r := by cases r <;> simp [roleOf,name]

/-- A mathematical read view at actual Tensor addresses. This introduces no
runtime copy. Values outside the source roles are irrelevant to every load. -/
def memory {M α β : Type} (api : API M α β) (zero : α) (runtime : Runtime c U S V) (m : M) :
    RegionName → Nat → α := fun r a => match roleOf r with
  | some role => api.read m (key runtime ⟨role,a⟩)
  | none => zero

def terms {M β : Type} (api : API M (BitVec 32) β) (runtime : Runtime c U S V) (m : M) :
    RegionName → Nat → Native.Term :=
  fun r a => Native.Term.leaf r a (memory api 0 runtime m r a)

structure Forward (M : Type) where
  output : Tensor
  finalState : Option Tensor
  saved : Tensor × Tensor × Option Tensor
  memory : M

structure Backward (M : Type) where
  dx : Tensor
  dg : Tensor
  initialGradient : Option Tensor
  flagGradient : Option Tensor
  memory : M

def returnedForward (runtime : Runtime c U S V) (m : M) : Forward M :=
  ⟨runtime.tensor .o,if S then some (runtime.tensor .ht) else none,
   (runtime.tensor .g,runtime.tensor .o,if U then some (runtime.tensor .h0) else none),m⟩

def returnedBackward (runtime : Runtime c U S V) (m : M) : Backward M :=
  ⟨runtime.tensor .dx,runtime.tensor .dg,none,none,m⟩

def mathematicalForward {M β : Type} (api : API M ℝ β) (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (BD : Nat) (m : M) : Forward M :=
  let source := Closed.first realOps c BD U S (memory api 0 runtime m)
  returnedForward runtime (commit api c U S V runtime Bool.true
    (fun r a => Execution.read source (name r) a) m)

def mathematicalBackward {M β : Type} (api : API M ℝ β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (BDF BDB : Nat) (initial incoming : M) : Backward M :=
  let source := Closed.second realOps c BDF BDB U S hasDo V (memory api 0 runtime initial)
    (fun a => api.read incoming (key runtime ⟨.incoming,a⟩)) (fun a => api.read incoming (key runtime ⟨.dht,a⟩))
  returnedBackward runtime (commit api c U S V runtime Bool.false
    (fun r a => Execution.read source (name r) a) incoming)

def nativeForward {M β : Type} (api : API M (BitVec 32) β) (p : Native.Instructions) (schedule : Native.Schedule)
    (c : Shape) (U S V : Bool) (runtime : Runtime c U S V) (BD : Nat) (m : M) : Forward M :=
  let source := Closed.first Native.sourceOps c BD U S (terms api runtime m)
  returnedForward runtime (commit api c U S V runtime Bool.true
    (fun r a => Native.observe p schedule source (name r) a) m)

def nativeBackward {M β : Type} (api : API M (BitVec 32) β) (p : Native.Instructions) (schedule : Native.Schedule)
    (c : Shape) (U S V hasDo : Bool) (runtime : Runtime c U S V) (BDF BDB : Nat) (initial incoming : M) : Backward M :=
  let source := Closed.second Native.sourceOps c BDF BDB U S hasDo V (terms api runtime initial)
    (fun a => .leaf "do" a (api.read incoming (key runtime ⟨.incoming,a⟩)))
    (fun a => .leaf "dht" a (api.read incoming (key runtime ⟨.dht,a⟩)))
  returnedBackward runtime (commit api c U S V runtime Bool.false
    (fun r a => Native.observe p schedule source (name r) a) incoming)

/-- Ordinary saved-Tensor storage preservation between the two autograd calls,
including the zero allocation when the output cotangent is absent. This law
mentions only local addresses, never a computed output or derivative formula. -/
structure SavedStorage {M α β : Type} (api : API M α β) (runtime : Runtime c U S V)
    (before after : M) : Prop where
  read : ∀ r,r=.g ∨ r=.h0 ∨ r=.o ∨ r=.ht → present U S V r=Bool.true → ∀ a,a<extent c r →
    api.read after (key runtime ⟨r,a⟩)=api.read before (key runtime ⟨r,a⟩)

theorem forward_row_member (c : Shape) (S : Bool) (bh t d : Nat)
    (hb : bh<c.BH) (ht : t<c.T) (hd : d<c.D) :
    Physical.Cell.mk .o (rowAddr bh c.T c.D t d)∈cells c S Bool.true := by
  apply (cells_mem c S Bool.true _).mpr
  exact Or.inl ⟨rfl,row_bound c.BH c.T c.D bh t d hb ht hd⟩

theorem backward_row_member (c : Shape) (S : Bool) (r : Role) (role : r=.dx ∨ r=.dg)
    (bh t d : Nat) (hb : bh<c.BH) (ht : t<c.T) (hd : d<c.D) :
    Physical.Cell.mk r (rowAddr bh c.T c.D t d)∈cells c S Bool.false := by
  apply (cells_mem c S Bool.false _).mpr
  exact ⟨role,row_bound c.BH c.T c.D bh t d hb ht hd⟩

theorem forward_slot_member (c : Shape) (S : Bool) (hs : S=Bool.true) (bh d : Nat)
    (hb : bh<c.BH) (hd : d<c.D) : Physical.Cell.mk .ht (stateAddr bh c.D d)∈cells c S Bool.true := by
  apply (cells_mem c S Bool.true _).mpr
  exact Or.inr ⟨hs,rfl,state_bound c.BH c.D bh d hb hd⟩

#axiomsClean forward_row_member
#axiomsClean backward_row_member
#axiomsClean forward_slot_member
#specNonCircular SavedStorage avoiding [Execution.gridRun,Closed.first,Closed.second,state,dx,dg]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Wrapper

import bench.tritonbench_g.attention_fwd_triton1.Proofs.Correct
import Mathlib.Data.Nat.Bitwise

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked
open Whole001

/-- The exact dimension test performed before allocation, with no fixed D cap. -/
def dimensionCheck (D : Nat) : Bool := decide (16≤D ∧ D &&& (D-1)=0)

theorem dimensionCheck_iff (D : Nat) :
    dimensionCheck D=Bool.true ↔ 16≤D ∧ ∃ n,D=2^n := by
  simp only [dimensionCheck,decide_eq_true_eq]
  constructor
  · rintro ⟨hD,hbit⟩
    exact ⟨hD,(Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 hbit⟩
  · rintro ⟨hD,hpow⟩
    exact ⟨hD,(Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).2 hpow⟩

def metadataChecks (c : Shape) (args : Host.Arguments) : Bool :=
  [args.q,args.k,args.v].all fun t => decide
    (t.shape=Host.mainShape c ∧ t.device=args.q.device ∧ t.cuda=Bool.true ∧
      t.fp32=Bool.true ∧ t.contiguous=Bool.true ∧ t.conjugate=Bool.false ∧ t.negative=Bool.false)

theorem metadataChecks_iff (c : Shape) (args : Host.Arguments) :
    metadataChecks c args=Bool.true ↔ Host.InputChecks c args := by
  simp [metadataChecks,Host.InputChecks,List.all_eq_true]

def launchChecks (c : Shape) : Bool :=
  (Machine.metadata c).all (fun x => decide (x≤Machine.i32max)) &&
    decide (4*c.mainExtent≤Machine.i64max ∧ 4*c.historyExtent≤Machine.i64max)

theorem launchChecks_iff (c : Shape) : launchChecks c=Bool.true ↔ Machine.Launch c := by
  simp only [launchChecks,Bool.and_eq_true,List.all_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨m,a,h⟩;exact ⟨m,a,h⟩
  · intro h;exact ⟨h.metadata,h.mainBytes,h.historyBytes⟩

/-- Ordinary data_ptr observations. The word model binds origins by quotient;
alignment is established by the executed pointer checks rather than assumed. -/
structure Pointers {c : Shape} (abi : ABI c) where
  q : Nat
  k : Nat
  v : Nat
  h : Nat
  o : Nat
  qWord : abi.q.base=q/4
  kWord : abi.k.base=k/4
  vWord : abi.v.base=v/4
  hWord : abi.h.base=h/4
  oWord : abi.o.base=o/4

def pointerCheck (base extent : Nat) : Bool := decide (base%4=0 ∧ base+4*extent<2^64)

def pointerChecks (c : Shape) {abi : ABI c} (p : Pointers abi) : Bool :=
  pointerCheck p.q c.mainExtent && pointerCheck p.k c.mainExtent &&
  pointerCheck p.v c.mainExtent && pointerCheck p.h c.historyExtent && pointerCheck p.o c.mainExtent

def setupChecks (c : Shape) {abi : ABI c} (p : Pointers abi) : Bool :=
  if c.T=0 ∨ c.heads=0 then Bool.true else launchChecks c && pointerChecks c p

theorem pointer_bound (base extent word : Nat) (binding : word=base/4)
    (guard : pointerCheck base extent=Bool.true) : 4*(word+extent)<2^64 := by
  simp only [pointerCheck,decide_eq_true_eq] at guard
  have reconstruct := Nat.mod_add_div base 4
  rw [binding,Nat.mul_add]
  omega

theorem admitted_machine (c : Shape) (abi : ABI c) (p : Pointers abi)
    (guard : setupChecks c p=Bool.true) : Machine.Domain c abi := by
  refine ⟨?_,?_,?_⟩
  all_goals intro ht hh
  all_goals have h := guard
  all_goals simp only [setupChecks,ht,hh,or_self,↓reduceIte,Bool.and_eq_true] at h
  · exact (launchChecks_iff c).1 h.1
  · intro a ha
    simp only [pointerChecks,Bool.and_eq_true] at h
    rcases h.2 with ⟨⟨⟨⟨hq,hk⟩,hv⟩,hhist⟩,ho⟩
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ha
    rcases ha with rfl|rfl|rfl|rfl
    · exact pointer_bound _ _ _ p.qWord hq
    · exact pointer_bound _ _ _ p.kWord hk
    · exact pointer_bound _ _ _ p.vWord hv
    · exact pointer_bound _ _ _ p.oWord ho
  · simp only [pointerChecks,Bool.and_eq_true] at h
    exact pointer_bound _ _ _ p.hWord h.2.1.2

/-- Dynamic Tensor/strided-layout and bool-flag checks precede descriptors. -/
inductive Arguments where
  | unsupported
  | tensors (shape : Shape) (args : Host.Arguments)

def inputChecks (c : Shape) (args : Host.Arguments) : Bool :=
  metadataChecks c args && args.store.isSome && args.ifcond.isSome

def checkArguments : Arguments → Bool
  | .unsupported => Bool.false
  | .tensors c args => inputChecks c args

#axiomsClean dimensionCheck_iff
#axiomsClean metadataChecks_iff
#axiomsClean admitted_machine
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked

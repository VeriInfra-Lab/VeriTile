import bench.repairs.tritonbench_g.rotary_emb.Proofs.Physical
import bench.repairs.tritonbench_g.rotary_emb.Spec

namespace VeriTile.Bench.Repairs.RotaryEmb.Host

/-- Runtime scalar selected by the literal Python expression
int(width*partial_rotary_factor). We do not replace it by exact-real scaling.
The revised source checks its range and parity after this conversion. -/
structure Metadata where
  shape : Shape
  identities : Region → Nat
  device : Nat

/-- Admission to the documented ordinary allocation domain. Structural
metadata is checked in Python. Ordinary CUDA allocation provenance/device
lifetime are host-environment facts, independent of all output values. -/
structure Admitted (call : Metadata) : Prop where
  shape : call.shape.Valid
  dataBounds : ∀ key,call.shape.tokens*call.shape.heads key*call.shape.width≤(2^63-1)/4
  cosBounds : call.shape.tokens*call.shape.cosWidth≤(2^63-1)/4
  sinBounds : call.shape.tokens*call.shape.sinWidth≤(2^63-1)/4

structure Launch where
  heads : Nat
  sequences : Nat
  blockHeads : Nat
  blockSequences : Nat
  blockPairs : Nat
  warps : Nat
  stages : Nat
  deriving DecidableEq,Repr

def launch (s : Shape) : Launch :=
  ⟨(max s.headsQ s.headsK+3)/4,(s.tokens+15)/16,4,16,s.pairBlock,
    if 128≤s.rotary then 8 else 4,1⟩

def empty (s : Shape) : Bool := decide (s.tokens=0 ∨ s.rotary=0)

/-- A normal completed invocation returns None and the original objects;
there is no clone, copyback, replacement allocation or output-tensor return. -/
structure Returned (α : Type) where
  value : Option Unit
  identities : Region → Nat
  deviceRestored : Nat
  memory : Memory α

noncomputable def mathematical (call : Metadata) (callerDevice : Nat) (m : Memory ℝ) : Returned ℝ :=
  ⟨none,call.identities,callerDevice,
    if empty call.shape then m else source call.shape realOps m⟩

def selected (call : Metadata) (callerDevice : Nat) (p : Native.Instructions)
    (m : Physical.Bytes) : Returned Physical.Byte :=
  ⟨none,call.identities,callerDevice,
    if empty call.shape then m else Physical.native call.shape p m⟩

theorem empty_schedule (s : Shape) (h : empty s=true) : schedule s=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  have active := schedule_active s w hw
  have noWork : s.tokens=0 ∨ s.rotary=0 := by simpa [empty] using h
  rcases noWork with hn|hr
  · have := active.1
    omega
  · have := active.2.2
    simp [Shape.pairs,hr] at this

theorem mathematical_memory (call : Metadata) (valid : call.shape.Valid) (callerDevice : Nat)
    (m : Memory ℝ) : (mathematical call callerDevice m).memory=source call.shape realOps m := by
  unfold mathematical
  by_cases h : empty call.shape=true
  · simp only [h,ite_true]
    rw [source_eq_run call.shape valid,run,empty_schedule call.shape h]
    rfl
  · simp [h]

theorem selected_memory (call : Metadata) (callerDevice : Nat) (p : Native.Instructions)
    (m : Physical.Bytes) : (selected call callerDevice p m).memory=Physical.native call.shape p m := by
  unfold selected
  by_cases h : empty call.shape=true
  · simp only [h,ite_true]
    rw [Physical.native,empty_schedule call.shape h]
    rfl
  · simp [h]

theorem returned_objects (call : Metadata) (callerDevice : Nat) (p : Native.Instructions)
    (m : Physical.Bytes) :
    (selected call callerDevice p m).value=none ∧
    (selected call callerDevice p m).identities=call.identities ∧
    (selected call callerDevice p m).deviceRestored=callerDevice := ⟨rfl,rfl,rfl⟩

theorem no_launch_no_writes (call : Metadata) (callerDevice : Nat) (p : Native.Instructions)
    (m : Physical.Bytes) (h : empty call.shape=true) : (selected call callerDevice p m).memory=m := by
  simp [selected,h]

/-- Canonical contiguous offsets are within their actual allocation for every
active data load/store. Positivity/range facts follow from the checked shape. -/
theorem data_offset_bound (s : Shape) (valid : s.Valid) (w : Work) (active : s.Active w) (odd : Bool) :
    s.cell w odd<s.tokens*s.heads w.key*s.width := by
  have hn := active.1
  have hh := active.2.1
  have hd := lt_of_lt_of_le (pair_dim_bound s valid w active odd) valid.rotary_le
  have rows : w.token*s.heads w.key+w.head<s.tokens*s.heads w.key := by
    calc
      _ < w.token*s.heads w.key+s.heads w.key := Nat.add_lt_add_left hh _
      _ = (w.token+1)*s.heads w.key := by ring
      _ ≤ s.tokens*s.heads w.key := Nat.mul_le_mul_right _ (by omega)
  unfold Shape.cell Shape.address
  calc
    _ < (w.token*s.heads w.key+w.head)*s.width+s.width := Nat.add_lt_add_left hd _
    _ = (w.token*s.heads w.key+w.head+1)*s.width := by ring
    _ ≤ s.tokens*s.heads w.key*s.width := Nat.mul_le_mul_right _ (by omega)

theorem coefficient_offset_bound (s : Shape) (valid : s.Valid) (w : Work)
    (active : s.Active w) (sine odd : Bool) :
    s.coefficient sine w.token (2*w.pair+if odd then 1 else 0)<
      s.tokens*(if sine then s.sinWidth else s.cosWidth) := by
  have hn := active.1
  have hd := pair_dim_bound s valid w active odd
  have width : s.rotary≤(if sine then s.sinWidth else s.cosWidth) := by
    cases sine <;> simp [valid.cosWidth,valid.sinWidth]
  unfold Shape.coefficient
  calc
    _ < w.token*(if sine then s.sinWidth else s.cosWidth)+(if sine then s.sinWidth else s.cosWidth) :=
      Nat.add_lt_add_left (lt_of_lt_of_le hd width) _
    _ = (w.token+1)*(if sine then s.sinWidth else s.cosWidth) := by ring
    _ ≤ _ := Nat.mul_le_mul_right _ (by omega)

theorem data_byte_int64 (call : Metadata) (admit : Admitted call) (w : Work)
    (active : call.shape.Active w) (odd : Bool) (i : Fin 4) :
    4*call.shape.cell w odd+i.val<2^63 := by
  have hb := data_offset_bound call.shape admit.shape w active odd
  have hn := admit.dataBounds w.key
  omega

#axiomsClean mathematical_memory
#axiomsClean selected_memory
#axiomsClean returned_objects
#axiomsClean no_launch_no_writes
#axiomsClean data_byte_int64
#axiomsClean coefficient_offset_bound
end VeriTile.Bench.Repairs.RotaryEmb.Host

import bench.tritonbench_g.flash_decode2_phi.Proofs.Physical

namespace VeriTile.Bench.Repairs.FlashDecode2.Numerical
open VeriTile.Triton FP.Structural Native
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

/-- Exactly seven local native scalar families. exp32 is the scalar composition
of rounded log2(e) multiplication and approximate exp2 found in the emitted
PTX. No tensor/reduction/output callback exists. Zero and -infinity bits and
load wires are fixed. Operand order and FMA selection remain structural. -/
structure ScalarOps where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  exp32 : Word → Word
  fmaRN : Word → Word → Word → Word

def lower (p : ScalarOps) : Instructions where
  literal := fun _ _ _ => 0
  binary := fun _ _ op x y => match op with
    | .add => p.add x y | .sub => p.sub x y | .mul => p.mul x y
    | .div => p.div x y | .max => p.max x y | _ => 0
  unary := fun _ op x => match op with | .exp => p.exp32 x | _ => 0
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fmaRN := p.fmaRN

structure ScalarLaws (actual reference : ScalarOps) : Prop where
  add : ∀ x y,actual.add x y=reference.add x y
  sub : ∀ x y,actual.sub x y=reference.sub x y
  mul : ∀ x y,actual.mul x y=reference.mul x y
  div : ∀ x y,actual.div x y=reference.div x y
  max : ∀ x y,actual.max x y=reference.max x y
  exp32 : ∀ x,actual.exp32 x=reference.exp32 x
  fmaRN : ∀ x y z,actual.fmaRN x y z=reference.fmaRN x y z

theorem ScalarLaws.operations_eq (a b : ScalarOps) (h : ScalarLaws a b) : a=b := by
  cases a
  cases b
  simp only [ScalarOps.mk.injEq]
  exact ⟨funext (fun x => funext (h.add x)),funext (fun x => funext (h.sub x)),
    funext (fun x => funext (h.mul x)),funext (fun x => funext (h.div x)),
    funext (fun x => funext (h.max x)),funext h.exp32,
    funext (fun x => funext (fun y => funext (h.fmaRN x y)))⟩

def supported : Term → Prop
  | .leaf _ _ _ => True
  | .literal p d x => (p=none ∨ p=some .fp32) ∧ d=.real ∧ x=0
  | .negInf => True
  | .binary p d op a b => p=some .fp32 ∧ d=.real ∧
      (op=.add ∨ op=.sub ∨ op=.mul ∨ op=.div ∨ op=.max) ∧ supported a ∧ supported b
  | .unary p op a => p=some .fp32 ∧ op=.exp ∧ supported a
  | .load a => supported a
  | _ => False

theorem values_supported (b : Bindings) (memory : RegionName → Nat → Word)
    (masked : Bool) (B H S D BD row n : Nat) :
    let s := Grid.setup B H row (entry b memory)
    supported (values algebra s masked H S D BD n).m ∧
    supported (values algebra s masked H S D BD n).z ∧
    ∀ k,supported ((values algebra s masked H S D BD n).a k) := by
  let s := Grid.setup B H row (entry b memory)
  have hx : ∀ j,supported (logvalue algebra s H S j) := by
    intro j
    simp [logvalue,algebra,s,Grid.setup,entry,Cell.read_same,supported]
  have hv : ∀ j (k : TileIndex [BD]),supported (VeriTile.Bench.Repairs.FlashDecode2.value algebra s masked H S D j k) := by
    intro j k
    by_cases h : active masked D k <;>
      simp [VeriTile.Bench.Repairs.FlashDecode2.value,algebra,s,Grid.setup,entry,Cell.read_same,supported,h,zero]
  change supported (values algebra s masked H S D BD n).m ∧
    supported (values algebra s masked H S D BD n).z ∧
    ∀ k,supported ((values algebra s masked H S D BD n).a k)
  induction n with
  | zero => simp [values,initial,zero,algebra,supported]
  | succ n ih =>
    simp only [algebra] at hx hv ih
    simp [values,next,maximum,exponential,sub,mul,add,algebra,supported,hx,hv,ih.1,ih.2.1,ih.2.2]

theorem value_supported (b : Bindings) (memory : RegionName → Nat → Word)
    (q : Host.Request) (row : Nat) (k : TileIndex [q.g.laneBlock]) :
    supported (Grid.value algebra (entry b memory) q.g.masked q.g.batches q.g.heads
      q.g.capacity q.g.dimension q.g.sequenceBlock q.g.laneBlock row k) := by
  have h := values_supported b memory q.g.masked q.g.batches q.g.heads q.g.capacity q.g.dimension
    q.g.laneBlock row (count (Grid.setup q.g.batches q.g.heads row (entry b memory)) q.g.sequenceBlock)
  exact ⟨rfl,rfl,Or.inr (Or.inr (Or.inr (Or.inl rfl))),h.2.2 k,h.2.1⟩

/-- All whole-call native output bytes are equivalent once these seven scalar
families agree. Only local scalar numerical laws are deferred; the loop,
addresses, masks, stores, source grid and host return have already been proved. -/
theorem whole_equiv (actual reference : ScalarOps) (laws : ScalarLaws actual reference)
    (S : Schedule) (q : Host.Request) (memory : Physical.Bytes)
    (call : Host.Preparation Term q) (hbefore : call.before=entry (bindings call) (Physical.load memory))
    (schedule : List Nat) (all : schedule.Perm (List.range q.g.rows)) :
    ∃ result,Host.invoke algebra q call schedule=some result ∧ result.returned=none ∧
      result.suppliedOutput=call.out ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.g.rows=0 then none else some call.owner) ∧
      (∀ row (k : TileIndex [q.g.laneBlock]),row<q.g.rows → lane k<q.g.dimension → ∀ i : Fin 4,
        Physical.executeBytes (lower actual) S call.out.allocation q schedule call.before memory call.out.allocation
          (4*(row*q.g.dimension+lane k)+i.val)=
          Physical.encode (Native.value (lower reference) S (bindings call) (Physical.load memory) q row k) i) ∧
      (∀ r b,r≠call.out.allocation ∨ 4*q.g.size≤b →
        Physical.executeBytes (lower actual) S call.out.allocation q schedule call.before memory r b=memory r b) ∧
      (∀ b,Physical.projectO (lower actual) S result.state b=
        Physical.executeBytes (lower actual) S call.out.allocation q schedule call.before memory call.out.allocation b) := by
  cases laws.operations_eq actual reference
  exact Physical.whole_bytes (lower actual) S q memory call hbefore schedule all

#axiomsClean ScalarLaws.operations_eq
#axiomsClean values_supported
#axiomsClean value_supported
#axiomsClean whole_equiv
#specNonCircular ScalarLaws avoiding [Host.invoke,Grid.source,FP.Structural.exec,Native.value]
end
end VeriTile.Bench.Repairs.FlashDecode2.Numerical

import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Grid
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
namespace Numeric
open VeriTile.Triton.FP
abbrev Word := Value .fp32

/-- Exact source-word observation, including NaN payload, sign and signaling
bits. No canonicalization is inserted into the native source graph. -/
def observeWord (x : Word) : Word := x

theorem observeWord_idempotent (x : Word) : observeWord (observeWord x)=observeWord x := rfl

/-- Two fixed unary models selected before tensor values exist. exp2FTZ models
ex2.approx.ftz.f32. hostScale models Python binary64 pow(DK,-0.5) followed by
FP32 launch conversion; it is not identified with exact real sqrt. -/
structure UnaryModels where
  exp2FTZ : Word → Word
  hostScale : BitVec 32 → Word

def unsignedCast (x : BitVec 32) : Word := round .fp32 .ieeeValue (Rat.ofInt x.toNat)

def arithmetic (u : UnaryModels) : Arithmetic Word where
  zero := Value.ofBits .fp32 0
  one := Value.ofBits .fp32 1065353216
  negFive := Value.ofBits .fp32 3231711232
  fromNat := fun n => unsignedCast (BitVec.ofNat 32 n)
  add := fun x y => observeWord (add .ieeeValue (observeWord x) (observeWord y))
  sub := fun x y => observeWord (sub .ieeeValue (observeWord x) (observeWord y))
  mul := fun x y => observeWord (mul .ieeeValue (observeWord x) (observeWord y))
  fma := fun x y z => observeWord (fma .ieeeValue (observeWord x) (observeWord y) (observeWord z))
  exp2 := fun x => observeWord (u.exp2FTZ (observeWord x))
  scale := fun n => observeWord (u.hostScale (BitVec.ofNat 32 n))

/-- Physical scalar operation ports. No tensor shape, sequence, input buffer,
carry, dot, reduction or expected output is passed to these ports. -/
structure ScalarPorts where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  castU32 : BitVec 32 → Word
  exp2FTZ : Word → Word
  hostScale : BitVec 32 → Word

structure ScalarLaws (u : UnaryModels) (native : ScalarPorts) : Prop where
  add : ∀ x y,observeWord (native.add x y)=(arithmetic u).add x y
  sub : ∀ x y,observeWord (native.sub x y)=(arithmetic u).sub x y
  mul : ∀ x y,observeWord (native.mul x y)=(arithmetic u).mul x y
  fma : ∀ x y z,observeWord (native.fma x y z)=(arithmetic u).fma x y z
  castU32 : ∀ x,observeWord (native.castU32 x)=unsignedCast x
  exp2FTZ : ∀ x,observeWord (native.exp2FTZ x)=(arithmetic u).exp2 x
  hostScale : ∀ x,observeWord (native.hostScale x)=observeWord (u.hostScale x)

/-- The scalar native event interpreter reads/writes its current typed memory.
Physical load/store and graph/scheduling bindings are ordinary runtime
contracts. Seven scalar laws above are the only numerical equalities. -/
def scalarStep (native : ScalarPorts) : {β : Type} → Event Word β → Memory Word → β × Memory Word
  | _,.read p,m => (m p,m)
  | _,.zero,m => (Value.ofBits .fp32 0,m)
  | _,.one,m => (Value.ofBits .fp32 1065353216,m)
  | _,.negFive,m => (Value.ofBits .fp32 3231711232,m)
  | _,.fromNat n,m => (observeWord (native.castU32 (BitVec.ofNat 32 n)),m)
  | _,.scale dk,m => (observeWord (native.hostScale (BitVec.ofNat 32 dk)),m)
  | _,.add x y,m => (observeWord (native.add x y),m)
  | _,.sub x y,m => (observeWord (native.sub x y),m)
  | _,.mul x y,m => (observeWord (native.mul x y),m)
  | _,.fma x y z,m => (observeWord (native.fma x y z),m)
  | _,.exp2 x,m => (observeWord (native.exp2FTZ x),m)
  | _,.store p x,m => ((),writeWord m p x)
  | _,.barrier,m => ((),m)

def eventABI (u : UnaryModels) (native : ScalarPorts) (laws : ScalarLaws u native) :
    LocalEventABI Word (Memory Word) (arithmetic u) where
  observe := id
  step := scalarStep native
  stepLaw := by
    intro β e m
    cases e <;> simp [scalarStep,Event.eval,arithmetic,laws.add,laws.sub,laws.mul,laws.fma,
      laws.castU32,laws.exp2FTZ,laws.hostScale]

theorem native_forward (u : UnaryModels) (native : ScalarPorts) (laws : ScalarLaws u native)
    (s : Shape) (p : Plans) (b : InputBuffers) (alloc : Allocation 3 caller)
    (inputs : ∀ id,id∈b.storageIds → id∈caller) (seed final : Bool) (m : Memory Word) :
    ((forwardCode s p b alloc seed final).native (eventABI u native laws) m).2=
      forwardRun (arithmetic u) s p b alloc seed final m := by
  have h := congrArg Prod.snd ((forwardCode s p b alloc seed final).native_refines (eventABI u native laws) m)
  rw [eval_forwardCode _ _ _ _ _ inputs] at h
  exact h

theorem native_backward (u : UnaryModels) (native : ScalarPorts) (laws : ScalarLaws u native)
    (s : Shape) (p : Plans) (b : InputBuffers) (alloc : Allocation 6 caller)
    (inputs : ∀ id,id∈b.storageIds → id∈caller) (seed final : Bool) (m : Memory Word) :
    ((backwardCode s p b alloc seed final).native (eventABI u native laws) m).2=
      backwardRun (arithmetic u) s p b alloc seed final m := by
  have h := congrArg Prod.snd ((backwardCode s p b alloc seed final).native_refines (eventABI u native laws) m)
  rw [eval_backwardCode _ _ _ _ _ inputs] at h
  exact h

#axiomsClean observeWord_idempotent
#axiomsClean eventABI
#axiomsClean native_forward
#axiomsClean native_backward
end Numeric
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention

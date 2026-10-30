import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Events
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

instance (b : Buffers) : DecidablePred b.writesStorage := by
  intro p
  unfold Buffers.writesStorage
  infer_instance

def activationCode (name : Option String) (x : α) : Code α α :=
  if name=some "tanh" then emit (.tanh x)
  else if name=some "gelu" then do
    let half ← emit (.literal .half)
    let halfx ← emit (.mul x half)
    let root ← emit (.literal .sqrtTwo)
    let scaled ← emit (.divFull x root)
    let erf ← emit (.erf scaled)
    let one ← emit (.literal .one)
    let shifted ← emit (.add erf one)
    emit (.mul halfx shifted)
  else if name=some "fast_gelu" then do
    let cubic ← emit (.literal .cubic)
    let t1 ← emit (.mul x cubic)
    let t2 ← emit (.mul x t1)
    let inner ← emit (.fma x t2 x)
    let scale ← emit (.literal .sqrtTwoOverPi)
    let scaled ← emit (.mul inner scale)
    let tanh ← emit (.tanh scaled)
    let one ← emit (.literal .one)
    let shifted ← emit (.add tanh one)
    let half ← emit (.literal .half)
    let halfx ← emit (.mul x half)
    emit (.mul halfx shifted)
  else if name=some "relu" then do
    let zero ← emit (.literal .zero)
    emit (.maximum x zero)
  else pure x

@[simp] theorem eval_activationCode (a : Arithmetic α) (name : Option String) (x : α) (m : Memory α) :
    (activationCode name x).eval a m=(activation a name x,m) := by
  unfold activationCode activation
  split_ifs <;> simp [bind,eval_bind,Event.eval,pure]

/-- One source output cell: all K loads and instruction steps precede the saved
store, then scalar activation, then the output store. The saved store uses the
actual compute dtype, including FP16 under autocast. -/
def laneCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String)
    (offset : Nat) : Code α Unit := do
  let z ← accumulationCode s b c mode (offset/max s.n 1) (offset%max s.n 1)
  let _ ← match b.saved with
    | none => pure ()
    | some saved => emit (.store mode (saved,offset) z)
  let y ← activationCode name z
  emit (.store mode (b.out,offset) y)

def laneStep (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode)
    (name : Option String) (offset : Nat) (memory : Memory α) : Memory α :=
  let z := sourceAccumulation a s b c mode memory (offset/max s.n 1) (offset%max s.n 1)
  let withSaved := match b.saved with
    | none => memory
    | some saved => writeWord memory (saved,offset) (a.encode mode z)
  writeWord withSaved (b.out,offset) (a.encode mode (activation a name z))

@[simp] theorem eval_laneCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (offset : Nat) (memory : Memory α) :
    (laneCode s b c mode name offset).eval a memory=
      ((),laneStep a s b c mode name offset memory) := by
  cases hb : b.saved <;> simp [laneCode,laneStep,bind,eval_bind,hb,Event.eval,pure]

/-- Each valid masked lane corresponds to two independent cells. -/
theorem laneStep_form (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (offset : Nat) (memory : Memory α) (sep : Separate s b) :
    laneStep a s b c mode name offset memory=
      Schedule.put (fun off p => b.writesStorage p ∧ p.2=off)
        (programValue a s b c mode name memory) offset memory := by
  classical
  funext p
  cases hb : b.saved with
  | none =>
    simp only [laneStep,hb,writeWord,Schedule.put,programValue,Buffers.writesStorage,Option.noConfusion,or_false]
    by_cases hp : p=(b.out,offset)
    · cases hp; simp
    · have hn : ¬(p.1=b.out ∧ p.2=offset) := by
        intro h; exact hp (Prod.ext h.1 h.2)
      simp [hp,hn]
  | some saved =>
    have hne : saved≠b.out := by intro he; exact sep.2 (he ▸ hb)
    simp only [laneStep,hb,writeWord,Schedule.put,programValue,Buffers.writesStorage,Option.some.injEq]
    by_cases ho : p=(b.out,offset)
    · cases ho; simp
    · by_cases hs : p=(saved,offset)
      · cases hs; simp [hne,ho]
      · have hn : ¬((p.1=b.out ∨ saved=p.1) ∧ p.2=offset) := by
          intro h
          rcases h.1 with h1|h1
          · exact ho (Prod.ext h1 h.2)
          · exact hs (Prod.ext h1.symm h.2)
        simp [ho,hs,hn]

#axiomsClean eval_activationCode
#axiomsClean eval_laneCode
#axiomsClean laneStep_form
end
end VeriTile.Bench.Repairs.TritonLinearActivation

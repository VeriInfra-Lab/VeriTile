import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.OutputCoverage

/- The source launcher selects a.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Host
noncomputable section
set_option maxHeartbeats 5000000

structure Arguments where
  A : Tensor
  B : Tensor
  deriving DecidableEq

def Arguments.problem (args : Arguments) (sms : Nat) : Problem :=
  ⟨args.A.rows,args.B.columns,args.A.columns,sms,args.A.dtype⟩

inductive Call where
  | properties (device sms : Nat)
  | empty (rows columns : Nat) (profile : Storage) (device : Nat) (returned : Tensor)
  | zero (output : Tensor)
  | launch (args : Arguments) (sms : Nat) (output : Tensor)
  deriving DecidableEq

structure World (α σ : Type) where
  memory : Memory α
  native : σ
  calls : List Call

/-- Only ordinary effects: property lookup, allocation and successful calls.
There is no mathematical output, dot equality or memory-frame oracle here. -/
structure Runtime (σ : Type) where
  properties : Nat → σ → Option Nat
  empty : Nat → Nat → Storage → Nat → σ → Option (Tensor × σ)
  zero : Tensor → σ → Option σ
  launch : Arguments → Nat → Tensor → σ → Option σ

structure InputContract {σ : Type} (live : σ → Nat → Prop) (args : Arguments) (initial : σ) : Prop where
  compatible : args.A.columns=args.B.rows
  dtype : args.A.dtype=args.B.dtype
  device : args.A.device=args.B.device
  aRaw : args.A.isNeg=Bool.false ∧ args.A.isConj=Bool.false
  bRaw : args.B.isNeg=Bool.false ∧ args.B.isConj=Bool.false
  aBound : args.A.Bounded
  bBound : args.B.Bounded
  aLive : live initial args.A.storage
  bLive : live initial args.B.storage

structure Allocation {σ : Type} (live : σ → Nat → Prop) (rows columns : Nat)
    (profile : Storage) (device : Nat) (initial : σ) (output : Tensor) (allocated : σ) : Prop where
  rowsExact : output.rows=rows
  columnsExact : output.columns=columns
  dtype : output.dtype=profile
  deviceExact : output.device=device
  rowStride : output.rowStride=max columns 1
  columnStride : output.columnStride=1
  offset : output.offset=0
  raw : output.isNeg=Bool.false ∧ output.isConj=Bool.false
  bounded : output.Bounded
  fresh : ∀ storage,live initial storage → output.storage≠storage
  preserveLive : ∀ storage,live initial storage → live allocated storage

/-- Coordinate-preserving allocation and property ABI contracts, independent
of matmul values and launch execution. Unsupported dtypes are excluded by the
two-constructor external storage dispatch, including empty paths. -/
structure Library {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop) : Prop where
  sms : ∀ device initial sms,runtime.properties device initial=some sms → 0<sms ∧ sms<2^31
  allocation : ∀ rows columns profile device initial output allocated,
    runtime.empty rows columns profile device initial=some (output,allocated) →
    Allocation live rows columns profile device initial output allocated

def outputCell (p : Problem) (b : Buffers) (point : Address) : Prop :=
  ∃ row<p.M,∃ column<p.N,b.C.address row column=point

instance (p : Problem) (b : Buffers) : DecidablePred (outputCell p b) := Classical.decPred _

def zeroMemory {α : Type} (zero : α) (p : Problem) (b : Buffers) (memory : Memory α) : Memory α :=
  fun point => if outputCell p b point then zero else memory point

def afterAllocate {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (sms : Nat) (output : Tensor) (initial : World α σ) : Option (Tensor × World α σ) :=
  let p := args.problem sms
  let b : Buffers := ⟨args.A,args.B,output⟩
  if p.M=0 ∨ p.N=0 then some (output,initial)
  else if p.K=0 then do
    let native ← runtime.zero output initial.native
    pure (output,⟨zeroMemory ops.inputZero p b initial.memory,native,initial.calls++[.zero output]⟩)
  else do
    let native ← runtime.launch args sms output initial.native
    pure (output,⟨Source.run ops p b initial.memory,native,initial.calls++[.launch args sms output]⟩)

/-- Compatibility, actual A-device properties, profile dispatch/allocation,
empty branch, zero-K fill, complete positive launch, exact allocation return. -/
def run {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (initial : World α σ) : Option (Tensor × World α σ) := do
  if args.A.columns≠args.B.rows ∨ args.A.dtype≠args.B.dtype then none
  else do
    let sms ← runtime.properties args.A.device initial.native
    let (output,native) ← runtime.empty args.A.rows args.B.columns args.A.dtype args.A.device initial.native
    afterAllocate ops runtime args sms output
      ⟨initial.memory,native,initial.calls++[.properties args.A.device sms,
        .empty args.A.rows args.B.columns args.A.dtype args.A.device output]⟩

inductive Branch {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (sms : Nat) (output : Tensor) (initial final : World α σ) : Prop
  | empty (empty : (args.problem sms).M=0 ∨ (args.problem sms).N=0) (same : final=initial)
  | zero (nonempty : ¬((args.problem sms).M=0 ∨ (args.problem sms).N=0))
      (zeroK : (args.problem sms).K=0) (native : σ)
      (executed : runtime.zero output initial.native=some native)
      (same : final=⟨zeroMemory ops.inputZero (args.problem sms) ⟨args.A,args.B,output⟩ initial.memory,
        native,initial.calls++[.zero output]⟩)
  | launch (nonempty : ¬((args.problem sms).M=0 ∨ (args.problem sms).N=0))
      (positiveK : (args.problem sms).K≠0) (native : σ)
      (executed : runtime.launch args sms output initial.native=some native)
      (same : final=⟨Source.run ops (args.problem sms) ⟨args.A,args.B,output⟩ initial.memory,
        native,initial.calls++[.launch args sms output]⟩)

structure Completion {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (live : σ → Nat → Prop) (args : Arguments) (initial final : World α σ) (output : Tensor) where
  sms : Nat
  smsPositive : 0<sms
  smsBound : sms<2^31
  allocated : σ
  properties : runtime.properties args.A.device initial.native=some sms
  allocation : runtime.empty args.A.rows args.B.columns args.A.dtype args.A.device initial.native=some (output,allocated)
  layout : Layout (args.problem sms) ⟨args.A,args.B,output⟩
  fresh : ∀ storage,live initial.native storage → output.storage≠storage
  branch : Branch ops runtime args sms output
    ⟨initial.memory,allocated,initial.calls++[.properties args.A.device sms,
      .empty args.A.rows args.B.columns args.A.dtype args.A.device output]⟩ final

theorem after_complete {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (sms : Nat) (output returned : Tensor) (initial final : World α σ)
    (executed : afterAllocate ops runtime args sms output initial=some (returned,final)) :
    returned=output ∧ Branch ops runtime args sms output initial final := by
  unfold afterAllocate at executed
  dsimp only at executed
  split_ifs at executed with empty zeroK
  · have same := Option.some.inj executed
    exact ⟨(congrArg Prod.fst same).symm,.empty empty (congrArg Prod.snd same).symm⟩
  · cases hn : runtime.zero output initial.native with
    | none => simp [hn] at executed
    | some native =>
      simp only [hn] at executed
      have same := Option.some.inj executed
      exact ⟨(congrArg Prod.fst same).symm,.zero empty zeroK native hn (congrArg Prod.snd same).symm⟩
  · cases hn : runtime.launch args sms output initial.native with
    | none => simp [hn] at executed
    | some native =>
      simp only [hn] at executed
      have same := Option.some.inj executed
      exact ⟨(congrArg Prod.fst same).symm,.launch empty zeroK native hn (congrArg Prod.snd same).symm⟩

theorem complete {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (live : σ → Nat → Prop) (args : Arguments) (initial final : World α σ) (output : Tensor)
    (library : Library runtime live) (input : InputContract live args initial.native)
    (executed : run ops runtime args initial=some (output,final)) :
    Nonempty (Completion ops runtime live args initial final output) := by
  unfold run at executed
  have compatible : ¬(args.A.columns≠args.B.rows ∨ args.A.dtype≠args.B.dtype) := by
    simp only [input.compatible,input.dtype,ne_eq,not_true_eq_false,or_self,not_false_eq_true]
  rw [if_neg compatible] at executed
  cases hp : runtime.properties args.A.device initial.native with
  | none => simp [hp] at executed
  | some sms =>
    simp only [hp,Option.bind_some] at executed
    cases ha : runtime.empty args.A.rows args.B.columns args.A.dtype args.A.device initial.native with
    | none => simp [ha] at executed
    | some allocated =>
      rcases allocated with ⟨fresh,allocated⟩
      simp only [ha,Option.bind_some] at executed
      obtain ⟨returned,branch⟩ := after_complete ops runtime args sms fresh output _ final executed
      subst fresh
      have allocation := library.allocation _ _ _ _ _ _ _ ha
      have smsLaw := library.sms _ _ _ hp
      have layout : Layout (args.problem sms) ⟨args.A,args.B,output⟩ := by
        exact ⟨⟨rfl,rfl⟩,⟨input.compatible.symm,rfl⟩,⟨allocation.rowsExact,allocation.columnsExact⟩,
          rfl,input.dtype.symm,allocation.dtype,input.aBound,input.bBound,allocation.bounded,
          ⟨allocation.rowStride,allocation.columnStride⟩,
          allocation.fresh _ input.aLive,allocation.fresh _ input.bLive⟩
      exact ⟨⟨sms,smsLaw.1,smsLaw.2,allocated,hp,ha,layout,
        allocation.fresh,branch⟩⟩

#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Host

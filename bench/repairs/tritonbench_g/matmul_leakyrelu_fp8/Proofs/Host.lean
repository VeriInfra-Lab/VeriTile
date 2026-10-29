import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Autotune

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host
noncomputable section
set_option maxHeartbeats 5000000

structure Arguments where
  A : Tensor
  B : Tensor
  activation : String
  deriving DecidableEq

def Arguments.problem (args : Arguments) : Problem :=
  ⟨args.A.rows,args.B.columns,args.A.columns,TypedSource.activationEnabled args.activation,args.A.dtype⟩

def contiguous (t : Tensor) : Prop := t.rows=0 ∨ t.columns=0 ∨
  ((t.columns≤1 ∨ t.columnStride=1) ∧ (t.rows≤1 ∨ t.rowStride=t.columns))
instance (t : Tensor) : Decidable (contiguous t) := inferInstanceAs (Decidable (_ ∨ _ ∨ _))

inductive Call where
  | empty (rows columns : Nat) (profile : Storage) (device : Nat) (returned : Tensor)
  | zero (output : Tensor)
  | autotune (trials : Nat)
  | launch (args : Arguments) (output : Tensor)
  deriving DecidableEq

structure World (α σ : Type) where
  memory : Memory α
  native : σ
  calls : List Call

/-- Ordinary runtime effects and autotuner trial-count selection. The table
has exactly one configuration; all trial and final launches use its kernel.
No output value, numerical equality or successful source run is a field. -/
structure Runtime (σ : Type) where
  empty : Nat → Nat → Storage → Nat → σ → Option (Tensor × σ)
  zero : Tensor → σ → Option σ
  trials : Arguments → Tensor → σ → Option Nat
  launch : Arguments → Tensor → σ → Option σ

structure InputContract {σ : Type} (live : σ → Nat → Prop) (args : Arguments) (initial : σ) : Prop where
  compatible : args.A.columns=args.B.rows
  aContiguous : contiguous args.A
  aType : args.A.dtype=.fp32
  bType : args.B.dtype=.fp32
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

structure Library {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop) : Prop where
  allocation : ∀ rows columns profile device initial output allocated,
    runtime.empty rows columns profile device initial=some (output,allocated) →
    Allocation live rows columns profile device initial output allocated

def outputCell (p : Problem) (b : Buffers) (point : Address) : Prop :=
  ∃ row<p.M,∃ column<p.N,b.C.address row column=point
instance (p : Problem) (b : Buffers) : DecidablePred (outputCell p b) := Classical.decPred _
def zeroMemory {α : Type} (zero : α) (p : Problem) (b : Buffers) (memory : Memory α) : Memory α :=
  fun point => if outputCell p b point then zero else memory point

/-- Each actual trial/final launch reads the memory left by the previous one. -/
def launchSequence {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (output : Tensor) : Nat → World α σ → Option (World α σ)
  | 0,initial => some initial
  | count+1,initial => do
    let native ← runtime.launch args output initial.native
    launchSequence ops runtime args output count
      ⟨Source.run ops args.problem ⟨args.A,args.B,output⟩ initial.memory,native,initial.calls++[.launch args output]⟩

theorem sequence_memory {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (output : Tensor) (count : Nat) (initial final : World α σ)
    (executed : launchSequence ops runtime args output count initial=some final) :
    final.memory=Source.replay ops args.problem ⟨args.A,args.B,output⟩ count initial.memory ∧
    final.calls=initial.calls++List.replicate count (.launch args output) := by
  induction count generalizing initial with
  | zero =>
    have same : initial=final := Option.some.inj executed
    subst final
    simp [Source.replay]
  | succ count ih =>
    cases hn : runtime.launch args output initial.native with
    | none => simp [launchSequence,hn] at executed
    | some native =>
      simp only [launchSequence,hn,Option.bind_some] at executed
      obtain ⟨memory,calls⟩ := ih _ executed
      refine ⟨memory,?_⟩
      simpa [List.replicate_succ,List.append_assoc] using calls

def afterAllocate {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (output : Tensor) (initial : World α σ) : Option (Tensor × World α σ) :=
  let p := args.problem
  let b : Buffers := ⟨args.A,args.B,output⟩
  if p.M=0 ∨ p.N=0 then some (output,initial)
  else if p.K=0 then do
    let native ← runtime.zero output initial.native
    pure (output,⟨zeroMemory ops.inputZero p b initial.memory,native,initial.calls++[.zero output]⟩)
  else do
    let trials ← runtime.trials args output initial.native
    let final ← launchSequence ops runtime args output (trials+1)
      ⟨initial.memory,initial.native,initial.calls++[.autotune trials]⟩
    pure (output,final)

/-- Ordered original shape/contiguity assertions, explicit resolved flags,
FP16 allocation, all empty/K0/positive branches, autotune and exact return. -/
def run {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (initial : World α σ) : Option (Tensor × World α σ) := do
  if args.A.columns≠args.B.rows then none
  else if ¬contiguous args.A then none
  else if args.A.isNeg || args.A.isConj then none
  else if args.B.isNeg || args.B.isConj then none
  else do
    let (output,native) ← runtime.empty args.A.rows args.B.columns .fp16 args.A.device initial.native
    afterAllocate ops runtime args output
      ⟨initial.memory,native,initial.calls++[.empty args.A.rows args.B.columns .fp16 args.A.device output]⟩

inductive Branch {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (output : Tensor) (initial final : World α σ) : Prop
  | empty (empty : args.problem.M=0 ∨ args.problem.N=0) (same : final=initial)
  | zero (nonempty : ¬(args.problem.M=0 ∨ args.problem.N=0)) (zeroK : args.problem.K=0) (native : σ)
      (executed : runtime.zero output initial.native=some native)
      (same : final=⟨zeroMemory ops.inputZero args.problem ⟨args.A,args.B,output⟩ initial.memory,
        native,initial.calls++[.zero output]⟩)
  | launch (nonempty : ¬(args.problem.M=0 ∨ args.problem.N=0)) (positiveK : args.problem.K≠0) (trials : Nat)
      (selected : runtime.trials args output initial.native=some trials)
      (executed : launchSequence ops runtime args output (trials+1)
        ⟨initial.memory,initial.native,initial.calls++[.autotune trials]⟩=some final)

structure Completion {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (live : σ → Nat → Prop) (args : Arguments) (initial final : World α σ) (output : Tensor) where
  allocated : σ
  allocation : runtime.empty args.A.rows args.B.columns .fp16 args.A.device initial.native=some (output,allocated)
  layout : Layout args.problem ⟨args.A,args.B,output⟩
  fresh : ∀ storage,live initial.native storage → output.storage≠storage
  branch : Branch ops runtime args output
    ⟨initial.memory,allocated,initial.calls++[.empty args.A.rows args.B.columns .fp16 args.A.device output]⟩ final

theorem after_complete {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (args : Arguments) (output returned : Tensor) (initial final : World α σ)
    (executed : afterAllocate ops runtime args output initial=some (returned,final)) :
    returned=output ∧ Branch ops runtime args output initial final := by
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
  · cases ht : runtime.trials args output initial.native with
    | none => simp [ht] at executed
    | some trials =>
      simp only [ht,Option.bind_some] at executed
      change (launchSequence ops runtime args output (trials+1)
        ⟨initial.memory,initial.native,initial.calls++[.autotune trials]⟩).bind
        (fun result => some (output,result))=some (returned,final) at executed
      cases hl : launchSequence ops runtime args output (trials+1)
        ⟨initial.memory,initial.native,initial.calls++[.autotune trials]⟩ with
      | none => simp [hl] at executed
      | some result =>
        simp only [hl,Option.bind] at executed
        have same := Option.some.inj executed
        have finalEq : final=result := (congrArg Prod.snd same).symm
        exact ⟨(congrArg Prod.fst same).symm,.launch empty zeroK trials ht (by rw [finalEq]; exact hl)⟩

theorem complete {α β σ : Type} (ops : Source.Arithmetic α β) (runtime : Runtime σ)
    (live : σ → Nat → Prop) (args : Arguments) (initial final : World α σ) (output : Tensor)
    (library : Library runtime live) (input : InputContract live args initial.native)
    (executed : run ops runtime args initial=some (output,final)) :
    Nonempty (Completion ops runtime live args initial final output) := by
  unfold run at executed
  simp only [input.compatible,ne_eq,not_true_eq_false,input.aContiguous,
    input.aRaw.1,input.aRaw.2,input.bRaw.1,input.bRaw.2,Bool.or_self,if_false] at executed
  cases ha : runtime.empty args.A.rows args.B.columns .fp16 args.A.device initial.native with
  | none => simp [ha] at executed
  | some allocated =>
    rcases allocated with ⟨fresh,allocated⟩
    simp only [ha,Option.bind_some] at executed
    obtain ⟨returned,branch⟩ := after_complete ops runtime args fresh output _ final executed
    subst fresh
    have allocation := library.allocation _ _ _ _ _ _ _ ha
    have layout : Layout args.problem ⟨args.A,args.B,output⟩ := by
      exact ⟨⟨rfl,rfl⟩,⟨input.compatible.symm,rfl⟩,⟨allocation.rowsExact,allocation.columnsExact⟩,
        rfl,input.bType.trans input.aType.symm,allocation.dtype,input.aBound,input.bBound,allocation.bounded,
        ⟨allocation.rowStride,allocation.columnStride⟩,allocation.fresh _ input.aLive,allocation.fresh _ input.bLive⟩
    exact ⟨⟨allocated,ha,layout,allocation.fresh,branch⟩⟩

#axiomsClean sequence_memory
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host

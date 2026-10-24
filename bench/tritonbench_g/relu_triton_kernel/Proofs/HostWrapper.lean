import bench.tritonbench_g.relu_triton_kernel.Proofs.TypedGrid

/-! Stateful, fallible host control flow. Native metadata access, allocation,
literal tensor creation and launch are explicit primitive boundaries. The
logical launch composes the actual typed grid for twelve profiles; unsupported
logical profiles are not presented as native compiler errors. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.HostWrapper
open VeriTile.Triton FP.Structural GatedGrid TypedValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Tensor where
  region : RegionName
  dtype : String
  device : String
  shape : List Nat
  strides : List Int
  storageOffset : Nat
  deriving DecidableEq, Repr

inductive Error where
  | native (message : String)
  | unsupportedLogicalProfile (dtype : String)
  | logicalInterpreter
  deriving DecidableEq, Repr

inductive Result (σ α : Type) where
  | ok (value : α) (state : σ)
  | error (reason : Error) (state : σ)

abbrev Run (σ α : Type) := σ → Result σ α

def Run.pure {σ α : Type} (a : α) : Run σ α := fun s => .ok a s

def Run.bind {σ α β : Type} (r : Run σ α) (f : α → Run σ β) : Run σ β :=
  fun s => match r s with
    | .ok a t => f a t
    | .error e t => .error e t

instance instMonadRun {σ : Type} : Monad (Run σ) where
  pure := Run.pure
  bind := Run.bind

inductive Event where
  | device (x : Tensor)
  | emptyLike (x : Tensor) (dtype device : String)
  | numel (output : Tensor)
  | computedNumBlocks (n result : Nat)
  | launch (x output : Tensor) (n grid block : Nat)
  | returned (output : Tensor)
  | tensor (values : List Rat) (dtype device : String)
  | inserted (key : String) (output : Tensor)
  | assignedResultGold
  deriving DecidableEq, Repr

structure HostState (σ : Type) where
  runtime : σ
  trace : List Event

structure Primitives (σ : Type) where
  device : Tensor → Run σ String
  emptyLike : Tensor → String → String → Run σ Tensor
  numel : Tensor → Run σ Nat
  launch : Tensor → Tensor → Nat → Nat → Nat → Run σ Unit
  tensor : List Rat → String → String → Run σ Tensor

def call {σ α : Type} (event : Event) (action : Run σ α) : Run (HostState σ) α :=
  fun s => match action s.runtime with
    | .ok a t => .ok a ⟨t, s.trace ++ [event]⟩
    | .error e t => .error e ⟨t, s.trace ++ [event]⟩

def note {σ : Type} (event : Event) : Run (HostState σ) Unit :=
  fun s => .ok () { s with trace := s.trace ++ [event] }

/-- x.device is evaluated before empty_like. The computed num_blocks is
retained even though the actual source launch uses its separate grid lambda.
The original has no CUDA, shape, contiguous-layout or dtype assertion. -/
def relu {σ : Type} (P : Primitives σ) (x : Tensor) : Run (HostState σ) Tensor := do
  let device ← call (.device x) (P.device x)
  let output ← call (.emptyLike x "float32" device) (P.emptyLike x "float32" device)
  let n ← call (.numel output) (P.numel output)
  let numBlocks := (n + 1024 - 1) / 1024
  note (.computedNumBlocks n numBlocks)
  call (.launch x output n (gridCount n 1024) 1024)
    (P.launch x output n (gridCount n 1024) 1024)
  note (.returned output)
  pure output

def inputs : List (String × List Rat) := [
  ("test_case_1", [-3, -1, -(1/2), -2, -5]),
  ("test_case_2", [3, 1, 1/2, 2, 5]),
  ("test_case_3", [-3, -1, 0, 2, 5]),
  ("test_case_4", [0, 0, 0, 0, 0]) ]

def runCases {σ : Type} (P : Primitives σ) :
    List (String × List Rat) → Run (HostState σ) (List (String × Tensor))
  | [] => pure []
  | (key, values) :: rest => do
      let x ← call (.tensor values "float32" "cuda") (P.tensor values "float32" "cuda")
      let output ← relu P x
      note (.inserted key output)
      let remaining ← runCases P rest
      pure ((key, output) :: remaining)

def testRelu {σ : Type} (P : Primitives σ) := runCases P inputs

def initializeModule {σ : Type} (P : Primitives σ) : Run (HostState σ) (List (String × Tensor)) := do
  let result ← testRelu P
  note .assignedResultGold
  pure result

def successTrace (x output : Tensor) (device : String) (n : Nat) : List Event :=
  [.device x, .emptyLike x "float32" device, .numel output,
   .computedNumBlocks n ((n + 1024 - 1) / 1024),
   .launch x output n (gridCount n 1024) 1024, .returned output]

theorem device_error_first {σ : Type} (P : Primitives σ) (x : Tensor)
    (s t : σ) (trace : List Event) (e : Error)
    (h : P.device x s = .error e t) :
    relu P x ⟨s, trace⟩ = .error e ⟨t, trace ++ [.device x]⟩ := by
  simp [Bind.bind, Pure.pure, instMonadRun, relu, call, Run.bind, h]

theorem allocation_error_after_device {σ : Type} (P : Primitives σ) (x : Tensor)
    (s d a : σ) (trace : List Event) (device : String) (e : Error)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .error e a) :
    relu P x ⟨s, trace⟩ = .error e
      ⟨a, trace ++ [.device x, .emptyLike x "float32" device]⟩ := by
  simp [Bind.bind, Pure.pure, instMonadRun, relu, call, Run.bind, hd, ha, List.append_assoc]

theorem numel_error_after_allocation {σ : Type} (P : Primitives σ) (x output : Tensor)
    (s d a m : σ) (trace : List Event) (device : String) (e : Error)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .error e m) :
    relu P x ⟨s, trace⟩ = .error e
      ⟨m, trace ++ [.device x, .emptyLike x "float32" device, .numel output]⟩ := by
  simp [Bind.bind, Pure.pure, instMonadRun, relu, call, Run.bind, hd, ha, hm, List.append_assoc]

theorem launch_error_stops_return {σ : Type} (P : Primitives σ) (x output : Tensor)
    (s d a m l : σ) (trace : List Event) (device : String) (n : Nat) (e : Error)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m)
    (hl : P.launch x output n (gridCount n 1024) 1024 m = .error e l) :
    relu P x ⟨s, trace⟩ = .error e
      ⟨l, trace ++ (successTrace x output device n).take 5⟩ := by
  simp [Bind.bind, Pure.pure, instMonadRun, relu, call, note, Run.bind, hd, ha, hm, hl, successTrace, List.append_assoc]

theorem relu_success {σ : Type} (P : Primitives σ) (x output : Tensor)
    (s d a m l : σ) (trace : List Event) (device : String) (n : Nat)
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m)
    (hl : P.launch x output n (gridCount n 1024) 1024 m = .ok () l) :
    relu P x ⟨s, trace⟩ = .ok output ⟨l, trace ++ successTrace x output device n⟩ := by
  simp [Bind.bind, Pure.pure, instMonadRun, relu, call, note, Run.bind, Run.pure, hd, ha, hm, hl, successTrace, List.append_assoc]

def profileDType : Profile → String
  | .i8 => "int8" | .i16 => "int16" | .i32 => "int32" | .i64 => "int64"
  | .u8 => "uint8" | .u16 => "uint16" | .u32 => "uint32" | .u64 => "uint64"
  | .boolean => "bool" | .fp16 => "float16" | .bf16 => "bfloat16" | .fp32 => "float32"

def selectProfile : String → Option Profile
  | "int8" => some .i8 | "int16" => some .i16 | "int32" => some .i32 | "int64" => some .i64
  | "uint8" => some .u8 | "uint16" => some .u16 | "uint32" => some .u32 | "uint64" => some .u64
  | "bool" => some .boolean | "float16" => some .fp16 | "bfloat16" => some .bf16 | "float32" => some .fp32
  | _ => none

theorem select_profile (p : Profile) : selectProfile (profileDType p) = some p := by cases p <;> rfl

def logicalLaunch {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (x output : Tensor) (n grid block : Nat) : Run (State α) Unit :=
  fun s => match selectProfile x.dtype with
    | none => .error (.unsupportedLogicalProfile x.dtype) s
    | some p => match runPrefix (backend M cmp16 cmp32)
        (source p x.region output.region n block) s grid with
      | none => .error .logicalInterpreter s
      | some t => .ok () t

def withLogicalLaunch {α : Type} [Inhabited α] (P : Primitives (State α))
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool) : Primitives (State α) :=
  { P with launch := logicalLaunch M cmp16 cmp32 }

/-- Kernel success and memory are proved from the typed source. The hypotheses
describe only the preceding native-boundary outcomes, never a kernel result. -/
theorem relu_logical_correct {α : Type} [Inhabited α] (P : Primitives (State α))
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool) (p : Profile) (x output : Tensor)
    (s d a m : State α) (trace : List Event) (device : String) (n : Nat)
    (hp : x.dtype = (profileDType p))
    (hd : P.device x s = .ok device d)
    (ha : P.emptyLike x "float32" device d = .ok output a)
    (hm : P.numel output a = .ok n m) :
    ∃ final, relu (withLogicalLaunch P M cmp16 cmp32) x ⟨s, trace⟩ =
        .ok output ⟨final, trace ++ successTrace x output device n⟩ ∧
      final.mem = if gridCount n 1024 = 0 then m.mem
        else reference M cmp16 cmp32 p.kind x.region output.region n 1024 m.mem := by
  obtain ⟨final, hf, hmem⟩ := prefix_correct M cmp16 cmp32 p
    x.region output.region n 1024 m (gridCount n 1024)
  refine ⟨final, ?_, hmem⟩
  apply relu_success
  · exact hd
  · exact ha
  · exact hm
  · simp only [withLogicalLaunch, logicalLaunch, hp, select_profile, hf]

def Result.Post {σ α : Type} (post : α → σ → Prop) : Result σ α → Prop
  | .ok a s => post a s
  | .error _ _ => True

theorem bind_post {σ α β : Type} (r : Run σ α) (f : α → Run σ β)
    (post : β → σ → Prop) (s : σ)
    (h : ∀ a t, r s = .ok a t → (f a t).Post post) :
    (Run.bind r f s).Post post := by
  simp only [Run.bind]
  split
  · exact h _ _ (by assumption)
  · trivial

theorem runCases_keys {σ : Type} (P : Primitives σ) (cases : List (String × List Rat))
    (s : HostState σ) :
    (runCases P cases s).Post (fun outputs _ => outputs.map Prod.fst = cases.map Prod.fst) := by
  induction cases generalizing s with
  | nil => rfl
  | cons entry rest ih =>
    rcases entry with ⟨key, values⟩
    apply bind_post
    intro x s1 hx
    apply bind_post
    intro output s2 ho
    apply bind_post
    intro _ s3 hn
    apply bind_post
    intro results s4 hr
    have hkeys := ih s3
    rw [hr] at hkeys
    simpa [Pure.pure, instMonadRun, Run.pure, Result.Post] using congrArg (List.cons key) hkeys

theorem harness_keys {σ : Type} (P : Primitives σ) (s : HostState σ) :
    (testRelu P s).Post (fun outputs _ => outputs.map Prod.fst =
      ["test_case_1", "test_case_2", "test_case_3", "test_case_4"]) :=
  runCases_keys P inputs s

theorem first_tensor_error_stops_harness {σ : Type} (P : Primitives σ)
    (s t : σ) (trace : List Event) (e : Error)
    (h : P.tensor [-3, -1, -(1/2), -2, -5] "float32" "cuda" s = .error e t) :
    initializeModule P ⟨s, trace⟩ = .error e
      ⟨t, trace ++ [.tensor [-3, -1, -(1/2), -2, -5] "float32" "cuda"]⟩ := by
  simp only [Bind.bind, Pure.pure, instMonadRun, initializeModule, testRelu,
    runCases, inputs, Run.bind, call, h]

theorem relu_error_stops_case {σ : Type} (P : Primitives σ)
    (key : String) (values : List Rat) (rest : List (String × List Rat))
    (s : HostState σ) (a : σ) (x : Tensor) (t : HostState σ) (e : Error)
    (ha : P.tensor values "float32" "cuda" s.runtime = .ok x a)
    (hr : relu P x ⟨a, s.trace ++ [.tensor values "float32" "cuda"]⟩ = .error e t) :
    runCases P ((key, values) :: rest) s = .error e t := by
  simp only [runCases, Bind.bind, Pure.pure, instMonadRun, Run.bind, call, ha, hr]

theorem module_error_preserves_state {σ : Type} (P : Primitives σ)
    (s t : HostState σ) (e : Error) (h : testRelu P s = .error e t) :
    initializeModule P s = .error e t := by
  simp only [initializeModule, Bind.bind, Pure.pure, instMonadRun, Run.bind, h]

theorem module_success_records_assignment {σ : Type} (P : Primitives σ)
    (s t : HostState σ) (outputs : List (String × Tensor))
    (h : testRelu P s = .ok outputs t) :
    initializeModule P s = .ok outputs { t with trace := t.trace ++ [.assignedResultGold] } := by
  simp only [initializeModule, Bind.bind, Pure.pure, instMonadRun, Run.bind, Run.pure, h, note]

#specNonCircular successTrace avoiding [relu, logicalLaunch, runPrefix, FP.Structural.exec]
#axiomsClean device_error_first
#axiomsClean allocation_error_after_device
#axiomsClean numel_error_after_allocation
#axiomsClean launch_error_stops_return
#axiomsClean relu_success
#axiomsClean select_profile
#axiomsClean relu_logical_correct
#axiomsClean bind_post
#axiomsClean runCases_keys
#axiomsClean harness_keys
#axiomsClean first_tensor_error_stops_harness
#axiomsClean relu_error_stops_case
#axiomsClean module_error_preserves_state
#axiomsClean module_success_records_assignment
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.HostWrapper

import bench.tritonbench_g.vector_addition.Proofs.HostGrid

/-! Source-corresponding logical host control flow for `vector_addition.py`.

Native primitives are explicit, fallible and stateful. Their implementation,
Tensor metadata interpretation, physical allocation and dtype-sensitive native
launch are not supplied by this file. The successful logical-launch theorem
constructs an actual `runGrid` execution; it does not assume kernel success or
output equality. The scalar kernel remains the legacy `.real` source.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.HostWrapper
open VeriTile.Triton FP.Structural
open ValueProofs
noncomputable section

/-- Unconstrained host metadata. In particular there is no dtype, shape,
stride, contiguity, storage-offset or same-device check in `add`. Natural
`numel` and logical regions still need a native metadata/address refinement. -/
structure Tensor where
  region : RegionName
  numel : Nat
  isCUDA : Bool
  dtype : String
  shape : List Nat
  strides : List Int
  storageOffset : Int
  deviceIndex : Option Nat

/-- Native state includes allocator/RNG/environment state, abstractly. -/
structure World (α σ : Type) where
  kernel : State α
  native : σ

inductive Failure where
  | native (message : String)
  | cudaAssertion
  | logicalInterpreter

/-- Errors preserve any state changes made before the error. -/
inductive Result (S A : Type) where
  | ok (value : A) (state : S)
  | error (reason : Failure) (state : S)

inductive Event where
  | emptyLike (input : RegionName)
  | cudaCheck (xCUDA yCUDA outputCUDA : Bool)
  | numel (output : RegionName) (n : Nat)
  | launch (n block grid : Nat)
  | returnTensor (output : RegionName)
  | manualSeed (seed : Nat)
  | rand (n : Nat) (device : String)
  | dictionary (keys : List String)
  | assign (name : String)

structure Run (S A : Type) where
  trace : List Event
  result : Result S A

def Run.bind {S A B : Type} (previous : Run S A)
    (next : A → S → Run S B) : Run S B :=
  match previous.result with
  | .error reason state => ⟨previous.trace, .error reason state⟩
  | .ok value state =>
    let following := next value state
    ⟨previous.trace ++ following.trace, following.result⟩

abbrev Primitive (S A : Type) := S → Result S A

def perform {S A : Type} (event : Event) (primitive : Primitive S A)
    (state : S) : Run S A := ⟨[event], primitive state⟩

/-- The original random calls specify only size and `device='cuda'`; dtype
and other defaults are governed by the opaque native environment. -/
structure Runtime (α σ : Type) where
  emptyLike : Tensor → Primitive (World α σ) Tensor
  manualSeed : Nat → Primitive (World α σ) Unit
  rand : Nat → String → Primitive (World α σ) Tensor
  launch : Tensor → Tensor → Tensor → Parameters → Primitive (World α σ) Unit

def cudaGuard (x y output : Tensor) : Bool :=
  x.isCUDA && y.isCUDA && output.isCUDA

def parameters (x y output : Tensor) : Parameters :=
  ⟨x.region, y.region, output.region, output.numel, 1024⟩

def successTrace (x y output : Tensor) : List Event :=
  [.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
   .numel output.region output.numel,
   .launch output.numel 1024 (gridCount (parameters x y output)),
   .returnTensor output.region]

/-- `empty_like(x)` precedes the assertion exactly as in the Python source.
The launch callback may reject unsupported dtype/device/metadata or fail. -/
def add {α σ : Type} (rt : Runtime α σ) (x y : Tensor)
    (initial : World α σ) : Run (World α σ) Tensor :=
  (perform (.emptyLike x.region) (rt.emptyLike x) initial).bind fun output allocated =>
    let check : Run (World α σ) Unit :=
      ⟨[.cudaCheck x.isCUDA y.isCUDA output.isCUDA],
       if cudaGuard x y output then .ok () allocated
       else .error .cudaAssertion allocated⟩
    check.bind fun _ checked =>
      let p := parameters x y output
      let launched := perform (.launch output.numel 1024 (gridCount p))
        (rt.launch x y output p) checked
      let result := launched.bind fun _ finished =>
        ⟨[.returnTensor output.region], .ok output finished⟩
      ⟨.numel output.region output.numel :: result.trace, result.result⟩

theorem allocation_error_first {α σ : Type} (rt : Runtime α σ)
    (x y : Tensor) (initial failed : World α σ) (reason : Failure)
    (h : rt.emptyLike x initial = .error reason failed) :
    add rt x y initial = ⟨[.emptyLike x.region], .error reason failed⟩ := by
  simp [add, perform, Run.bind, h]

theorem cuda_error_after_allocation {α σ : Type} (rt : Runtime α σ)
    (x y output : Tensor) (initial allocated : World α σ)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.false) :
    add rt x y initial =
      ⟨[.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA],
       .error .cudaAssertion allocated⟩ := by
  simp [add, perform, Run.bind, ha, hc]

theorem launch_error_after_metadata {α σ : Type} (rt : Runtime α σ)
    (x y output : Tensor) (initial allocated failed : World α σ) (reason : Failure)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (hl : rt.launch x y output (parameters x y output) allocated = .error reason failed) :
    add rt x y initial =
      ⟨[.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
         .numel output.region output.numel,
         .launch output.numel 1024 (gridCount (parameters x y output))],
       .error reason failed⟩ := by
  simp [add, perform, Run.bind, ha, hc, hl]

/-- An explicit logical instantiation of the launch primitive. It deliberately
ignores tensor dtype/device/layout metadata and executes the legacy scalar
kernel at the supplied logical regions. This is not an implementation of
Torch allocation or Triton's native launch; native callbacks receive all
three Tensor descriptors explicitly so they can dispatch or reject metadata. -/
def logicalLaunch {α σ : Type} [Inhabited α] (M : Algebra α)
    (_x _y _output : Tensor) (p : Parameters)
    (initial : World α σ) : Result (World α σ) Unit :=
  match runGrid M p initial.kernel with
  | none => .error .logicalInterpreter initial
  | some final => .ok () {initial with kernel := final}

theorem add_logical_correct {α σ : Type} [Inhabited α] (M : Algebra α)
    (rt : Runtime α σ) (x y output : Tensor) (initial allocated : World α σ)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true) :
    ∃ final,
      add {rt with launch := logicalLaunch M} x y initial =
        ⟨successTrace x y output, .ok output {allocated with kernel := final}⟩ ∧
      final.mem = gridExpectedMemory M (parameters x y output) allocated.kernel := by
  obtain ⟨final, he, hm⟩ := grid_correct M (parameters x y output) allocated.kernel
    (by change 0 < (1024 : Nat); decide)
  refine ⟨final, ?_, hm⟩
  simp [add, perform, Run.bind, logicalLaunch, ha, hc, he, successTrace]

/-- One harness case: x random draw, then y random draw, then the wrapper. -/
def testCase {α σ : Type} (rt : Runtime α σ) (size : Nat)
    (initial : World α σ) : Run (World α σ) Tensor :=
  (perform (.rand size "cuda") (rt.rand size "cuda") initial).bind fun x afterX =>
    (perform (.rand size "cuda") (rt.rand size "cuda") afterX).bind fun y afterY =>
      add rt x y afterY

def resultKeys : List String :=
  ["test_case_1", "test_case_2", "test_case_3", "test_case_4"]

/-- The four cases are sequential: no random draws for the next case occur
until the preceding `add` returns. Any primitive failure stops this run. -/
def testAdd {α σ : Type} (rt : Runtime α σ) (initial : World α σ) :
    Run (World α σ) (List (String × Tensor)) :=
  (perform (.manualSeed 0) (rt.manualSeed 0) initial).bind fun _ seeded =>
    (testCase rt 98432 seeded).bind fun first afterFirst =>
      (testCase rt 1024 afterFirst).bind fun second afterSecond =>
        (testCase rt 2048 afterSecond).bind fun third afterThird =>
          (testCase rt 4096 afterThird).bind fun fourth finished =>
            ⟨[.dictionary resultKeys],
             .ok [("test_case_1", first), ("test_case_2", second),
                  ("test_case_3", third), ("test_case_4", fourth)] finished⟩

/-- The source invokes `test_add` at module scope as `result_gold = test_add()`.
Successful assignment is modeled; Python import/runtime setup is external. -/
def moduleInitialization {α σ : Type} (rt : Runtime α σ) (initial : World α σ) :
    Run (World α σ) (List (String × Tensor)) :=
  (testAdd rt initial).bind fun results finished =>
    ⟨[.assign "result_gold"], .ok results finished⟩

theorem seed_error_stops_harness {α σ : Type} (rt : Runtime α σ)
    (initial failed : World α σ) (reason : Failure)
    (h : rt.manualSeed 0 initial = .error reason failed) :
    testAdd rt initial = ⟨[.manualSeed 0], .error reason failed⟩ := by
  simp [testAdd, perform, Run.bind, h]

theorem first_random_error_stops_case {α σ : Type} (rt : Runtime α σ)
    (size : Nat) (initial failed : World α σ) (reason : Failure)
    (h : rt.rand size "cuda" initial = .error reason failed) :
    testCase rt size initial = ⟨[.rand size "cuda"], .error reason failed⟩ := by
  simp [testCase, perform, Run.bind, h]

theorem second_random_error_stops_case {α σ : Type} (rt : Runtime α σ)
    (size : Nat) (x : Tensor) (initial afterX failed : World α σ) (reason : Failure)
    (hx : rt.rand size "cuda" initial = .ok x afterX)
    (hy : rt.rand size "cuda" afterX = .error reason failed) :
    testCase rt size initial =
      ⟨[.rand size "cuda", .rand size "cuda"], .error reason failed⟩ := by
  simp [testCase, perform, Run.bind, hx, hy]

/-- A postcondition constrains returns and permits explicit failures. -/
def Result.Post {S A : Type} (post : A → S → Prop) : Result S A → Prop
  | .error _ _ => True
  | .ok value state => post value state

theorem Run.bind_post {S A B : Type} (previous : Run S A)
    (next : A → S → Run S B) (post : B → S → Prop)
    (h : ∀ value state, Result.Post post (next value state).result) :
    Result.Post post (previous.bind next).result := by
  cases hp : previous.result with
  | error reason state => simp [Run.bind, hp, Result.Post]
  | ok value state => simpa [Run.bind, hp] using h value state

/-- A structural property of every harness outcome, without any assumption
that seed, random generation, allocation or launches succeed. -/
theorem harness_result_keys {α σ : Type} (rt : Runtime α σ) (initial : World α σ) :
    Result.Post (fun results _ => results.map Prod.fst = resultKeys)
      (testAdd rt initial).result := by
  unfold testAdd
  apply Run.bind_post
  intro _ seeded
  apply Run.bind_post
  intro first afterFirst
  apply Run.bind_post
  intro second afterSecond
  apply Run.bind_post
  intro third afterThird
  apply Run.bind_post
  intro fourth finished
  rfl

#specNonCircular successTrace avoiding [add, logicalLaunch, runGrid, FP.Structural.exec]
#axiomsClean allocation_error_first
#axiomsClean cuda_error_after_allocation
#axiomsClean launch_error_after_metadata
#axiomsClean add_logical_correct
#axiomsClean seed_error_stops_harness
#axiomsClean first_random_error_stops_case
#axiomsClean second_random_error_stops_case
#axiomsClean Run.bind_post
#axiomsClean harness_result_keys
end
end VeriTile.Bench.TritonBenchG.VectorAddition.HostWrapper

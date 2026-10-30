import bench.tritonbench_g.matrix_reduction.Proofs.HostMeta
import bench.tritonbench_g.matrix_reduction.Proofs.ConcreteCorrect

/-! Composition of the production `load_reduce`, under ordinary Torch primitive
contracts. The launch executes the shared source interpreter. Its native compiler,
index/address and signed-zero refinement boundary remains explicit. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Host
open VeriTile.Triton FP.Structural TypedSource MachineMax ConcreteCorrect
set_option linter.unusedSimpArgs false
noncomputable section

structure Allocated where
  tensor : Tensor
  state : State Word

structure Golden (storage : Storage) (M : Nat) where
  tensor : Tensor
  values : Fin M → Bits storage

/-- Fallible ordinary calls. `permitLaunch` exposes compilation/resource failure;
on admission, `run` executes the original shared kernel, not a result oracle.
No claim that every shape admits a native launch is made. -/
structure Ops (storage : Storage) (M N : Nat) where
  randn : Except String Allocated
  empty : Allocated → Except String Allocated
  permitLaunch : Launch → Except String Unit
  torchMax : Tensor → State Word → Nat → Except String (Golden storage M)
  setPrintOptions : String → Except String Unit
  assertClose : Tensor → State Word → Golden storage M → CloseOptions → Except String Unit

structure Completed (storage : Storage) (M : Nat) where
  input : Tensor
  output : Tensor
  state : State Word
  golden : Golden storage M

def run (storage : Storage) (rest : Algebra Word) (M N : Nat)
    (ops : Ops storage M N) : Except Failure (Completed storage M) := do
  let x ← checked .randn ops.randn
  let y ← checked .empty (ops.empty x)
  let _ ← checked .launch (ops.permitLaunch (launchCall x.tensor y.tensor M N))
  let t ← match FP.Structural.exec (backend rest)
      (kernel storage x.tensor.region y.tensor.region
        (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) M N) y.state with
    | none => .error ⟨.launch, "source execution failed"⟩
    | some t => .ok t
  let golden ← checked .torchMax (ops.torchMax x.tensor t 1)
  let _ ← checked .printOptions (ops.setPrintOptions "full")
  let _ ← checked .assertClose (ops.assertClose y.tensor t golden closeOptions)
  pure ⟨x.tensor, y.tensor, t, golden⟩

/-- Python returns None, not the output tensor or the reference vector. -/
def loadReduce (rest : Algebra Word) (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N) : Except Failure Unit := do
  let storage ← selectDType dtype
  let _ ← run storage rest M N (ops storage)
  pure ()

def InputValues (storage : Storage) (M N : Nat) (x : Tensor) (s : State Word)
    (xs : Fin M → Fin N → Bits storage) : Prop :=
  ∀ i j, s.mem x.region (i.val * x.stride 0 + j.val * x.stride 1) =
    .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (xs i j)))

/-- Ordinary successful randn/empty contracts. The generated bit values are
observations of randn, not a replacement random algorithm. No non-NaN or
distribution claim is hidden in allocation. Empty preserves exterior memory. -/
structure Setup (storage : Storage) (M N : Nat) (x y : Allocated)
    (xs : Fin M → Fin N → Bits storage) : Prop where
  x_shape : x.tensor.shape = [M, N]
  x_strides : x.tensor.strides = [max N 1, 1]
  x_storage : x.tensor.storage = storage
  x_cuda : x.tensor.cuda = Bool.true
  y_shape : y.tensor.shape = [M]
  y_strides : y.tensor.strides = [1]
  y_storage : y.tensor.storage = storage
  y_cuda : y.tensor.cuda = Bool.true
  same_device : y.tensor.device = x.tensor.device
  distinct : x.tensor.region ≠ y.tensor.region
  input : InputValues storage M N x.tensor x.state xs
  empty_frame : ∀ r o, r ≠ y.tensor.region → y.state.mem r o = x.state.mem r o

def VectorValues (storage : Storage) (M : Nat) (y : Tensor) (s : State Word)
    (values : Fin M → Bits storage) : Prop :=
  ∀ i, s.mem y.region i.val = .mk (channel storage).toTileDType
    (ofFloat (channel storage) (encode storage (values i)))

/-- These contracts concern ordinary Torch primitives on arbitrary tensors.
Neither mentions the Triton kernel, its execution or the wrapper conclusion.
Equality of non-NaN IEEE order keys is value equality, including signed zero. -/
structure TorchContracts (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) : Prop where
  max_values : ∀ x s golden xs,
    x.shape = [M, N] → x.storage = storage →
    InputValues storage M N x s xs →
    (∀ i j, isNaN storage (xs i j) = Bool.false) →
    ops.torchMax x s 1 = .ok golden →
    golden.tensor.shape = [M] ∧ golden.tensor.storage = storage ∧
      golden.tensor.cuda = x.cuda ∧ golden.tensor.device = x.device ∧
      (∀ i, IsMaximum storage (xs i) (golden.values i))
  assert_equal : ∀ y s golden actual,
    y.shape = [M] → y.strides = [1] → y.storage = storage →
    golden.tensor.shape = [M] → golden.tensor.storage = storage →
    y.cuda = golden.tensor.cuda → y.device = golden.tensor.device →
    VectorValues storage M y s actual →
    (∀ i, isNaN storage (actual i) = Bool.false) →
    (∀ i, isNaN storage (golden.values i) = Bool.false) →
    (∀ i, rank storage (actual i) = rank storage (golden.values i)) →
    ops.assertClose y s golden closeOptions = .ok ()

theorem maximum_rank_eq (storage : Storage) {N : Nat} (xs : Fin N → Bits storage)
    (a b : Bits storage) (ha : IsMaximum storage xs a) (hb : IsMaximum storage xs b) :
    rank storage a = rank storage b := by
  obtain ⟨⟨ia, rfl⟩, ha⟩ := ha
  obtain ⟨⟨ib, rfl⟩, hb⟩ := hb
  exact Nat.le_antisymm (hb ia) (ha ib)

def SuccessSpec (storage : Storage) (rest : Algebra Word) (M N : Nat)
    (x y : Allocated) (xs : Fin M → Fin N → Bits storage)
    (c : Completed storage M) : Prop :=
  c.input = x.tensor ∧ c.output = y.tensor ∧
  FP.Structural.exec (backend rest)
    (kernel storage x.tensor.region y.tensor.region
      (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) M N) y.state = some c.state ∧
  (∀ i : Fin M, ∃ v : Bits storage,
    c.state.mem y.tensor.region i.val = .mk (channel storage).toTileDType
      (ofFloat (channel storage) (encode storage v)) ∧
    IsMaximum storage (xs i) v ∧ isNaN storage v = Bool.false) ∧
  (∀ r o, r ≠ y.tensor.region ∨ M ≤ o → c.state.mem r o = y.state.mem r o) ∧
  c.golden.tensor.shape = [M] ∧ c.golden.tensor.storage = storage ∧
  c.golden.tensor.cuda = c.input.cuda ∧ c.golden.tensor.device = c.input.device ∧
  (∀ i, IsMaximum storage (xs i) (c.golden.values i))

/-- Successful allocations are connected to the actual source execution and
Torch reference. Under non-NaN observations, an admitted launch produces the
same values as torch.max; therefore assert_close cannot reject those values.
Compilation/resource, torch.max and print-option failures still propagate. -/
theorem source_wrapper_outcomes (storage : Storage) (rest : Algebra Word)
    (M N : Nat) (ops : Ops storage M N) (x y : Allocated)
    (xs : Fin M → Fin N → Bits storage) (hN : 0 < N)
    (hx : ops.randn = .ok x) (hy : ops.empty x = .ok y)
    (setup : Setup storage M N x y xs)
    (torch : TorchContracts storage M N ops)
    (hnan : ∀ i j, isNaN storage (xs i j) = Bool.false) :
    match run storage rest M N ops with
    | .error e => e.stage = .launch ∨ e.stage = .torchMax ∨ e.stage = .printOptions
    | .ok c => SuccessSpec storage rest M N x y xs c := by
  have hin : InputValues storage M N x.tensor y.state xs := by
    intro i j
    rw [setup.empty_frame _ _ setup.distinct]
    exact setup.input i j
  obtain ⟨t, he, hv, hf⟩ := source_correct storage rest y.state
    x.tensor.region y.tensor.region (x.tensor.stride 0) (x.tensor.stride 1)
    (y.tensor.stride 0) M N hN xs hin hnan
  have hinput : InputValues storage M N x.tensor t xs := by
    intro i j
    rw [hf _ _ (Or.inl setup.distinct)]
    exact hin i j
  cases hl : ops.permitLaunch (launchCall x.tensor y.tensor M N) with
  | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, hl]
  | ok u =>
    cases u
    cases hm : ops.torchMax x.tensor t 1 with
    | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, hl, he, hm]
    | ok golden =>
      obtain ⟨hshape, hstorage, hcuda, hdevice, hgold⟩ := torch.max_values x.tensor t golden xs
        setup.x_shape setup.x_storage hinput hnan hm
      let actual : Fin M → Bits storage := fun i => (hv i).choose
      have hav i := (hv i).choose_spec
      have hactual : VectorValues storage M y.tensor t actual := fun i => (hav i).1
      have hnactual : ∀ i, isNaN storage (actual i) = Bool.false := fun i => (hav i).2.2
      have hngold : ∀ i, isNaN storage (golden.values i) = Bool.false := by
        intro i
        obtain ⟨j, hj⟩ := (hgold i).1
        rw [hj]
        exact hnan i j
      have hequal : ∀ i, rank storage (actual i) = rank storage (golden.values i) :=
        fun i => maximum_rank_eq storage (xs i) (actual i) (golden.values i)
          (hav i).2.1 (hgold i)
      have ha := torch.assert_equal y.tensor t golden actual setup.y_shape setup.y_strides setup.y_storage
        hshape hstorage (setup.y_cuda.trans (setup.x_cuda.symm.trans hcuda.symm))
        (setup.same_device.trans hdevice.symm) hactual hnactual hngold hequal
      cases hp : ops.setPrintOptions "full" with
      | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, hl, he, hm, hp]
      | ok u =>
        cases u
        simp only [run, checked, hx, hy, hl, he, hm, hp, ha,
          Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq]
        exact ⟨rfl, rfl, he, hv, hf, hshape, hstorage, hcuda, hdevice, hgold⟩

theorem randn_error (storage : Storage) (rest : Algebra Word) (M N : Nat)
    (ops : Ops storage M N) (message : String) (h : ops.randn = .error message) :
    run storage rest M N ops = .error ⟨.randn, message⟩ := by
  simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, h]

theorem empty_error (storage : Storage) (rest : Algebra Word) (M N : Nat)
    (ops : Ops storage M N) (x : Allocated) (message : String)
    (hx : ops.randn = .ok x) (hy : ops.empty x = .error message) :
    run storage rest M N ops = .error ⟨.empty, message⟩ := by
  simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy]

theorem empty_axis (storage : Storage) (rest : Algebra Word) (s : State Word)
    (X Y : RegionName) (sm sn sy M : Nat) :
    FP.Structural.exec (backend rest) (kernel storage X Y sm sn sy M 0) s = none := by
  cases storage <;>
    simp [kernel, source16, source32, ComputeKernel.surfaceBody, FP.Structural.exec, FP.Structural.run,
      step, evalExpr, evalComputeOp, evalOp_unfold, State.setReg, TileShape.axisDim, Option.bind]

/-- Random generation's non-NaN property is an explicit numerical input-domain
hypothesis. It is separate from the ordinary allocation/metadata contracts. -/
def SamplesNonNaN (storage : Storage) (M N : Nat) (ops : Ops storage M N) : Prop :=
  ∀ x y xs, ops.randn = .ok x → ops.empty x = .ok y →
    Setup storage M N x y xs → ∀ i j, isNaN storage (xs i j) = Bool.false

def AllocationContracts (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) : Prop :=
  ∀ x y, ops.randn = .ok x → ops.empty x = .ok y →
    ∃ xs, Setup storage M N x y xs

def OutcomeSpec (storage : Storage) (rest : Algebra Word) (M N : Nat)
    (ops : Ops storage M N) : Except Failure (Completed storage M) → Prop
  | .error e => e.stage = .randn ∨ e.stage = .empty ∨ e.stage = .launch ∨
      e.stage = .torchMax ∨ e.stage = .printOptions
  | .ok c => ∃ x y xs, ops.randn = .ok x ∧ ops.empty x = .ok y ∧
      Setup storage M N x y xs ∧ SuccessSpec storage rest M N x y xs c

theorem full_source_outcomes (storage : Storage) (rest : Algebra Word)
    (M N : Nat) (ops : Ops storage M N)
    (alloc : AllocationContracts storage M N ops)
    (torch : TorchContracts storage M N ops)
    (samples : SamplesNonNaN storage M N ops) :
    OutcomeSpec storage rest M N ops (run storage rest M N ops) := by
  cases hx : ops.randn with
  | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, OutcomeSpec]
  | ok x =>
    cases hy : ops.empty x with
    | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, OutcomeSpec]
    | ok y =>
      obtain ⟨xs, setup⟩ := alloc x y hx hy
      by_cases hN : 0 < N
      · have h := source_wrapper_outcomes storage rest M N ops x y xs hN hx hy setup
          torch (samples x y xs hx hy setup)
        cases hr : run storage rest M N ops with
        | error e =>
          rw [hr] at h
          exact Or.inr (Or.inr h)
        | ok c =>
          rw [hr] at h
          exact ⟨x, y, xs, hx, hy, setup, h⟩
      · have hn : N = 0 := by omega
        subst N
        cases hl : ops.permitLaunch (launchCall x.tensor y.tensor M 0) with
        | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, hl, OutcomeSpec]
        | ok u =>
          cases u
          simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, hl, empty_axis, OutcomeSpec]

theorem unknown_dtype (rest : Algebra Word) (M N : Nat) (name : String)
    (ops : (storage : Storage) → Ops storage M N)
    (h16 : name ≠ "float16") (h32 : name ≠ "float32") :
    loadReduce rest M N name ops = .error ⟨.dtype, name⟩ := by
  simp [loadReduce, selectDType, h16, h32, Except.bind, bind, pure]

def FunctionSpec (rest : Algebra Word) (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N) (result : Except Failure Unit) : Prop :=
  match selectDType dtype with
  | .error e => result = .error e
  | .ok storage =>
      OutcomeSpec storage rest M N (ops storage) (run storage rest M N (ops storage)) ∧
      result = (run storage rest M N (ops storage)).map (fun _ => ())

/-- Dispatch is part of the production function: both literal dtype keys use
their source profile, every other key fails, and successful completion returns
None. The per-profile outcome includes the actual maximum and memory frame. -/
theorem load_reduce_outcomes (rest : Algebra Word) (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N)
    (alloc : ∀ storage, AllocationContracts storage M N (ops storage))
    (torch : ∀ storage, TorchContracts storage M N (ops storage))
    (samples : ∀ storage, SamplesNonNaN storage M N (ops storage)) :
    FunctionSpec rest M N dtype ops (loadReduce rest M N dtype ops) := by
  cases hd : selectDType dtype with
  | error e => simp [FunctionSpec, hd, loadReduce, Except.bind, bind, pure]
  | ok storage =>
    simp only [FunctionSpec, hd]
    constructor
    · exact full_source_outcomes storage rest M N (ops storage)
        (alloc storage) (torch storage) (samples storage)
    · simp [loadReduce, hd, Except.bind, Except.map, Except.pure, bind, pure, Functor.map]

#axiomsClean source_wrapper_outcomes
#axiomsClean maximum_rank_eq
#axiomsClean randn_error
#axiomsClean empty_error
#axiomsClean unknown_dtype
#axiomsClean empty_axis
#axiomsClean full_source_outcomes
#axiomsClean load_reduce_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Host

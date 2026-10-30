import bench.tritonbench_g.matrix_reduction.Proofs.NativeMemory

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton TypedSource MachineMax ConcreteCorrect
open Host (Tensor Golden Failure Stage checked closeOptions CloseOptions Launch launchCall selectDType)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure Allocated (S : Type) where
  tensor : Tensor
  base : Nat
  state : S

def layout (d : Storage) (M N : Nat) (x y : Allocated S) : Layout d M N := ⟨x.base,y.base⟩

/-- Fallible ordinary Python/Torch/runtime calls, with an admitted launch
expanded into primitive loads, reduction trees, casts and row stores below.
Admission certifies structural frontend shape constraints only. It supplies
neither a completed memory nor an output-value oracle. -/
structure Ops (S : Type) (d : Storage) (M N : Nat) where
  randn : Except String (Allocated S)
  empty : Allocated S → Except String (Allocated S)
  admit : Launch → Except String (Admitted M N)
  schedule : Admitted M N → Schedule M N
  torchMax : Tensor → Nat → S → Nat → Except String (Golden d M)
  setPrintOptions : String → Except String Unit
  assertClose : Tensor → Nat → S → Golden d M → CloseOptions → Except String Unit

structure Completed (S : Type) (d : Storage) (M : Nat) where
  input : Allocated S
  output : Allocated S
  state : S
  golden : Golden d M

def run (mem : Memory S d) (num : Numerics) (M N : Nat) (ops : Ops S d M N) : Except Failure (Completed S d M) := do
  let x ← checked .randn ops.randn
  let y ← checked .empty (ops.empty x)
  let admitted ← checked .launch (ops.admit (launchCall x.tensor y.tensor M N))
  let t ← checked .launch (sourceKernel mem num admitted (layout d M N x y) (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) (ops.schedule admitted) y.state)
  let golden ← checked .torchMax (ops.torchMax x.tensor x.base t 1)
  let _ ← checked .printOptions (ops.setPrintOptions "full")
  let _ ← checked .assertClose (ops.assertClose y.tensor y.base t golden closeOptions)
  pure ⟨x,y,t,golden⟩

/-- Successful Torch constructors give contiguous selected-dtype storage,
actual nonwrapping physical spans on the same CUDA device, fresh output, and
ordinary exterior-memory preservation. Random bit observations are unconstrained
here; non-NaN is a separate numerical premise. -/
structure Setup (mem : Memory S d) (M N : Nat) (x y : Allocated S)
    (xs : Fin M → Fin N → Bits d) : Prop where
  x_shape : x.tensor.shape = [M,N]
  x_strides : x.tensor.strides = [max N 1,1]
  x_storage : x.tensor.storage = d
  x_cuda : x.tensor.cuda = Bool.true
  y_shape : y.tensor.shape = [M]
  y_strides : y.tensor.strides = [1]
  y_storage : y.tensor.storage = d
  y_cuda : y.tensor.cuda = Bool.true
  same_device : y.tensor.device = x.tensor.device
  valid : (layout d M N x y).Valid
  input : Input mem (layout d M N x y) x.state xs
  output_bytes : ∀ a, y.base ≤ a → a < y.base+bytes d*M → (mem.byte y.state a).isSome
  empty_load_frame : ∀ a, a+bytes d ≤ y.base ∨ y.base+bytes d*M ≤ a → mem.load y.state a = mem.load x.state a
  empty_byte_frame : ∀ a, a < y.base ∨ y.base+bytes d*M ≤ a → mem.byte y.state a = mem.byte x.state a

/-- Trusted Torch max/assert contracts with the actual physical input/output
addresses. Defaults include device checks even though dtype checks are disabled.
Equal non-NaN value ranks include opposite signed-zero representations. -/
structure TorchContracts (mem : Memory S d) (M N : Nat) (ops : Ops S d M N) : Prop where
  max_values : ∀ x base s golden (xs : Fin M → Fin N → Bits d),
    x.shape = [M,N] → x.strides = [max N 1,1] → x.storage = d →
    (∀ i j, mem.load s (base+bytes d*(i.val*N+j.val)) = some (xs i j)) →
    (∀ i j, isNaN d (xs i j) = Bool.false) →
    ops.torchMax x base s 1 = .ok golden →
    golden.tensor.shape = [M] ∧ golden.tensor.storage = d ∧
    golden.tensor.cuda = x.cuda ∧ golden.tensor.device = x.device ∧
    (∀ i, IsMaximum d (xs i) (golden.values i))
  assert_equal : ∀ y base s golden (actual : Fin M → Bits d),
    y.shape = [M] → y.strides = [1] → y.storage = d →
    golden.tensor.shape = [M] → golden.tensor.storage = d →
    y.cuda = golden.tensor.cuda → y.device = golden.tensor.device →
    (∀ i, mem.load s (base+bytes d*i.val) = some (actual i)) →
    (∀ i, isNaN d (actual i) = Bool.false) →
    (∀ i, isNaN d (golden.values i) = Bool.false) →
    (∀ i, rank d (actual i) = rank d (golden.values i)) →
    ops.assertClose y base s golden closeOptions = .ok ()

def SuccessSpec (mem : Memory S d) (num : Numerics) (M N : Nat) (ops : Ops S d M N)
    (x y : Allocated S) (xs : Fin M → Fin N → Bits d) (c : Completed S d M) : Prop :=
  c.input=x ∧ c.output=y ∧
  (∃ admitted, ops.admit (launchCall x.tensor y.tensor M N) = .ok admitted ∧
    sourceKernel mem num admitted (layout d M N x y) (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) (ops.schedule admitted) y.state = .ok c.state) ∧
  KernelSpec mem (layout d M N x y) y.state c.state xs ∧
  (∀ a, a < y.base ∨ y.base+bytes d*M ≤ a → mem.byte c.state a = mem.byte x.state a) ∧
  c.golden.tensor.shape=[M] ∧ c.golden.tensor.storage=d ∧
  c.golden.tensor.cuda=x.tensor.cuda ∧ c.golden.tensor.device=x.tensor.device ∧
  (∀ i, IsMaximum d (xs i) (c.golden.values i))

theorem allocated_outcomes (mem : Memory S d) (num : Numerics) (laws : NumericalContracts num)
    (M N : Nat) (ops : Ops S d M N) (x y : Allocated S) (xs : Fin M → Fin N → Bits d)
    (hx : ops.randn = .ok x) (hy : ops.empty x = .ok y) (setup : Setup mem M N x y xs)
    (torch : TorchContracts mem M N ops) (hn : ∀ i j, isNaN d (xs i j) = Bool.false) :
    match run mem num M N ops with
    | .error e => e.stage=.launch ∨ e.stage=.torchMax ∨ e.stage=.printOptions
    | .ok c => SuccessSpec mem num M N ops x y xs c := by
  have input : Input mem (layout d M N x y) y.state xs := by
    intro i j
    have bounds := input_inside (layout d M N x y) i j
    have fresh := setup.valid.fresh
    have outside : (layout d M N x y).xAddress i j+bytes d ≤ y.base ∨
        y.base+bytes d*M ≤ (layout d M N x y).xAddress i j := by
      change x.base+bytes d*(M*N) ≤ y.base ∨ y.base+bytes d*M ≤ x.base at fresh
      change x.base ≤ _ ∧ _ ≤ x.base+bytes d*(M*N) at bounds
      omega
    rw [setup.empty_load_frame _ outside]
    exact setup.input i j
  cases hl : ops.admit (launchCall x.tensor y.tensor M N) with
  | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl]
  | ok admitted =>
    obtain ⟨t,hkc,hs⟩ := kernel_correct mem num laws admitted (layout d M N x y) setup.valid
      (ops.schedule admitted) y.state xs input hn
    have hk : sourceKernel mem num admitted (layout d M N x y) (x.tensor.stride 0)
        (x.tensor.stride 1) (y.tensor.stride 0) (ops.schedule admitted) y.state = .ok t := by
      simpa [kernel, source16, source32,Tensor.stride,setup.x_strides,setup.y_strides] using hkc
    cases hm : ops.torchMax x.tensor x.base t 1 with
    | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl,hk,hm]
    | ok golden =>
      obtain ⟨hshape,hstorage,hcuda,hdevice,hgold⟩ := torch.max_values x.tensor x.base t golden xs
        setup.x_shape setup.x_strides setup.x_storage hs.2.1 hn hm
      let actual : Fin M → Bits d := fun i => (hs.1 i).choose
      have ha i := (hs.1 i).choose_spec
      have hngold : ∀ i, isNaN d (golden.values i) = Bool.false := by
        intro i
        obtain ⟨j,hj⟩ := (hgold i).1
        rw [hj]
        exact hn i j
      have hequal : ∀ i, rank d (actual i) = rank d (golden.values i) :=
        fun i => Host.maximum_rank_eq d (xs i) (actual i) (golden.values i) (ha i).2.1 (hgold i)
      have hassert := torch.assert_equal y.tensor y.base t golden actual setup.y_shape setup.y_strides setup.y_storage
        hshape hstorage (setup.y_cuda.trans (setup.x_cuda.symm.trans hcuda.symm))
        (setup.same_device.trans hdevice.symm) (fun i => (ha i).1)
        (fun i => (ha i).2.2.1) hngold hequal
      cases hp : ops.setPrintOptions "full" with
      | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl,hk,hm,hp]
      | ok u =>
        cases u
        simp only [run,checked,hx,hy,hl,hk,hm,hp,hassert,
          Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq]
        refine ⟨rfl,rfl,⟨admitted,hl,hk⟩,hs,?_,hshape,hstorage,hcuda,hdevice,hgold⟩
        intro a outside
        exact (hs.2.2 a outside).trans (setup.empty_byte_frame a outside)

def AllocationContracts (mem : Memory S d) (M N : Nat) (ops : Ops S d M N) : Prop :=
  ∀ x y, ops.randn=.ok x → ops.empty x=.ok y → ∃ xs, Setup mem M N x y xs
/-- The exact deferred RNG premise: every observed FP16/FP32 input sample of
successful source randn/empty calls is non-NaN. No distribution claim is used. -/
def SamplesNonNaN (mem : Memory S d) (M N : Nat) (ops : Ops S d M N) : Prop :=
  ∀ x y xs, ops.randn=.ok x → ops.empty x=.ok y → Setup mem M N x y xs →
    ∀ i j, isNaN d (xs i j) = Bool.false

def OutcomeSpec (mem : Memory S d) (num : Numerics) (M N : Nat) (ops : Ops S d M N) :
    Except Failure (Completed S d M) → Prop
  | .error e => e.stage=.randn ∨ e.stage=.empty ∨ e.stage=.launch ∨ e.stage=.torchMax ∨ e.stage=.printOptions
  | .ok c => ∃ x y xs, ops.randn=.ok x ∧ ops.empty x=.ok y ∧ Setup mem M N x y xs ∧ SuccessSpec mem num M N ops x y xs c

theorem full_native_outcomes (mem : Memory S d) (num : Numerics) (laws : NumericalContracts num)
    (M N : Nat) (ops : Ops S d M N) (alloc : AllocationContracts mem M N ops)
    (torch : TorchContracts mem M N ops) (samples : SamplesNonNaN mem M N ops) :
    OutcomeSpec mem num M N ops (run mem num M N ops) := by
  cases hx : ops.randn with
  | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,OutcomeSpec]
  | ok x =>
    cases hy : ops.empty x with
    | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,OutcomeSpec]
    | ok y =>
      obtain ⟨xs,setup⟩ := alloc x y hx hy
      have h := allocated_outcomes mem num laws M N ops x y xs hx hy setup torch (samples x y xs hx hy setup)
      cases hr : run mem num M N ops with
      | error e => rw [hr] at h; exact Or.inr (Or.inr h)
      | ok c => rw [hr] at h; exact ⟨x,y,xs,hx,hy,setup,h⟩

def loadReduce (mem : (d : Storage) → Memory S d) (num : Numerics) (M N : Nat) (dtype : String)
    (ops : (d : Storage) → Ops S d M N) : Except Failure Unit := do
  let d ← selectDType dtype
  let _ ← run (mem d) num M N (ops d)
  pure ()

def FunctionSpec (mem : (d : Storage) → Memory S d) (num : Numerics) (M N : Nat) (dtype : String)
    (ops : (d : Storage) → Ops S d M N) (result : Except Failure Unit) : Prop :=
  match selectDType dtype with
  | .error e => result=.error e
  | .ok d => OutcomeSpec (mem d) num M N (ops d) (run (mem d) num M N (ops d)) ∧
      result=(run (mem d) num M N (ops d)).map (fun _ => ())


#axiomsClean allocated_outcomes
#axiomsClean full_native_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native

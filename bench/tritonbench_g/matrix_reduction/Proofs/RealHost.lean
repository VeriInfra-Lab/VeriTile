import bench.tritonbench_g.matrix_reduction.Proofs.Maximum
import bench.tritonbench_g.matrix_reduction.Proofs.HostMeta
import VeriTile.Meta.StatementAudit

/-! Independent mathematical interpretation of the entire `load_reduce` wrapper.
The original dtype dispatcher and ordinary Torch metadata are shared; values are
real numbers and the shared original kernel executes in the algorithm semantics.
No typed/native theorem or FP equivalence result is imported. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.RealHost
open VeriTile.Triton TypedSource Host KernelMath
export KernelMath (IsMaximum maximum_eq reduction_maximum)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure Allocated where
  tensor : Tensor
  state : BlockState

structure Golden (M : Nat) where
  tensor : Tensor
  values : Fin M → ℝ

structure Ops (storage : Storage) (M N : Nat) where
  randn : Except String Allocated
  empty : Allocated → Except String Allocated
  permitLaunch : Launch → Except String Unit
  torchMax : Tensor → BlockState → Nat → Except String (Golden M)
  setPrintOptions : String → Except String Unit
  assertClose : Tensor → BlockState → Golden M → CloseOptions → Except String Unit

structure Completed (M : Nat) where
  input : Tensor
  output : Tensor
  state : BlockState
  golden : Golden M

def run (storage : Storage) (M N : Nat) (ops : Ops storage M N) :
    Except Failure (Completed M) := do
  let x ← checked .randn ops.randn
  let y ← checked .empty (ops.empty x)
  let _ ← checked .launch (ops.permitLaunch (launchCall x.tensor y.tensor M N))
  let t ← match exec (load_reduce_kernel x.tensor.region y.tensor.region
      (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) M N) y.state with
    | none => .error ⟨.launch, "source execution failed"⟩
    | some t => .ok t
  let golden ← checked .torchMax (ops.torchMax x.tensor t 1)
  let _ ← checked .printOptions (ops.setPrintOptions "full")
  let _ ← checked .assertClose (ops.assertClose y.tensor t golden closeOptions)
  pure ⟨x.tensor,y.tensor,t,golden⟩

def loadReduce (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N) : Except Failure Unit := do
  let storage ← selectDType dtype
  let _ ← run storage M N (ops storage)
  pure ()

def InputValues (M N : Nat) (x : Tensor) (s : BlockState)
    (xs : Fin M → Fin N → ℝ) : Prop :=
  ∀ i j, s.readMem x.region (i.val*x.stride 0+j.val*x.stride 1)=xs i j

def VectorValues (M : Nat) (y : Tensor) (s : BlockState)
    (values : Fin M → ℝ) : Prop := ∀ i, s.readMem y.region i.val=values i

structure Setup (storage : Storage) (M N : Nat) (x y : Allocated)
    (xs : Fin M → Fin N → ℝ) : Prop where
  x_shape : x.tensor.shape=[M,N]
  x_strides : x.tensor.strides=[max N 1,1]
  x_storage : x.tensor.storage=storage
  x_cuda : x.tensor.cuda=Bool.true
  y_shape : y.tensor.shape=[M]
  y_strides : y.tensor.strides=[1]
  y_storage : y.tensor.storage=storage
  y_cuda : y.tensor.cuda=Bool.true
  same_device : y.tensor.device=x.tensor.device
  distinct : x.tensor.region≠y.tensor.region
  input : InputValues M N x.tensor x.state xs
  empty_frame : ∀ r o, r≠y.tensor.region → y.state.mem r o=x.state.mem r o

/-- Ordinary successful Torch primitive contracts over real values. The max
contract reads the supplied strides. Assertion uses the proved contiguous
output layout, same-device metadata, and exact equality of values. -/
structure TorchContracts (storage : Storage) (M N : Nat) (ops : Ops storage M N) : Prop where
  max_values : ∀ x s golden xs,
    x.shape=[M,N] → x.storage=storage → InputValues M N x s xs →
    ops.torchMax x s 1=.ok golden →
    golden.tensor.shape=[M] ∧ golden.tensor.storage=storage ∧
      golden.tensor.cuda=x.cuda ∧ golden.tensor.device=x.device ∧
      ∀ i, IsMaximum (xs i) (golden.values i)
  assert_equal : ∀ y s golden actual,
    y.shape=[M] → y.strides=[1] → y.storage=storage →
    golden.tensor.shape=[M] → golden.tensor.storage=storage →
    y.cuda=golden.tensor.cuda → y.device=golden.tensor.device →
    VectorValues M y s actual → (∀ i, actual i=golden.values i) →
    ops.assertClose y s golden closeOptions=.ok ()

theorem source_correct (M N : Nat) (hN : 0<N) (x y : Tensor) (s : BlockState)
    (xs : Fin M → Fin N → ℝ) (input : InputValues M N x s xs) :
    ∃ t, exec (load_reduce_kernel x.region y.region (x.stride 0) (x.stride 1)
      (y.stride 0) M N) s=some t ∧
      (∀ i : Fin M, IsMaximum (xs i) (t.readMem y.region i.val)) ∧
      (∀ r o, r≠y.region ∨ M≤o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := matrix_reduction_region_run x.region y.region
    (x.stride 0) (x.stride 1) (y.stride 0) M N hN s (fun k => xs k.1 k.2.1)
    (fun k => input k.1 k.2.1)
  refine ⟨t,he,?_,?_⟩
  · intro i
    rw [hv (i,⟨0,hN⟩,PUnit.unit)]
    exact reduction_maximum M N hN xs i
  · intro r o h
    apply hf r o
    rcases h with h|h
    · exact Or.inl h
    · exact Or.inr fun i => by omega

def SuccessSpec (storage : Storage) (M N : Nat) (x y : Allocated)
    (xs : Fin M → Fin N → ℝ) (c : Completed M) : Prop :=
  c.input=x.tensor ∧ c.output=y.tensor ∧
  exec (load_reduce_kernel x.tensor.region y.tensor.region
    (x.tensor.stride 0) (x.tensor.stride 1) (y.tensor.stride 0) M N) y.state=some c.state ∧
  (∀ i : Fin M, IsMaximum (xs i) (c.state.readMem y.tensor.region i.val)) ∧
  (∀ r o, r≠y.tensor.region ∨ M≤o → c.state.mem r o=y.state.mem r o) ∧
  c.golden.tensor.shape=[M] ∧ c.golden.tensor.storage=storage ∧
  c.golden.tensor.cuda=c.input.cuda ∧ c.golden.tensor.device=c.input.device ∧
  (∀ i, IsMaximum (xs i) (c.golden.values i)) ∧
  (∀ i, c.state.readMem y.tensor.region i.val=c.golden.values i)

theorem source_wrapper_outcomes (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) (x y : Allocated) (xs : Fin M → Fin N → ℝ)
    (hN : 0<N) (hx : ops.randn=.ok x) (hy : ops.empty x=.ok y)
    (setup : Setup storage M N x y xs) (torch : TorchContracts storage M N ops) :
    match run storage M N ops with
    | .error e => e.stage=.launch ∨ e.stage=.torchMax ∨ e.stage=.printOptions
    | .ok c => SuccessSpec storage M N x y xs c := by
  have hin : InputValues M N x.tensor y.state xs := by
    intro i j
    rw [BlockState.readMem,setup.empty_frame _ _ setup.distinct]
    exact setup.input i j
  obtain ⟨t,he,hv,hf⟩ := source_correct M N hN x.tensor y.tensor y.state xs hin
  have hinput : InputValues M N x.tensor t xs := by
    intro i j
    rw [BlockState.readMem,hf _ _ (Or.inl setup.distinct)]
    exact hin i j
  cases hl : ops.permitLaunch (launchCall x.tensor y.tensor M N) with
  | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl]
  | ok u =>
    cases u
    cases hm : ops.torchMax x.tensor t 1 with
    | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl,he,hm]
    | ok golden =>
      obtain ⟨hshape,hstorage,hcuda,hdevice,hgold⟩ := torch.max_values x.tensor t golden xs
        setup.x_shape setup.x_storage hinput hm
      have hequal : ∀ i, t.readMem y.tensor.region i.val=golden.values i :=
        fun i => maximum_eq (xs i) _ _ (hv i) (hgold i)
      have ha := torch.assert_equal y.tensor t golden (fun i => t.readMem y.tensor.region i.val)
        setup.y_shape setup.y_strides setup.y_storage hshape hstorage
        (setup.y_cuda.trans (setup.x_cuda.symm.trans hcuda.symm))
        (setup.same_device.trans hdevice.symm) (fun _ => rfl) hequal
      cases hp : ops.setPrintOptions "full" with
      | error message => simp [run,checked,Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq,hx,hy,hl,he,hm,hp]
      | ok u =>
        cases u
        simp only [run,checked,hx,hy,hl,he,hm,hp,ha,
          Except.mapError,Except.bind,Except.map,bind,pure,Functor.map,SeqRight.seqRight,Seq.seq]
        exact ⟨rfl,rfl,he,hv,hf,hshape,hstorage,hcuda,hdevice,hgold,hequal⟩

theorem randn_error (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) (message : String) (h : ops.randn = .error message) :
    run storage M N ops = .error ⟨.randn, message⟩ := by
  simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, h]

theorem empty_error (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) (x : Allocated) (message : String)
    (hx : ops.randn = .ok x) (hy : ops.empty x = .error message) :
    run storage M N ops = .error ⟨.empty, message⟩ := by
  simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy]

theorem empty_axis (s : BlockState) (X Y : RegionName) (sm sn sy M : Nat) :
    exec (load_reduce_kernel X Y sm sn sy M 0) s = none := by
  simp [exec, mathematical_projection, LegacyBody.load_reduce_kernel,stepStmts,stepStmt,evalOp.eq_def,
    Tile.reduceMax,Tile.reduceMaxDrop,TileShape.axisDim,TileShape.eraseAxis]

def AllocationContracts (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) : Prop :=
  ∀ x y, ops.randn = .ok x → ops.empty x = .ok y →
    ∃ xs, Setup storage M N x y xs

def OutcomeSpec (storage : Storage) (M N : Nat)
    (ops : Ops storage M N) : Except Failure (Completed M) → Prop
  | .error e => e.stage = .randn ∨ e.stage = .empty ∨ e.stage = .launch ∨
      e.stage = .torchMax ∨ e.stage = .printOptions
  | .ok c => ∃ x y xs, ops.randn = .ok x ∧ ops.empty x = .ok y ∧
      Setup storage M N x y xs ∧ SuccessSpec storage M N x y xs c

theorem full_source_outcomes (storage : Storage) (M N : Nat) (ops : Ops storage M N)
    (alloc : AllocationContracts storage M N ops)
    (torch : TorchContracts storage M N ops)
    :
    OutcomeSpec storage M N ops (run storage M N ops) := by
  cases hx : ops.randn with
  | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, OutcomeSpec]
  | ok x =>
    cases hy : ops.empty x with
    | error message => simp [run, checked, Except.mapError, Except.bind, Except.map, bind, pure, Functor.map, SeqRight.seqRight, Seq.seq, hx, hy, OutcomeSpec]
    | ok y =>
      obtain ⟨xs, setup⟩ := alloc x y hx hy
      by_cases hN : 0 < N
      · have h := source_wrapper_outcomes storage M N ops x y xs hN hx hy setup
          torch
        cases hr : run storage M N ops with
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

theorem unknown_dtype (M N : Nat) (name : String)
    (ops : (storage : Storage) → Ops storage M N)
    (h16 : name ≠ "float16") (h32 : name ≠ "float32") :
    loadReduce M N name ops = .error ⟨.dtype, name⟩ := by
  simp [loadReduce, selectDType, h16, h32, Except.bind, bind, pure]

def FunctionSpec (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N) (result : Except Failure Unit) : Prop :=
  match selectDType dtype with
  | .error e => result = .error e
  | .ok storage =>
      OutcomeSpec storage M N (ops storage) (run storage M N (ops storage)) ∧
      result = (run storage M N (ops storage)).map (fun _ => ())

/-- Dispatch is part of the production function: both literal dtype keys use
their source profile, every other key fails, and successful completion returns
None. The per-profile outcome includes the actual maximum and memory frame. -/
theorem load_reduce_outcomes (M N : Nat) (dtype : String)
    (ops : (storage : Storage) → Ops storage M N)
    (alloc : ∀ storage, AllocationContracts storage M N (ops storage))
    (torch : ∀ storage, TorchContracts storage M N (ops storage))
    :
    FunctionSpec M N dtype ops (loadReduce M N dtype ops) := by
  cases hd : selectDType dtype with
  | error e => simp [FunctionSpec, hd, loadReduce, Except.bind, bind, pure]
  | ok storage =>
    simp only [FunctionSpec, hd]
    constructor
    · exact full_source_outcomes storage M N (ops storage)
        (alloc storage) (torch storage)
    · simp [loadReduce, hd, Except.bind, Except.map, Except.pure, bind, pure, Functor.map]

#axiomsClean randn_error
#axiomsClean empty_error
#axiomsClean empty_axis
#axiomsClean unknown_dtype
#axiomsClean full_source_outcomes
#axiomsClean load_reduce_outcomes
#axiomsClean maximum_eq
#axiomsClean reduction_maximum
#axiomsClean source_correct
#axiomsClean source_wrapper_outcomes
#specNonCircular IsMaximum avoiding [exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.RealHost

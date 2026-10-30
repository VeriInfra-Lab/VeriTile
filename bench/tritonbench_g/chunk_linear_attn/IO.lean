import bench.tritonbench_g.chunk_linear_attn.IOLayout
import bench.tritonbench_g.chunk_linear_attn.ReductionIO
import bench.tritonbench_g.chunk_linear_attn.Candidate
import bench.tritonbench_g.chunk_linear_attn.Mathematical
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IO
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section

/-- The original quotation, primitive backend and complete launch geometry. -/
def invocation {α : Type} (M : Algebra α) (D : DotPrimitive α) (stage : Stage)
    (p : Parameters) (scale : ℝ) (seed final : Bool) (pid : PID) : Invocation α where
  kernel := stageKernel stage p scale seed final
  backend := engine (withDot M D)
  clearRegisters := Bool.true
  pids := fun n => match n with | 0 => pid.x | 1 => pid.y | _ => pid.z
  numPids := some (fun n => match n with
    | 0 => (gridSize stage p).x | 1 => (gridSize stage p).y | _ => (gridSize stage p).z)

def jobs {α : Type} (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) : List (Invocation α) :=
  work.forwardState.map (invocation M D .forwardState c.forward c.scale c.seed c.final) ++
  work.forwardOutput.map (invocation M D .forwardOutput c.forward c.scale Bool.false Bool.false) ++
  work.backwardState.map (invocation M D .backwardState c.backward c.scale c.terminal c.seed) ++
  work.backwardGradient.map (invocation M D .backwardGradient c.backward c.scale Bool.false Bool.false)

/-- Four complete source grids followed by the existing finite FP32 library
sum. `work` chooses legal grid permutations; `selection` retains every sum tree. -/
def stages {α : Type} (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) : List (ReductionStage α) :=
  (jobs M D c work).map ReductionStage.invocation ++
    [.fp32Sum (LibraryReduction.sumStage M c.backward c.selection)]

def real (c : KernelInterface.Config) (work : KernelSchedule.Work) : ReductionProgramIO ℝ ℝ where
  toMemoryIO := IOLayout.realMemory ℝ c
  stages := fun _ => stages Real.algebra Real.dotPrimitive c.call work
  encode := id
  observe := fun _ _ => id

/-- Native FP32/FMA observation of the same quotations, including all written
intermediates. The runtime scale is the actual ABI word, bound after source
term construction; no real-number specialization replaces it. -/
def native (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config) (scale : BitVec 32)
    (work : KernelSchedule.Work) : ReductionProgramIO (BitVec 32) NativeTerms.Term where
  toMemoryIO := IOLayout.nativeMemory (BitVec 32) c
  stages := fun _ => stages NativeTerms.algebra NativeTerms.dotPrimitive
    (KernelExecution.symbolicCall c.call) work
  encode := NativeTerms.Term.bits
  observe := fun o _ term =>
    (NativeTerms.lowerAt c.call.backward (IOLayout.nativeOutputs c.dims)[o].buf
      (NativeBound.bindScale scale term)).eval profile.isa

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) : Invocation α :=
  { IO.invocation M D stage p scale seed final pid with
    kernel := match stage with
      | .forwardOutput => forwardOHoisted p scale
      | _ => stageKernel stage p scale seed final }

def jobs (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (work : KernelSchedule.Work) :=
  work.forwardState.map (invocation M D .forwardState c.forward c.scale c.seed c.final) ++
  work.forwardOutput.map (invocation M D .forwardOutput c.forward c.scale Bool.false Bool.false) ++
  work.backwardState.map (invocation M D .backwardState c.backward c.scale c.terminal c.seed) ++
  work.backwardGradient.map (invocation M D .backwardGradient c.backward c.scale Bool.false Bool.false)

def stages (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (work : KernelSchedule.Work) :=
  (jobs M D c work).map ReductionStage.invocation ++
    [ReductionStage.fp32Sum (LibraryReduction.sumStage M c.backward c.selection)]

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.LogicalTensors
open VeriTile Triton FP.Structural
noncomputable section

/- Every finite tensor, including inactive optional channels, has a true
coordinate inverse. The original typed memory/schedule/observer is reused. -/
structure Inputs (c : KernelInterface.Config) where
  query : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.K) → ℝ
  key : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.K) → ℝ
  value : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.V) → ℝ
  incoming : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.V) → ℝ
  initial : Fin (c.dims.B*c.dims.H) → Fin (c.dims.K) → Fin (c.dims.V) → ℝ
  terminal : Fin (c.dims.B*c.dims.H) → Fin (c.dims.K) → Fin (c.dims.V) → ℝ

structure Outputs (c : KernelInterface.Config) where
  output : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.V) → ℝ
  dQuery : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.K) → ℝ
  dKey : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.K) → ℝ
  dValue : Fin (c.dims.B*c.dims.H) → Fin (c.dims.T) → Fin (c.dims.V) → ℝ
  finalState : Fin (c.dims.B*c.dims.H) → Fin (c.dims.K) → Fin (c.dims.V) → ℝ
  dInitial : Fin (c.dims.B*c.dims.H) → Fin (c.dims.K) → Fin (c.dims.V) → ℝ

def inputs (c : KernelInterface.Config) : ChannelValues ℝ (IOLayout.inputs c.dims) ≃ Inputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (values (0 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (values (1 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (values (2 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (values (3 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (values (4 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (values (5 : Fin 6))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K]).symm values.query
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K]).symm values.key
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V]).symm values.value
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V]).symm values.incoming
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V]).symm values.initial
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V]).symm values.terminal
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputs (c : KernelInterface.Config) : ChannelValues ℝ (IOLayout.outputs c.dims) ≃ Outputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (values (0 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (values (1 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (values (2 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (values (3 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (values (4 : Fin 6)),
    Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (values (5 : Fin 6))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V]).symm values.output
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K]).symm values.dQuery
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.K]).symm values.dKey
    | ⟨3,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.T,c.dims.V]).symm values.dValue
    | ⟨4,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V]).symm values.finalState
    | ⟨5,_⟩ => (Values.curryEquiv ℝ .real [c.dims.B*c.dims.H,c.dims.K,c.dims.V]).symm values.dInitial
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def realInputs (c : KernelInterface.Config) (values : Inputs c) (head : Fin (c.dims.B*c.dims.H)) :=
  Mathematical.inputs c ((inputs c).symm values) head
def incoming (c : KernelInterface.Config) (values : Inputs c) (head : Fin (c.dims.B*c.dims.H)) :=
  Mathematical.incoming c ((inputs c).symm values) head
def terminal (c : KernelInterface.Config) (values : Inputs c) (head : Fin (c.dims.B*c.dims.H)) :=
  Mathematical.terminal c ((inputs c).symm values) head

def io (c : KernelInterface.Config) (work : KernelSchedule.Work) : LogicalIO (ChannelValues ℝ (IOLayout.inputs c.dims)) (ChannelValues ℝ (IOLayout.outputs c.dims)) (Inputs c) (Outputs c) (IO.real c work).Implements where
  inputs := inputs c
  outputs := outputs c

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.LogicalTensors

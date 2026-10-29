import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.NativeInputBinding
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.IOBounds
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.IOExecution
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.TensorView
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RealGrid
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.ByteGrid

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.PhysicalIOBinding
open VeriTile.Triton FP.Structural Execution Interface TensorView
noncomputable section
set_option maxHeartbeats 4000000

/-- Each logical channel is a typed view of the same original memory reader.
The simultaneous half/int views preserve the old default-read and byte domain. -/
def inputs {α : Type} [Inhabited α] (s : State α) : Inputs α
  | ⟨0,_⟩ => fun o => (s.mem "A" o).read .fp16
  | ⟨1,_⟩ => fun o => (s.mem "W1" o).read .fp16
  | ⟨2,_⟩ => fun o => (s.mem "W3" o).read .fp16
  | ⟨3,_⟩ => fun o => (s.mem "RMS" o).read .fp16
  | ⟨4,_⟩ => fun o => (s.mem "W1" o).read .int
  | ⟨5,_⟩ => fun o => (s.mem "W3" o).read .int
  | ⟨6,_⟩ => fun o => (s.mem "RMS" o).read .int

theorem input_reads {α : Type} [Inhabited α] (s : State α) : Reads (inputs s) s :=
  ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl,fun _ => rfl,
    fun _ => rfl,fun _ => rfl,fun _ => rfl⟩

theorem binding {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (s : State α) :
    (programIO native a g p).InputBinding (inputs s) s := by
  intro i o
  have hi : i.val<7 := i.isLt
  fin_cases i <;> rfl

theorem domain {α : Type} [Inhabited α] (s : State α) (bytes : ByteValues s) :
    byteDomain (inputs s) :=
  ⟨fun o => bytes "W1" (by simp) o,
   fun o => bytes "W3" (by simp) o,
   fun o => bytes "RMS" (by simp) o⟩

def nativeInputs (p : Parameters) (mem : Words.Memory) : Inputs Words.F := inputs (Words.entry p mem)
def realInputs (p : Parameters) (mem : Words.Memory) : Inputs ℝ := inputs (realEntry p mem)

theorem native_binding (a : Words.Instructions) (p : Parameters) (mem : Words.Memory) :
    (programIO Bool.true (Words.arithmetic a) (Words.gates a) p).InputBinding
      (nativeInputs p mem) (Words.entry p mem) := binding _ _ _ _ _

theorem native_domain (p : Parameters) (mem : Words.Memory) : byteDomain (nativeInputs p mem) :=
  domain _ (Words.entry_bytes p mem)

theorem real_binding (p : Parameters) (mem : Words.Memory) :
    (programIO Bool.false realArithmetic RealGrid.unusedGate p).InputBinding
      (realInputs p mem) (realEntry p mem) := binding _ _ _ _ _

theorem real_domain (p : Parameters) (mem : Words.Memory) : byteDomain (realInputs p mem) :=
  domain _ (realEntry_bytes p mem)

/-- Exact byte-to-input meaning, including the int8 sign interpretation and
all 65,536 half patterns. No finite-value assumption is introduced. -/
theorem native_input_half (p : Parameters) (mem : Words.Memory) (o : Nat) :
    nativeInputs p mem (0 : Fin 7) o = Words.embed (Words.read16 mem "A" o) := by
  simp [nativeInputs,inputs,Words.entry,Words.isByteRegion]

theorem native_weight_byte (p : Parameters) (mem : Words.Memory) (h : p.byte=Bool.true) (o : Nat) :
    nativeInputs p mem (4 : Fin 7) o = (mem "W1" o).toInt ∧
    nativeInputs p mem (5 : Fin 7) o = (mem "W3" o).toInt ∧
    nativeInputs p mem (6 : Fin 7) o = (mem "RMS" o).toInt := by
  simp [nativeInputs,inputs,Words.entry,Words.isByteRegion,h]

theorem native_target (a : Words.Instructions) (p : Parameters) (mem : Words.Memory) (x : Grid.Index p) :
    Reference.nativeFormula a p (nativeInputs p mem) (0 : Fin 1) (x.1,x.2,()) =
      ByteGrid.target a p mem x := by
  exact congrArg Words.half ((input_reads (Words.entry p mem)).native_target
    (Words.arithmetic a) (Words.gates a) p x.1.val x.2.val).symm

theorem real_target (p : Parameters) (mem : Words.Memory) (x : Grid.Index p) :
    Reference.realFormula p (realInputs p mem) (0 : Fin 1) (x.1,x.2,()) =
      Real.target p (realEntry p mem) x.1.val x.2.val :=
  ((input_reads (realEntry p mem)).real_target p x.1.val x.2.val).symm

/-- The preserved byte commitment theorem and the public IO relation execute
exactly the same native schedule, including the no-launch identity. -/
theorem native_execution (a : Words.Instructions) (p : Parameters) (mem : Words.Memory) :
    runSchedule ((programIO Bool.true (Words.arithmetic a) (Words.gates a) p).schedule
      (nativeInputs p mem)) (Words.entry p mem) = ByteGrid.run a p mem :=
  program_run _ _ _ _ _ _

theorem real_execution (p : Parameters) (mem : Words.Memory) :
    runSchedule ((programIO Bool.false realArithmetic RealGrid.unusedGate p).schedule
      (realInputs p mem)) (realEntry p mem) = RealGrid.run p (realEntry p mem) :=
  program_run _ _ _ _ _ _

#axiomsClean native_binding
#axiomsClean native_domain
#axiomsClean real_binding
#axiomsClean real_domain
#axiomsClean native_target
#axiomsClean real_target
#axiomsClean native_execution
#axiomsClean real_execution
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.PhysicalIOBinding

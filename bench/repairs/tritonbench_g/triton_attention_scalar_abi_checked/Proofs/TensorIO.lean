import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.TensorReturns
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.IOExecution
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- The actual Tensor storage reads at the five declared finite input ports. -/
def ioInputs {α : Type} (zero : α) (c : Shape) (t : Objects c) (read : Nat→α) : PublicIO.Inputs α c :=
  fun port=>match port with
  | ⟨0,_⟩=>fun i=>inputAt c t read "Q" (PublicIO.position c i)
  | ⟨1,_⟩=>fun i=>inputAt c t read "K" (PublicIO.position c i)
  | ⟨2,_⟩=>fun i=>inputAt c t read "V" (PublicIO.position c i)
  | ⟨3,_⟩=>fun i=>inputAt c t read "DO" (PublicIO.position c i)
  | ⟨4,_⟩=>fun _=>zero

theorem ioInputs_zero {α : Type} (zero : α) (c : Shape) (t : Objects c) (read : Nat→α) :
    PublicIO.ZeroDQ zero c (ioInputs zero c t read) := fun _=>rfl

theorem real_io_inputs (c : Shape) (t : Objects c) (read : Nat→ℝ) (scale : ℝ)
    (fw pw bw : List Grid.Program) :
    (PublicIO.realIO c scale fw pw bw).InputCells (ioInputs 0 c t read) (initialReal c t read) := by
  intro port
  fin_cases port <;> intro i _ <;>
    simp [PublicIO.realIO,PublicIO.layout,initialReal,ioInputs,PublicIO.inputs,PublicIO.inputOffset,mapValue]
  all_goals rfl

theorem native_io_inputs (c : Shape) (t : Objects c) (read : Nat→Native.Word)
    (scalar : Checked.ScalarABI) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Native.Prior.Instructions) (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program) :
    (PublicIO.nativeIO scalar c owners I literal fw pw bw).InputCells (ioInputs 0 c t read)
      (Native.Selected.words (initialWords c t read)) := by
  intro port
  fin_cases port <;> intro i _ <;>
    simp [PublicIO.nativeIO,PublicIO.layout,initialWords,ioInputs,PublicIO.inputs,PublicIO.inputOffset,
      Native.Selected.words,Native.mapCell,Native.mapValue,mapValue]
  all_goals rfl

/-- The actual Real Tensor publication consumes the same complete shared IO
execution. Tensor identities and storage frames are still supplied by the
original ordinary Tensor laws. -/
theorem real_io_execution (c : Shape) (t : Objects c) (read : Nat→ℝ) (scale : ℝ)
    (fw pw bw : List Grid.Program) :
    invokeReal c t read scale fw pw bw=
      (runMaterializedSchedule ((PublicIO.realIO c scale fw pw bw).schedule (ioInputs 0 c t read))
        (initialReal c t read)).map (result c t (fun a=>some (read a))) := by
  rw [PublicIO.real_schedule]
  rfl

/-- Original native graph evaluation and actual Tensor publication occur after
the shared complete materialized source schedule, with no change of owners. -/
theorem native_io_execution (c : Shape) (t : Objects c) (read : Nat→Native.Word)
    (scalar : Checked.ScalarABI) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Native.Prior.Instructions) (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program) :
    invokeNative c t read scalar owners I literal fw pw bw=
      (runMaterializedSchedule
        ((PublicIO.nativeIO scalar c owners I literal fw pw bw).schedule (ioInputs 0 c t read))
        (Native.Selected.words (initialWords c t read))).map
        (fun final=>result c t read (Native.Selected.evaluate I literal final)) := by
  rw [PublicIO.native_schedule]
  simp only [invokeNative,Native.Selected.nativeRun,Option.map_map]
  rfl

#axiomsClean real_io_inputs
#axiomsClean native_io_inputs
#axiomsClean real_io_execution
#axiomsClean native_io_execution
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage

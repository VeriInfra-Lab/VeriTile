import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.NativeIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarrySource

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage NativeSchedule
noncomputable section
open scoped Classical

/-- The carry-first candidate uses the original physical grid and write ownership. -/
abbrev Program (B : Nat) (c : Config) := NativeIO.Program B c

def program (B : Nat) (c : Config) (i : Fin (Fintype.card (Program B c))) : Program B c :=
  NativeIO.program B c i

/-- Only the displayed kernel changes; scalar binding and program IDs are shared. -/
def invocation (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (p : Program B c) : Invocation (Term α) :=
  { NativeIO.invocation C arg B c p with kernel := CarrySource.kernel c }

/-- Full ordinary grid on the same caller input memory. -/
def schedule (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) : List (Invocation (Term α)) :=
  List.ofFn (fun i=>invocation C arg B c (program B c i))

def writes (B : Nat) (c : Config) (p : Program B c) (r : RegionName) (o : Nat) : Prop :=
  NativeIO.writes B c p r o

/-- The canonical native memory, encoding, observation and write sets are reused. -/
@[reducible] def io (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) : OrdinaryProgramIO α (Term α) :=
  { NativeIO.io A C arg B c with
    schedule := fun _=>schedule C arg B c
    writes := fun _ k=>writes B c (program B c
      ⟨k.val,by simpa only [schedule,List.length_ofFn] using k.isLt⟩) }

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryIO

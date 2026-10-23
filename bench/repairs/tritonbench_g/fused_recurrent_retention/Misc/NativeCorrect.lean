import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.NativeResult

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
open VeriTile.Triton FP.Structural
noncomputable section
open Numeric
local instance : Inhabited Word := ⟨FP.Value.ofBits .fp32 0⟩
/-- Selected FP32 source, exact FMA/reduction topology and runtime scale ABI.
All original symbolic inputs and all FP32 words remain admitted. This is a
source-to-native observation under the existing primitive scalar laws, with
no optimization rewrite or native-equals-Real assertion. -/
abbrev retentionNativeIO (ports : ScalarPorts) (mode : Quoted.Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    ProgramIO (Staged.Node Word) (Staged.Node Word) :=
  PublicData.io mode (nativeArithmetic ports) s p incoming finalGradient seed final

/-- State-free FP32 recurrence on logical input values. The fixed reduction
plans describe topology only; no expected output or reached carry is supplied. -/
def nativeExpected (u : UnaryModels) (s : Shape) (p : PlanBinding s) (seed final : Bool)
    (xs : ChannelValues (Staged.Node Word) (PublicData.inputPorts s)) :
    ChannelValues Word (PublicData.outputPorts s) :=
  fun i => match i with
  | ⟨0,_⟩ => fun j => partialValue (arithmetic u) s p.plans
      (PublicData.logicalInputs (arithmetic u) s xs) seed final .output j.1.val
  | ⟨1,_⟩ => fun j => partialValue (arithmetic u) s p.plans
      (PublicData.logicalInputs (arithmetic u) s xs) seed final .dq j.1.val
  | ⟨2,_⟩ => fun j => partialValue (arithmetic u) s p.plans
      (PublicData.logicalInputs (arithmetic u) s xs) seed final .dk j.1.val
  | ⟨3,_⟩ => fun j => partialValue (arithmetic u) s p.plans
      (PublicData.logicalInputs (arithmetic u) s xs) seed final .dv j.1.val
  | ⟨4,_⟩ => fun j => partialValue (arithmetic u) s p.plans
      (PublicData.logicalInputs (arithmetic u) s xs) seed final .finalState j.1.val

/-- Pure FP32 output payload observation, preserving the original input carrier. -/
def nativeObserver (ports : ScalarPorts) (mode : Quoted.Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (o : Fin (retentionNativeIO ports mode s p incoming finalGradient seed final).outputs.length) →
      TileIndex ((retentionNativeIO ports mode s p incoming finalGradient seed final).outputs[o]).shape →
        Staged.Node Word → Word := fun _ _ n => n.value

theorem kernel_forward_native (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (retentionNativeIO ports .forward s p incoming finalGradient seed final).Observes
      (nativeObserver ports .forward s p incoming finalGradient seed final) (nativeExpected u s p seed final) :=
  PublicData.native_complete .forward u ports laws s shape p incoming finalGradient seed final

#axiomsClean kernel_forward_native
#stmtSurfaceSubset kernel_forward_native ⊆
  [UnaryModels,ScalarPorts,ScalarLaws,Shape,Shape.Admitted,PlanBinding,retentionNativeIO,
   Quoted.Mode.forward,ProgramIO.Observes,nativeObserver,Staged.Node,Staged.instInhabitedNode,instInhabitedWord,Word,nativeExpected]

theorem kernel_backward_native (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (retentionNativeIO ports .backward s p incoming finalGradient seed final).Observes
      (nativeObserver ports .backward s p incoming finalGradient seed final) (nativeExpected u s p seed final) :=
  PublicData.native_complete .backward u ports laws s shape p incoming finalGradient seed final

#axiomsClean kernel_backward_native
#stmtSurfaceSubset kernel_backward_native ⊆
  [UnaryModels,ScalarPorts,ScalarLaws,Shape,Shape.Admitted,PlanBinding,retentionNativeIO,
   Quoted.Mode.backward,ProgramIO.Observes,nativeObserver,Staged.Node,Staged.instInhabitedNode,instInhabitedWord,Word,nativeExpected]

#specNonCircular nativeExpected avoiding [Quoted.gridExec,sourceCell,programO,programQ,programK,programV]
/- ScalarLaws is the retained primitive-model boundary, not a transformation
assumption. Its existing deferred numerical status is unchanged. -/
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention

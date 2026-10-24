import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeMemory
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridFrame
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 4000000

theorem storeStep_frame (kind : Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (pid : Grid.Program) (mem : Grid.Memory Word)
    (hp : Grid.Legal (gridFor c kind) pid) (r : RegionName) (o : Nat) (h : ¬Grid.Written c r o) :
    storeStep kind c scale owners I literal pid mem r o=mem r o := by
  have back:=congrFun (congrFun (project_inject kind c owners I literal mem) r) o
  cases kind
  · change mapCell _ (Grid.forwardStep (model .forward) c scale pid (inject mem) r o)=_
    rw [Grid.forwardStep_frame _ c scale pid _ hp r o h]
    exact back
  · change mapCell _ (Grid.preprocessStep (model .preprocess) c pid (inject mem) r o)=_
    rw [Grid.preprocessStep_frame _ c pid _ hp r o h]
    exact back
  · change mapCell _ (Grid.backwardStep (model .backward) c scale pid (inject mem) r o)=_
    rw [Grid.backwardStep_frame _ c scale pid _ hp r o h]
    exact back

theorem storeReference_frame (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (fw pw bw : List Grid.Program) (mem : Grid.Memory Word)
    (hf : ∀ pid∈fw,Grid.Legal (Grid.forwardGrid c) pid)
    (hp : ∀ pid∈pw,Grid.Legal (Grid.preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Grid.Legal (Grid.backwardGrid c) pid)
    (r : RegionName) (o : Nat) (h : ¬Grid.Written c r o) :
    storeReference c scale owners I literal fw pw bw mem r o=mem r o := by
  unfold storeReference
  rw [Grid.fold_frame _ bw _ r o (by intro pid hpid m;exact storeStep_frame .backward c scale owners I literal pid m (hb pid hpid) r o h)]
  rw [Grid.fold_frame _ pw _ r o (by intro pid hpid m;exact storeStep_frame .preprocess c scale owners I literal pid m (hp pid hpid) r o h)]
  exact Grid.fold_frame _ fw mem r o (by intro pid hpid m;exact storeStep_frame .forward c scale owners I literal pid m (hf pid hpid) r o h)

theorem whole_execution_frame (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions)
    (literal : ℝ → Word) (fw pw bw : List Grid.Program) (mem : Grid.Memory Word)
    (hf : ∀ pid∈fw,Grid.Legal (Grid.forwardGrid c) pid)
    (hp : ∀ pid∈pw,Grid.Legal (Grid.preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Grid.Legal (Grid.backwardGrid c) pid) :
    ∃ final,wholeRun c scale owners I literal fw pw bw mem=some final ∧
      ∀ r o,¬Grid.Written c r o → final r o=mem r o := by
  exact ⟨_,whole_success c scale owners I literal fw pw bw mem,
    storeReference_frame c scale owners I literal fw pw bw mem hf hp hb⟩

/-- The numerical boundary is only local native instructions. This transport
does not supply compiler conformance or equality to an independent reference. -/
theorem whole_local_transport (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (actual reference : Prior.Instructions)
    (literal : ℝ → Word) (fw pw bw : List Grid.Program) (mem : Grid.Memory Word)
    (h : Prior.LocalConformance actual reference) :
    wholeRun c scale owners actual literal fw pw bw mem=wholeRun c scale owners reference literal fw pw bw mem := by
  rw [Prior.local_conformance_eq actual reference h]

#axiomsClean whole_execution_frame
#axiomsClean whole_local_transport
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native

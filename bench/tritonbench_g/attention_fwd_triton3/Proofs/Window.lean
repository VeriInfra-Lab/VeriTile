import bench.tritonbench_g.attention_fwd_triton3.Proofs.WindowBranch
import VeriTile.Triton.Float.Integer
import VeriTile.Triton.Float.SignedCast

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem window (c : Call) (o : Ops α) (initialState s : State α)
    (live : Started c o (inputs initialState) s) (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) ((parts c).before.drop 21) s=some t ∧
      Matches c o (inputs initialState) (initial c o (inputs initialState)) t ∧
      t.mem=initialState.mem := by
  obtain ⟨captured,hcapture,ready,captureMemory⟩:=capture c o initialState s live memory
  obtain ⟨t,hwindow,live',memory'⟩:=window_branch c o initialState captured ready captureMemory
  refine ⟨t,?_,live',memory'⟩
  have split : (parts c).before.drop 21=((parts c).before.drop 21).take 12 ++
      (parts c).before.drop 33 := rfl
  rw [split,run_append,hcapture]
  exact hwindow

theorem prelude (c : Call) (o : Ops α) (s : State α) :
    ∃ t,run (engine o.algebra) (parts c).before s=some t ∧
      Matches c o (inputs s) (initial c o (inputs s)) t ∧ t.mem=s.mem := by
  obtain ⟨headerState,hheader,headerLive,headerMemory⟩ := header c o s
  obtain ⟨startedState,hstarted,startedLive,startedMemory⟩ :=
    initialization c o s headerState headerLive headerMemory
  obtain ⟨ready,hready,readyLive,readyMemory⟩ :=
    window c o s startedState startedLive startedMemory
  refine ⟨ready,?_,readyLive,readyMemory⟩
  have split : (parts c).before=(parts c).before.take 17 ++
      ((parts c).before.drop 17).take 4 ++ (parts c).before.drop 21 := rfl
  rw [split,run_append,run_append,hheader]
  simp only [Option.bind_some,hstarted]
  exact hready

#axiomsClean prelude
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference

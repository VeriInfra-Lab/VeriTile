import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEScore
import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEUpdate
import bench.tritonbench_g.attention_fwd_triton3.Proofs.BodyProduct

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (engine o.algebra) (parts c).body
        (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _ => cur))=some t ∧
      Matches c o (inputs initialState) (advance c o (inputs initialState) cur old) t ∧
      t.mem=initialState.mem := by
  obtain ⟨scored,hscore,scoredReady,scoreMemory⟩:=score_phase c o positive initialState s old live memory cur
  obtain ⟨updated,hupdate,updateReady,updateMemory⟩:=
    update_phase c o positive initialState scored old cur scoredReady scoreMemory
  obtain ⟨t,hproduct,live',memory'⟩:=
    Reference.product_phase c o positive initialState updated old cur updateReady updateMemory
  have product_code : (parts c).body.drop 19=(Reference.parts c).body.drop 19 := rfl
  rw [←product_code] at hproduct
  refine ⟨t,?_,live',memory'⟩
  have split : (parts c).body=(parts c).body.take 7 ++
      ((parts c).body.drop 7).take 12 ++ (parts c).body.drop 19 := rfl
  rw [split,run_append,run_append,hscore]
  simp only [Option.bind_some,hupdate]
  exact hproduct

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE

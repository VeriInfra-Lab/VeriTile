import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.PairIO
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.IOPair
open VeriTile Triton FP.Structural FusedRotaryEmbedding
noncomputable section

private theorem map_pair {ι : Type} (left right : ι → Invocation NativeSource.Payload)
    (same : ∀ i s, (left i).run s = (right i).run s)
    (safe : ∀ i s bounds, (left i).Safe bounds s ↔ (right i).Safe bounds s)
    (ids : List ι) (s : State NativeSource.Payload) :
    runSchedule (ids.map left) s = runSchedule (ids.map right) s ∧
    ∀ bounds, ScheduleSafe bounds (ids.map left) s ↔ ScheduleSafe bounds (ids.map right) s := by
  induction ids generalizing s with
  | nil => exact ⟨rfl,fun _ => Iff.rfl⟩
  | cons i ids ih =>
    constructor
    · simp only [List.map_cons,runSchedule,same]
      congr 1
      funext t
      exact (ih t).1
    · intro bounds
      simp only [List.map_cons,ScheduleSafe,safe,same]
      exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (ih t).2 bounds)

theorem schedules (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config)
    (s : State NativeSource.Payload) :
    runSchedule ((programs c.shape).map (Source.invocation rest (NativeSource.ops p) c)) s =
      runSchedule ((programs c.shape).map (job rest p c)) s ∧
    ∀ bounds, ScheduleSafe bounds ((programs c.shape).map (Source.invocation rest (NativeSource.ops p) c)) s ↔
      ScheduleSafe bounds ((programs c.shape).map (job rest p c)) s := by
  exact map_pair (Source.invocation rest (NativeSource.ops p) c) (job rest p c)
    (fun pid s => SourcePair.source_exec (Source.algebra rest (NativeSource.ops p)) (Source.parameters c.shape) _)
    (fun pid s bounds => SourcePair.source_safe (Source.algebra rest (NativeSource.ops p)) (Source.parameters c.shape) _ bounds)
    (programs c.shape) s

/-- The unchanged source-native IO certificate supplies success and the full
frame. The exact signed-index identity transports the entire Q/K/V grid. -/
theorem equivalent (rest : Algebra NativeSource.Payload) (p : Native.Instructions)
    (c : Config) (valid : c.Valid) :
    (io rest p c).EquivalentOn (variant rest p c) (PublicIO.metadataDomain c) := by
  have implemented := PublicIO.implements rest (NativeSource.ops p) (fun word _=>Native.atom word)
    (fun kind value=>(value kind).value) c valid
    (fun xs=>wordFormula c.shape p (PublicIO.inputMemory c.shape xs))
    (fun xs w hw high=>PublicIO.native_pair c p xs w hw high)
  intro values domain bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := implemented values domain bounds hb s hin
  have same := schedules rest p c s
  refine ⟨safe,(same.2 bounds).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact same.1.symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.IOPair

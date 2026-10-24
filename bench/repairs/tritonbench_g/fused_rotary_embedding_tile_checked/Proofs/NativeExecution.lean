import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Native

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.NativeExecution
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000

/-- Complete raw quoted grid, source-ordered native word formula, and every
untouched typed cell. Q/K retain their distinct mul/FMA orders and V copies
raw words. No Tensor admission, returned Tensor or output equation is assumed. -/
theorem fused_rotary_native (rest : Algebra NativeSource.Payload) (c : Config)
    (valid : c.Valid) (p : Native.Instructions) (initial : Memory Native.Word) :
    ∃ final,Source.grid rest (NativeSource.ops p) (Source.parameters c.shape) (programs c.shape)
        (Source.lift c (NativeSource.initial initial))=some final ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        ((final (Source.region kind.output) (logicalOutput c kind t h d)).read .real kind).value=
          wordFormula c.shape p initial kind t h d) ∧
      (∀ r a,¬Footprint c (Source.decoded r) a →
        final r a=Source.lift c (NativeSource.initial initial) r a) ∧
      (c.shape.tokens=0 → final=Source.lift c (NativeSource.initial initial)) := by
  let result := actualSource c (fun _=>NativeSource.ops p) (NativeSource.initial initial)
  have executed := Source.complete rest (NativeSource.ops p) c valid (NativeSource.initial initial)
  refine ⟨Source.lift c result,executed,?_,?_,?_⟩
  · intro kind t h d ht hh hd
    rw [Source.lift_output]
    have selected := congrFun (congrFun (NativeSource.source_selected p c valid initial)
      kind.output) (logicalOutput c kind t h d)
    change (result kind.output (logicalOutput c kind t h d) kind).value = _
    have selected' : (result kind.output (logicalOutput c kind t h d) kind).value =
        Native.selected c p initial kind.output (logicalOutput c kind t h d) := by
      simpa only [NativeSource.project,NativeSource.coordinate_output] using selected
    rw [selected']
    let w := columnWork c.shape kind t h d
    let o := secondHalf c.shape d
    have hw := column_work c.shape valid.shape kind t h d ht hh hd
    have value := Native.selected_readback c valid p initial w hw.1 o
    rw [output_logical,pairWord_formula c.shape w hw.1 p initial o,hw.2] at value
    simpa only [w,columnWork] using value
  · intro r a outside
    have frame : result (Source.decoded r) a = NativeSource.initial initial (Source.decoded r) a := by
      unfold result
      rw [actualSource_eq_run c valid]
      exact run_frame c (fun _=>NativeSource.ops p) (NativeSource.initial initial) _ _
        (fun w hw written => outside (writes_footprint c valid w hw _ _ written))
    simp only [Source.lift]
    split_ifs <;> try rfl
    rw [frame]
  · intro empty
    have noPrograms : programs c.shape=[] := by simp [programs,empty]
    rw [noPrograms,Source.grid] at executed
    exact Option.some.inj executed.symm

#axiomsClean fused_rotary_native
#specNonCircular wordFormula avoiding [Source.grid,Source.kernel,actualSource,Native.selected]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.NativeExecution

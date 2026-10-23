import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.IOLayout

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem binding_pair [Inhabited α] [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config) (hc : c.Valid)
    (xs : ChannelValues α (inputChannels c.shape)) (s : State β)
    (binding : (io rest ops encode observe c).InputBinding xs s) (w : Work) (hw : c.shape.Active w) :
    pairValues c.shape (fun _=>ops) w (Source.rawValues s.mem) =
      pairValues c.shape (fun _=>ops) w (fun r a=>encode (inputMemory c.shape xs r a)) := by
  have data0 := binding_data rest ops encode observe c xs s binding w.kind _ (input_bound c.shape hc.shape w hw Bool.false)
  have data1 := binding_data rest ops encode observe c xs s binding w.kind _ (input_bound c.shape hc.shape w hw Bool.true)
  have co := binding_coefficient rest ops encode observe c xs s binding Bool.false _ (coefficient_bound c.shape hc.shape w hw Bool.false)
  have si := binding_coefficient rest ops encode observe c xs s binding Bool.true _ (coefficient_bound c.shape hc.shape w hw Bool.true)
  simp only [Bool.false_eq_true,ite_false,ite_true] at co si
  cases kw : w.kind <;> simp only [pairValues,kw,Kind.input,ite_true,ite_false,Kind.noConfusion] at *
  · rw [data0,data1,co,si]
  · rw [data0,data1,co,si]
  · rw [data0,data1]

/-- Compose full source execution with the independent pair-level numerical
interpretation already proved for this source's selected primitive model. -/
theorem correct [Inhabited α] [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config) (hc : c.Valid)
    (f : ChannelValues α (inputChannels c.shape) → Kind → Nat → Nat → Nat → α)
    (lower : ∀ xs w, c.shape.Active w → ∀ high,
      observe w.kind (component (pairValues c.shape (fun _=>ops) w
        (fun r a=>encode (inputMemory c.shape xs r a))) high) =
          f xs w.kind w.token w.head (c.shape.dim w high))
    (xs : ChannelValues α (inputChannels c.shape)) (domain : metadataDomain c xs) (s : State β)
    (binding : (io rest ops encode observe c).InputBinding xs s) :
    ∃ final, runSchedule ((io rest ops encode observe c).schedule xs) s = some final ∧
      (io rest ops encode observe c).OutputBinding xs (outputFormula c.shape (f xs)) final ∧
      (io rest ops encode observe c).Frame xs s final := by
  have metadataOk := binding_metadata rest ops encode observe c hc xs domain s binding
  obtain ⟨final,executed,outputs,frame⟩ := Source.complete_raw_output rest ops c hc s metadataOk
  have values (kind : Kind) (t h d : Nat) (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind) (hd : d<c.shape.width) :
      mapCell (observe kind) (final.mem (Source.region kind.output) (logicalOutput c kind t h d)) =
        Cell.mk .real (f xs kind t h d) := by
    rw [outputs kind t h d ht hh hd]
    change Cell.mk .real (observe kind (actualSource c (fun _=>ops) (Source.rawValues s.mem)
      kind.output (logicalOutput c kind t h d))) = Cell.mk .real (f xs kind t h d)
    apply congrArg (Cell.mk .real)
    let w := columnWork c.shape kind t h d
    let high := secondHalf c.shape d
    have hw := column_work c.shape hc.shape kind t h d ht hh hd
    have readback := run_readback c hc (fun _=>ops) (Source.rawValues s.mem) w hw.1 high
    rw [output_logical,hw.2] at readback
    change FusedRotaryEmbedding.run c (fun _=>ops) (Source.rawValues s.mem) kind.output
      (logicalOutput c kind t h d) = _ at readback
    rw [actualSource_eq_run c hc,readback,binding_pair rest ops encode observe c hc xs s binding w hw.1]
    have low := lower xs w hw.1 high
    have dimEq : c.shape.dim w high=d := hw.2
    rw [dimEq] at low
    exact low
  refine ⟨final,executed,?_,?_⟩
  · intro o j _
    fin_cases o
    · exact values .q j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt
    · exact values .k j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt
    · exact values .v j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt
  · intro r a outside
    exact frame r a (outside_footprint rest ops encode observe c xs r a outside)

#axiomsClean binding_pair
#axiomsClean correct
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO

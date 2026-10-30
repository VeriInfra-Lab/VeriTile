import bench.tritonbench_g.int8_quantization.Proofs.SourceStages

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem registers_mem (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).mem=entry.mem := by
  cases query : p.query <;> simp [registers,reduced,classified,prescaled,addressed,query]

@[simp] theorem registers_pids (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).pids=entry.pids := by
  cases query : p.query <;> simp [registers,reduced,classified,prescaled,addressed,query]

@[simp] theorem registers_block (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .nat [] "block"=some (fun _=> Model.group p entry) := by
  cases query : p.query <;> simp [registers,reduced,classified,prescaled,addressed,query]

@[simp] theorem registers_offset (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .nat [p.width] "offset"=some (Model.offset p entry) := by
  cases query : p.query <;> simp [registers,reduced,classified,prescaled,addressed,query]

@[simp] theorem registers_active (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .bool [p.width] "active"=
      some (fun i=> decide (Model.active p entry i)) := by
  cases query : p.query <;> simp [registers,reduced,classified,prescaled,addressed,query]

@[simp] theorem registers_quantized (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .int [p.width] "quantized"=
      some (quantized M ops p positive entry) := by simp [registers]

@[simp] theorem registers_bad (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .int [] "bad"=
      some (fun _=> badValue M ops p positive entry) := by simp [registers,reduced,classified]

@[simp] theorem registers_scale (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (registers M ops p positive entry).regs .real [] "scale"=
      some (fun _=> scale M ops p entry) := by simp [registers,reduced]

/-- Exactly the final two stores, including the integer byte channel and
the exceptional block-scale selection. Registers remain those of the source. -/
def finalState (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : State α :=
  ((TileShape.allIndices [p.width]).foldl (fun t i=> if Model.active p entry i then
    t.write outRegion (Model.offset p entry i) (.mk .int (quantized M ops p positive entry i))
    else t) (registers M ops p positive entry)).write scaleRegion (Model.group p entry)
      (.mk .real (savedScale M ops p positive entry))

theorem stores_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 21 2) (registers M ops p positive entry)=
      some (finalState M ops p positive entry) := by
  simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Model.backend,Option.bind,numericEq,bop,numeric,FP.Structural.store,
    Region.cast,ComputeDType.eraseDType,masked_scatter_regs,finalState,savedScale]

theorem segments_cover (p : Parameters) :
    (source p).surfaceBody=segment p 0 8++segment p 8 1++segment p 9 3++
      segment p 12 4++segment p 16 5++segment p 21 2 := rfl

/-- Constructed successful execution of the whole typed production source,
with the actual final registers and both observable stores. -/
theorem source_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    FP.Structural.exec (Model.backend M ops) (source p) entry=
      some (finalState M ops p positive entry) := by
  change run (Model.backend M ops) (source p).surfaceBody entry=_
  rw [segments_cover]
  simp only [run_append,addressed_run,prescaled_run,classified_run M ops p positive,reduced_run,registers_run,stores_run,
    Option.bind_some]

theorem final_model (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    finalState M ops p positive entry=
      Model.stored M p (registers M ops p positive entry) (registers M ops p positive entry) := by
  simp [finalState,Model.stored,Model.offset,Model.active,Model.group,
    Model.quantized,Model.scaleValue,savedScale]
  norm_num [scalar32]

theorem source_readback (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) (i : Lane p) (active : Model.active p entry i) :
    (finalState M ops p positive entry).mem outRegion (Model.offset p entry i)=
      .mk .int (quantized M ops p positive entry i) := by
  rw [final_model]
  simpa [Model.offset,Model.group,Model.quantized,Model.active] using
    Model.stored_quantized M p (registers M ops p positive entry) (registers M ops p positive entry) i
      (by simpa [Model.active,Model.group] using active)

theorem source_scale (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    (finalState M ops p positive entry).mem scaleRegion (Model.group p entry)=
      .mk .real (savedScale M ops p positive entry) := State.write_same _ _ _ _

theorem source_frame (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) (r : RegionName) (o : Nat)
    (notScale : r≠scaleRegion ∨ o≠Model.group p entry)
    (notOutput : r≠outRegion ∨ ∀ i,Model.active p entry i → o≠Model.offset p entry i) :
    (finalState M ops p positive entry).mem r o=entry.mem r o := by
  rw [final_model]
  have h:=Model.stored_frame M p (registers M ops p positive entry) (registers M ops p positive entry) r o
    (by simpa [Model.group] using notScale)
    (by simpa [Model.active,Model.offset,Model.group] using notOutput)
  simpa using h

#axiomsClean source_run
#axiomsClean source_readback
#axiomsClean source_scale
#axiomsClean source_frame
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002

import bench.repairs.tritonbench_g.flash_attn.Proofs.Source
namespace VeriTile.Bench.Repairs.FlashAttn
noncomputable section
structure WholeReal (x : Layout) (m : Memory ℝ) (scale : ℝ) (ws : List Work) : Prop where
  output : ∀ b h row col,(Work.output b h row col).active x.shape →
    execute x Real.ops (some scale) ws (Real.lift m) .o (address x.o b h row col)=
      some (Spec.output x m scale b h row col)
  logsum : ∀ b h row,(Work.logsum b h row).active x.shape →
    execute x Real.ops (some scale) ws (Real.lift m) .l (lAddress x.shape b h row)=
      some (Spec.lse x m scale b h row)
  frame : ∀ r a,¬footprint x r a → execute x Real.ops (some scale) ws (Real.lift m) r a=some (m r a)

theorem whole_real (x : Layout) (hs : x.shape.Valid) (ho : Layouts.Admitted x.shape x.o)
    (m : Memory ℝ) (scale : ℝ) (ws : List Work) (complete : Complete x.shape ws) :
    WholeReal x m scale ws := by
  exact ⟨Real.whole_output x hs (Layouts.output_injective x ho) m scale ws complete,
    Real.whole_logsum x hs (Layouts.output_injective x ho) m scale ws complete,
    call_frame x hs Real.ops (some scale) ws complete (Real.lift m)⟩

theorem whole_source_real (x : Layout) (hs : x.shape.Valid) (ho : Layouts.Admitted x.shape x.o)
    (m : Memory ℝ) (scale : ℝ) (b h row col : Nat) (active : (Work.output b h row col).active x.shape) :
    Source.launch x Real.ops (some scale) (Real.lift m) .o (address x.o b h row col)=
      some (Spec.output x m scale b h row col) := by
  rw [Source.launch_execute]
  exact (whole_real x hs ho m scale (sourceSchedule x.shape) (List.Perm.refl _)).output b h row col active
#axiomsClean whole_real
#axiomsClean whole_source_real
#specNonCircular Spec.output avoiding [execute,Source.launch,outputValue,blockStep]
#specNonCircular Spec.lse avoiding [execute,Source.launch,logValue,blockStep]
end
end VeriTile.Bench.Repairs.FlashAttn

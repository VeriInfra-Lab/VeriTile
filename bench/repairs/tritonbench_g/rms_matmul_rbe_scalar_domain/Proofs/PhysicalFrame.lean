import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def OutputWrite (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (e : Write α) : Prop :=
  ∃ c : Commit,c.Valid i ∧ e.fp8=qkvOutputFP8 (useFP8 i) c.phase ∧
    e.address=outputAddress i q c.id c.phase c.lane

theorem output_write_apart (i : Contract.Inputs) (checked : Contract.InputContract i)
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (e : Write α) (he : OutputWrite i q e) (t : RmsMatmulRbe.Contract.Tensor)
    (ht : t∈i.original.reads) (j : Nat) (hj : inBacking t j) :
    Apart e.address (outputWidth e.fp8) (address t j) (RmsMatmulRbe.Contract.width t.metadata.dtype) := by
  obtain ⟨c,hc,hf,ha⟩ := he
  rw [hf,ha,caller_output_width i checked q hq]
  exact backing_apart _ _ _ _ (output_inBacking i checked ordinary q hq c hc) hj
    (caller_output_read_apart i checked q hq c.phase t ht)

theorem run_read32_frame (access : Access α Store Byte) (events : List (Write α)) (s : Store)
    (j : Nat) (apart : ∀ e∈events,Apart e.address (outputWidth e.fp8) j 4) :
    access.read32 (run access events s) j=access.read32 s j := by
  induction events generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [run,List.foldl_cons,←run,ih]
    · exact access.read32_apart _ _ _ _ _ (apart e (by simp))
    · exact fun e he => apart e (List.mem_cons_of_mem _ he)

theorem run_read8_frame (access : Access α Store Byte) (events : List (Write α)) (s : Store)
    (j : Nat) (apart : ∀ e∈events,Apart e.address (outputWidth e.fp8) j 1) :
    access.read8 (run access events s) j=access.read8 s j := by
  induction events generalizing s with
  | nil => rfl
  | cons e rest ih =>
    rw [run,List.foldl_cons,←run,ih]
    · exact access.read8_apart _ _ _ _ _ (apart e (by simp))
    · exact fun e he => apart e (List.mem_cons_of_mem _ he)

theorem format_width (i : Contract.Inputs) (checked : Contract.InputContract i)
    (t : RmsMatmulRbe.Contract.Tensor) (format : t.metadata.dtype=i.original.qw.metadata.dtype) :
    RmsMatmulRbe.Contract.width t.metadata.dtype=outputWidth (useFP8 i) := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  have types := (f.others i.original.qw (by simp [RmsMatmulRbe.Contract.Inputs.others])).2.2.1
  simp only [RmsMatmulRbe.Contract.weightTypes,List.mem_cons,List.not_mem_nil,or_false] at types
  rcases types with h|h <;>
    simp [format,h,outputWidth,useFP8,RmsMatmulRbe.Contract.width,
      RmsMatmulRbe.Contract.fp32,RmsMatmulRbe.Contract.fp16,RmsMatmulRbe.Contract.int8]

theorem xCell_frame {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (events : List (Write α))
    (writes : ∀ e∈events,OutputWrite i q e) (s : Store) (id : Program) :
    (state access zero i q (run access events s) id).mem "X"=(state access zero i q s id).mem "X" := by
  have dtype := (RmsMatmulRbe.Checked.facts i.original checked.1).xAdmitted.2.2.1
  have width : RmsMatmulRbe.Contract.width i.original.x.metadata.dtype=4 := by
    simp only [List.mem_singleton] at dtype
    simp [dtype,RmsMatmulRbe.Contract.width]
  funext j
  simp only [state,ite_true]
  by_cases live : inBacking i.original.x j
  · rw [if_pos live,if_pos live,run_read32_frame]
    intro e he
    simpa only [width] using output_write_apart i checked ordinary q hq e (writes e he)
      i.original.x (by simp [RmsMatmulRbe.Contract.Inputs.reads]) j live
  · simp only [if_neg live]

theorem weightCell_frame (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (events : List (Write α))
    (writes : ∀ e∈events,OutputWrite i q e) (s : Store) (t : RmsMatmulRbe.Contract.Tensor)
    (ht : t∈i.original.reads) (format : t.metadata.dtype=i.original.qw.metadata.dtype) (j : Nat) :
    weightCell access zero (useFP8 i) t (run access events s) j=weightCell access zero (useFP8 i) t s j := by
  have width := format_width i checked t format
  have apart (live : inBacking t j) (e : Write α) (he : e∈events) :=
    output_write_apart i checked ordinary q hq e (writes e he) t ht j live
  by_cases live : inBacking t j
  · cases hf : useFP8 i
    · have equal := run_read_frame access events Bool.false s (address t j) (by
        intro e he
        simpa only [width,hf,outputWidth] using apart live e he)
      simpa only [weightCell,hf,Bool.false_eq_true,ite_false,if_pos live,Access.read] using
        congrArg (Cell.mk .fp16) equal
    · have equal := run_read8_frame access events s (address t j) (by
        intro e he
        simpa only [width,hf,outputWidth] using apart live e he)
      simp only [weightCell,hf,ite_true,sourceByte,if_pos live,equal]
  · simp [weightCell,sourceByte,live]

theorem state_input_frame {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (i : Contract.Inputs) (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (events : List (Write α))
    (writes : ∀ e∈events,OutputWrite i q e) (s : Store) (id : Program) (phase : Fin 3) :
    InputMemory (qkvArgument (arguments i) phase)
      (state access zero i q (run access events s) id) (state access zero i q s id) := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  have weights (t : RmsMatmulRbe.Contract.Tensor) (ht : t∈i.original.reads)
      (format : t.metadata.dtype=i.original.qw.metadata.dtype) :=
    weightCell_frame access zero i checked ordinary q hq events writes s t ht format
  refine ⟨xCell_frame access zero i checked ordinary q hq events writes s id,?_,?_⟩
  · funext j
    fin_cases phase <;> simp only [qkvArgument,arguments,weightRegion,state]
    · exact weights i.original.qw (by simp [RmsMatmulRbe.Contract.Inputs.reads]) rfl j
    · exact weights i.original.kw (by simp [RmsMatmulRbe.Contract.Inputs.reads])
        (f.format i.original.kw (by simp [RmsMatmulRbe.Contract.Inputs.sameFormat])) j
    · exact weights i.original.vw (by simp [RmsMatmulRbe.Contract.Inputs.reads])
        (f.format i.original.vw (by simp [RmsMatmulRbe.Contract.Inputs.sameFormat])) j
  · funext j
    simpa only [qkvArgument,arguments,state,ite_true,ite_false] using
      weights i.original.rms (by simp [RmsMatmulRbe.Contract.Inputs.reads])
        (f.format i.original.rms (by simp [RmsMatmulRbe.Contract.Inputs.sameFormat])) j

#axiomsClean output_write_apart
#axiomsClean xCell_frame
#axiomsClean weightCell_frame
#axiomsClean state_input_frame
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel

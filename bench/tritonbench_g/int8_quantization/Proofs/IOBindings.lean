import bench.tritonbench_g.int8_quantization.IO
import bench.tritonbench_g.int8_quantization.Proofs.RawKernel
import bench.tritonbench_g.int8_quantization.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem invocation_initial (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (w : Schedule.Program) (s : State α) : (invocation M ops p w).initial s = Grid.atProgram s w := by
  simp [invocation,Invocation.initial,bindRegisters,Grid.atProgram]

theorem schedule_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (order : List Schedule.Program) (s : State α) :
    runSchedule (order.map (invocation M ops p)) s = Grid.execute M ops p order s := by
  induction order generalizing s with
  | nil => rfl
  | cons w rest ih => simp only [List.map_cons,runSchedule,Invocation.run,invocation_initial,Grid.execute,invocation,ih,Invocation.initial,bindRegisters,List.foldl_nil,Grid.atProgram,Option.getD_none,beq_iff_eq,Bool.false_eq_true,if_false,Bind.bind]

theorem input_read (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (values : Inputs α p) (s : State α) (h : (quantizationIO M ops p).InputBinding values s)
    (group : Fin (count p)) (lane : Fin p.width) (active : active p group.val lane.val) :
    (s.mem xRegion (offset p group.val lane.val)).read .real = raw p values group lane :=
  h ⟨0,by simp [quantizationIO,memory,inputChannels]⟩ (group,lane,PUnit.unit) active

theorem loaded_eq (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (values : Inputs α p) (s : State α) (h : (quantizationIO M ops p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) :
    FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram s w) = padded M p values ⟨w.index,hw⟩ := by
  funext lane
  unfold FPTrace.inputs loaded padded
  change M.fp32Load (if active p w.index lane.val then _ else _) = _
  by_cases ha : active p w.index lane.val
  · simp only [ha,if_true]
    exact congrArg M.fp32Load (input_read M ops p values s h ⟨w.index,hw⟩ lane ha)
  · simp [ha]

theorem input_preserved (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0 < p.width) (order : List Schedule.Program) (values : Inputs α p)
    (s : State α) (h : (quantizationIO M ops p).InputBinding values s) :
    (quantizationIO M ops p).InputBinding values (Grid.model M ops p positive order s) := by
  intro k i hi
  fin_cases k
  change ((Grid.model M ops p positive order s).mem xRegion _).read .real = _
  rw [Grid.input_preserved M ops p positive order s]
  exact h _ i hi

theorem invocation_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0 < p.width) (values : Inputs α p) (bounds : RegionName → Nat)
    (hb : (quantizationIO M ops p).InBounds bounds values) (w : Schedule.Program)
    (hw : w.index < count p) (s : State α) : (invocation M ops p w).Safe bounds s := by
  unfold Invocation.Safe
  rw [invocation_initial]
  apply SourceSafety.source_safe M ops (Grid.parameters p w) positive (Grid.atProgram s w)
  · intro i hi
    exact hb.2.1 ⟨0,by simp [quantizationIO,memory,inputChannels]⟩ (⟨w.index,hw⟩,i.1,PUnit.unit) hi
  · intro i hi
    exact hb.2.2.1 ⟨0,by simp [quantizationIO,memory,outputChannels]⟩ (⟨w.index,hw⟩,i.1,PUnit.unit) hi
  · exact hb.2.2.1 ⟨1,by simp [quantizationIO,memory,outputChannels]⟩ (⟨w.index,hw⟩,PUnit.unit) trivial

theorem schedule_safe (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0 < p.width) (values : Inputs α p) (bounds : RegionName → Nat)
    (hb : (quantizationIO M ops p).InBounds bounds values) (order : List Schedule.Program)
    (members : ∀ w ∈ order, w.index < count p) (s : State α) :
    ScheduleSafe bounds (order.map (invocation M ops p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons w rest ih =>
    constructor
    · exact invocation_safe M ops p positive values bounds hb w (members w (by simp)) s
    · intro t _
      exact ih (fun w hw => members w (by simp [hw])) t

theorem frame (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (values : Inputs α p) (s final : State α) (h : RawKernel.Frame p s final) :
    (quantizationIO M ops p).Frame values s final := by
  intro r o outside
  apply h
  · by_cases hr : r = scaleRegion
    · right
      intro w hw
      have bound := (Schedule.program_bounds 0 (count p) w hw).2.2
      have hx := outside.1 ⟨1,by simp [quantizationIO,memory,outputChannels]⟩
        (⟨w.index,by simpa using bound⟩,PUnit.unit) trivial
      simpa [quantizationIO,memory,outputChannels,hr] using hx
    · exact Or.inl hr
  · by_cases hr : r = outRegion
    · right
      intro w hw i hi
      have bound := (Schedule.program_bounds 0 (count p) w hw).2.2
      have hx := outside.1 ⟨0,by simp [quantizationIO,memory,outputChannels]⟩
        (⟨w.index,by simpa using bound⟩,i.1,PUnit.unit) hi
      simpa [quantizationIO,memory,outputChannels,offset,Grid.offset,hr] using hx
    · exact Or.inl hr

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

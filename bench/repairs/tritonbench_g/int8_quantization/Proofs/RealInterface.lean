import bench.repairs.tritonbench_g.int8_quantization.Proofs.IOBindings
import bench.repairs.tritonbench_g.int8_quantization.Proofs.RealRaw

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem raw_eq (rest : Algebra ℝ) (p : Parameters) (values : Inputs ℝ p)
    (s : State ℝ) (h : (quantizationIO (Math.algebra rest) Math.primitives p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) (i : Lane p) (hi : Grid.active p w i) :
    Math.raw (Grid.parameters p w) (Grid.atProgram s w) i = raw p values ⟨w.index,hw⟩ i.1 :=
  input_read _ _ p values s h ⟨w.index,hw⟩ i.1 hi

theorem block_eq (rest : Algebra ℝ) (p : Parameters) (values : Inputs ℝ p)
    (s : State ℝ) (h : (quantizationIO (Math.algebra rest) Math.primitives p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) :
    Math.logicalBlock (Grid.parameters p w) (Grid.atProgram s w) = logicalBlock p values ⟨w.index,hw⟩ := by
  apply List.map_congr_left
  intro j hj
  have ha : Grid.active p w (j,PUnit.unit) := by
    simpa [Math.logicalIndices,Grid.parameters,Grid.atProgram,Model.active,Model.group] using (List.mem_filter.mp hj).2
  exact raw_eq rest p values s h w hw (j,PUnit.unit) ha

theorem finite_eq (rest : Algebra ℝ) (p : Parameters) (values : Inputs ℝ p)
    (s : State ℝ) (h : (quantizationIO (Math.algebra rest) Math.primitives p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) :
    Math.FiniteDomain (Grid.parameters p w) (Grid.atProgram s w) ↔ FiniteBlock p values ⟨w.index,hw⟩ := by
  constructor
  · intro finite lane active
    have hx := finite (lane,PUnit.unit) active
    have eq := raw_eq rest p values s h w hw (lane,PUnit.unit) active
    exact (congrArg (fun x : ℝ => |RealSpec.factor p.query p.channels*x| ≤
      340282346638528859811704183484516925440) eq).mp hx
  · intro finite i active
    rw [raw_eq rest p values s h w hw i active]
    exact finite i.1 active

theorem output_eq (rest : Algebra ℝ) (p : Parameters) (values : Inputs ℝ p)
    (s : State ℝ) (h : (quantizationIO (Math.algebra rest) Math.primitives p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) (i : Lane p) (hi : Grid.active p w i) :
    Specification.totalOutput (Grid.parameters p w) (Grid.atProgram s w) i =
      realTarget p values ⟨0,by simp [outputChannels]⟩ (⟨w.index,hw⟩,i.1,PUnit.unit) := by
  classical
  unfold Specification.totalOutput realTarget
  rw [finite_eq rest p values s h w hw]
  unfold Math.blockMaximum
  rw [block_eq rest p values s h w hw,raw_eq rest p values s h w hw i hi]
  rfl

theorem scale_eq (rest : Algebra ℝ) (p : Parameters) (values : Inputs ℝ p)
    (s : State ℝ) (h : (quantizationIO (Math.algebra rest) Math.primitives p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) :
    Specification.totalScale (Grid.parameters p w) (Grid.atProgram s w) =
      realTarget p values ⟨1,by simp [outputChannels]⟩ (⟨w.index,hw⟩,PUnit.unit) := by
  classical
  unfold Specification.totalScale realTarget
  rw [finite_eq rest p values s h w hw,block_eq rest p values s h w hw]
  rfl

/-- Full-grid mathematical IO, including arbitrary invalid blocks and empty
geometry, is assembled from the existing total source proof and actual safety. -/
theorem real_implements (p : Parameters) (positive : 0 < p.width)
    (block : 0 < p.blockRows) (factor : p.factor = RealSpec.factor p.query p.channels) (rest : Algebra ℝ) :
    (quantizationIO (Math.algebra rest) Math.primitives p).Implements (realTarget p) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    exact schedule_safe _ _ p positive values bounds hb _
      (fun w hw => by simpa using (Schedule.program_bounds 0 (count p) w hw).2.2) s
  · intro values s hin
    obtain ⟨final,ran,out,scales,exterior⟩ := Specification.per_block_int8_real_raw p positive block factor rest s
    refine ⟨final,?_,?_,frame _ _ p values s final exterior⟩
    · exact (schedule_run _ _ p _ s).trans ran
    · intro o i hi
      fin_cases o
      · obtain ⟨w,hw,index⟩ := Schedule.complete (count p) i.1.val i.1.isLt
        have wb : w.index < count p := by rw [index]; exact i.1.isLt
        have wa : Grid.active p w (i.2.1,PUnit.unit) := by simpa only [Grid.active,active,index] using hi
        have result := out w hw (i.2.1,PUnit.unit) wa
        rw [output_eq rest p values s hin w wb _ wa] at result
        rcases i with ⟨group,lane,⟨⟩⟩
        have result' : final.mem outRegion (offset p group.val lane.val) =
            .mk .int (realTarget p values ⟨0,by simp [outputChannels]⟩ (group,lane,PUnit.unit)) := by
          simpa only [Grid.offset,offset,index] using result
        change mapCell id (final.mem outRegion (offset p group.val lane.val)) = _
        rw [result']
        rfl
      · obtain ⟨w,hw,index⟩ := Schedule.complete (count p) i.1.val i.1.isLt
        have wb : w.index < count p := by rw [index]; exact i.1.isLt
        have result := scales w hw
        rw [scale_eq rest p values s hin w wb] at result
        rcases i with ⟨group,⟨⟩⟩
        have result' : final.mem scaleRegion group.val =
            .mk .real (realTarget p values ⟨1,by simp [outputChannels]⟩ (group,PUnit.unit)) := by
          simpa only [index] using result
        change mapCell id (final.mem scaleRegion group.val) = _
        rw [result']
        rfl

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

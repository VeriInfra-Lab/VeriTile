import bench.repairs.tritonbench_g.int8_quantization.Proofs.IOBindings
import bench.repairs.tritonbench_g.int8_quantization.Proofs.NativeRaw

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem context_eq (M : Algebra Word) (ops : Model.Primitives Word) (p : Parameters)
    (values : Inputs Word p) (s : State Word) (h : (quantizationIO M ops p).InputBinding values s)
    (w : Schedule.Program) (hw : w.index < count p) (ctx : Context M p values ⟨w.index,hw⟩) :
    FPTrace.Context M (Grid.parameters p w) (Grid.atProgram s w) := by
  constructor
  · exact ctx.1
  · intro i
    have load := congrFun (loaded_eq M ops p values s h w hw) i.1
    change loaded M (Grid.parameters p w) (Grid.atProgram s w) (i.1,PUnit.unit) = _ at load
    have pair : (i.1,PUnit.unit) = i := by cases i; rfl
    rw [pair] at load
    unfold transformed
    rw [load]
    exact ctx.2 i.1

/-- The previous conditional source/software relation is expressed over
logical operands. Only the original context and fourteen reached primitive
agreements remain in the domain; source safety is proved independently. -/
theorem software_implements (p : Parameters) (positive : 0 < p.width) (block : 0 < p.blockRows)
    (M : Algebra Word) (ops : Model.Primitives Word) (software : Floating.Numerics Word) :
    (quantizationIO M ops p).ImplementsOn (Agreements M ops software p) (softwareTarget M software p) := by
  intro values agree bounds hb s hin
  have context (w : Schedule.Program) (hw : w ∈ Schedule.programs 0 (count p)) :
      FPTrace.Context M (Grid.parameters p w) (Grid.atProgram s w) :=
    context_eq M ops p values s hin w (by simpa using (Schedule.program_bounds 0 (count p) w hw).2.2)
      (agree _).1
  have agreements (w : Schedule.Program) (hw : w ∈ Schedule.programs 0 (count p)) :
      Floating.Local (FPTrace.constants M (Grid.parameters p w)) (FPTrace.numerics M ops) software p.query
        (FPTrace.inputs M (Grid.parameters p w) (Grid.atProgram s w)) := by
    rw [loaded_eq M ops p values s hin w (by simpa using (Schedule.program_bounds 0 (count p) w hw).2.2)]
    exact (agree _).2
  obtain ⟨final,ran,out,scales,exterior⟩ := FloatingSpecification.per_block_int8_equiv_raw
    p positive block M ops s software context agreements
  refine ⟨schedule_safe M ops p positive values bounds hb _
    (fun w hw => by simpa using (Schedule.program_bounds 0 (count p) w hw).2.2) s,
    final,(schedule_run M ops p _ s).trans ran,?_,frame M ops p values s final exterior⟩
  intro o i hi
  fin_cases o
  · obtain ⟨w,hw,index⟩ := Schedule.complete (count p) i.1.val i.1.isLt
    have wb : w.index < count p := by rw [index]; exact i.1.isLt
    have wa : Grid.active p w (i.2.1,PUnit.unit) := by simpa only [Grid.active,active,index] using hi
    have result := out w hw (i.2.1,PUnit.unit) wa
    rw [loaded_eq M ops p values s hin w wb] at result
    rcases i with ⟨group,lane,⟨⟩⟩
    have result' : final.mem outRegion (offset p group.val lane.val) =
        .mk .int (softwareTarget M software p values ⟨0,by simp [outputChannels]⟩ (group,lane,PUnit.unit)) := by
      simpa only [Grid.offset,offset,index,softwareTarget] using result
    change mapCell id (final.mem outRegion (offset p group.val lane.val)) = _
    rw [result']
    rfl
  · obtain ⟨w,hw,index⟩ := Schedule.complete (count p) i.1.val i.1.isLt
    have wb : w.index < count p := by rw [index]; exact i.1.isLt
    have result := scales w hw
    rw [loaded_eq M ops p values s hin w wb] at result
    rcases i with ⟨group,⟨⟩⟩
    have result' : final.mem scaleRegion group.val =
        .mk .real (softwareTarget M software p values ⟨1,by simp [outputChannels]⟩ (group,PUnit.unit)) := by
      simpa only [index,softwareTarget] using result
    change mapCell id (final.mem scaleRegion group.val) = _
    rw [result']
    rfl

/-- Native instructions use the original exact context choices and the same
ordered reduction plan, so their local agreements are proved by reflexivity. -/
theorem native_implements (p : Parameters) (positive : 0 < p.width) (block : 0 < p.blockRows)
    (rest : Algebra Word) (op : NativeGraph.Instructions) (plans : NativeGraph.Plans) :
    (quantizationIO (NativeGraph.algebra rest op plans) (NativeGraph.comparisons op) p).Implements
      (nativeTarget op plans p) := by
  intro values _ bounds hb s hin
  have laws : Agreements (NativeGraph.algebra rest op plans) (NativeGraph.comparisons op)
      (NativeGraph.numerics op plans) p values := by
    intro group
    refine ⟨⟨rfl,fun _ => rfl⟩,?_⟩
    exact NativeGraph.local_agreement rest op plans p p.query _
  have result := software_implements p positive block (NativeGraph.algebra rest op plans)
    (NativeGraph.comparisons op) (NativeGraph.numerics op plans) values laws bounds hb s hin
  exact result

#axiomsClean software_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.TypedGrid
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.KernelIndex
import bench.repairs.tritonbench_g.rotary_transform_ops.Spec

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelEndpoints
open VeriTile.Triton FP.Structural
noncomputable section

/-- The batch/row coordinate statements cover every packed or fixed-layout
token, even when cumulative lengths contain empty intervals. -/
theorem every_token (c : Config) (valid : c.Valid) (token : Nat) (inside : token<c.shape.total) :
    ∃ batch row,batch<c.shape.batches ∧ row<c.length batch ∧ c.start batch+row=token := by
  have bound : token<c.start c.shape.batches := by rw [start_total c valid];exact inside
  obtain ⟨batch,hb,lo,hi⟩ := intervals_cover c valid c.shape.batches (by omega) token bound
  refine ⟨batch,token-c.start batch,hb,?_,?_⟩ <;> omega

def TailReady {α : Type} [Inhabited α] (c : Config) (initial : State α) : Prop :=
  ∀ token head col,token<c.shape.total → head<c.shape.heads → col<c.shape.width → 2*c.shape.half≤col →
    (initial.mem (KernelAlgebra.regionName c.shape.output) (c.shape.address token head col)).read .real=
      (initial.mem "x" (c.shape.address token head col)).read .real

theorem tail_outside_typed (c : Config) (valid : c.Valid) (token head col : Nat)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) (tail : 2*c.shape.half≤col) :
    TypedGrid.Outside c (KernelAlgebra.regionName c.shape.output) (c.shape.address token head col) := by
  intro w hw written
  exact tail_outside c valid token head col hh hd tail w hw ⟨rfl,written.2⟩

/-- Complete mathematical rotation at every logical output coordinate. The
suffix condition is the ordinary kernel-entry invariant of the existing
out-of-place copy; for inplace it is reflexive. -/
theorem real (c : Config) (valid : c.Valid) (initial : State ℝ)
    (md : ProgramModel.MetadataBinding c initial) (suffix : TailReady c initial)
    (cumulative64 offsets64 : Bool) :
    ∃ final,TypedGrid.run realOps c cumulative64 offsets64 initial=some final ∧
      (∀ batch row head col,batch<c.shape.batches → row<c.length batch →
        head<c.shape.heads → col<c.shape.width →
        (final.mem (KernelAlgebra.regionName c.shape.output)
          (c.shape.address (c.start batch+row) head col)).read .real=
            realFormula c (KernelAlgebra.memory initial) batch row head col) ∧
      (∀ r o,TypedGrid.Outside c r o → final.mem r o=initial.mem r o) := by
  obtain ⟨final,success,_,values,frame⟩ :=
    TypedGrid.complete c valid realOps cumulative64 offsets64 initial md
  refine ⟨final,success,?_,frame⟩
  intro batch row head col hb hr hh hd
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have live : c.Active w := ⟨hb,hr,hh,hp⟩
    have got := (values w live (c.shape.second col)).trans
      (real_pair_formula c (KernelAlgebra.memory initial) w live (c.shape.second col))
    simpa only [w,Config.cell,back] using got
  · rw [frame _ _ (tail_outside_typed c valid _ head col hh hd (by omega)),
      suffix _ head col (row_bound c valid batch row hb hr) hh hd (by omega)]
    simp only [realFormula,if_neg rotary,KernelAlgebra.memory,KernelAlgebra.regionName]

instance : Inhabited Native.Node := ⟨Native.atom Native.zero⟩

/-- Exact raw-word graph for the compiler-selected default arithmetic. The
four PTX primitive callbacks and this fixed compiler pattern are explicit
model boundaries; no tensor-output equality or numerical experiment is a
premise. All raw FP32 payloads, including exceptional words, are retained. -/
theorem native (c : Config) (valid : c.Valid) (p : Native.Instructions) (initial : State Native.Node)
    (md : ProgramModel.MetadataBinding c initial) (suffix : TailReady c initial)
    (cumulative64 offsets64 : Bool) :
    ∃ final,TypedGrid.run (Native.ops p) c cumulative64 offsets64 initial=some final ∧
      (∀ batch row head col,batch<c.shape.batches → row<c.length batch →
        head<c.shape.heads → col<c.shape.width →
        ((final.mem (KernelAlgebra.regionName c.shape.output)
          (c.shape.address (c.start batch+row) head col)).read .real).value=
            nativeFormula c p (Native.project (KernelAlgebra.memory initial)) batch row head col) ∧
      (∀ r o,TypedGrid.Outside c r o → final.mem r o=initial.mem r o) := by
  obtain ⟨final,success,_,values,frame⟩ :=
    TypedGrid.complete c valid (Native.ops p) cumulative64 offsets64 initial md
  refine ⟨final,success,?_,frame⟩
  intro batch row head col hb hr hh hd
  by_cases rotary : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rotary
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have live : c.Active w := ⟨hb,hr,hh,hp⟩
    have got := values w live (c.shape.second col)
    rw [Native.pairWord_source] at got
    have words := congrArg Native.Node.value got
    rw [native_pair_formula c p (Native.project (KernelAlgebra.memory initial)) w live] at words
    simpa only [w,Config.cell,back,Native.atom] using words
  · rw [frame _ _ (tail_outside_typed c valid _ head col hh hd (by omega)),
      suffix _ head col (row_bound c valid batch row hb hr) hh hd (by omega)]
    simp only [nativeFormula,if_neg rotary,Native.project,KernelAlgebra.memory,KernelAlgebra.regionName]

theorem inplace_tail {α : Type} [Inhabited α] (c : Config) (initial : State α)
    (inplace : c.shape.inplace=Bool.true) : TailReady c initial := by
  intro token head col _ _ _ _
  simp [Shape.output,KernelAlgebra.regionName,inplace]

/-- Independent payload view at actual input Tensor pointers. The coordinate
formulas use only input, cosine and sine roles. No execution state or output
formula occurs in this input view. -/
def inputMemory {α : Type} (observe : Native.Word→α)
    (tensors : RegionName→KernelTensor.Tensor) (heap : KernelTensor.Heap) : Memory α :=
  fun role offset => observe (KernelTensor.word (tensors (KernelAlgebra.regionName role)) heap offset)

theorem real_formula_input (cfg : Config) (observe : Native.Word→ℝ) (c : Checked.Call)
    (tensors : RegionName→KernelTensor.Tensor) (heap : KernelTensor.Heap) (other : State ℝ) :
    realFormula cfg (KernelAlgebra.memory (KernelTensor.lift observe c tensors heap other)) =
      realFormula cfg (inputMemory observe tensors heap) := by
  funext batch row head col
  simp [realFormula,KernelAlgebra.memory,KernelTensor.lift,inputMemory,KernelAlgebra.regionName]

theorem native_formula_input (cfg : Config) (p : Native.Instructions) (c : Checked.Call)
    (tensors : RegionName→KernelTensor.Tensor) (heap : KernelTensor.Heap) (other : State Native.Node) :
    nativeFormula cfg p (Native.project (KernelAlgebra.memory (KernelTensor.lift Native.atom c tensors heap other))) =
      nativeFormula cfg p (inputMemory id tensors heap) := by
  funext batch row head col
  simp [nativeFormula,Native.project,KernelAlgebra.memory,KernelTensor.lift,
    inputMemory,KernelAlgebra.regionName,Native.atom]

/-- Actual Tensor inputs and both original signed metadata widths feed the
entire mathematical kernel theorem. Configuration is decoded from initial
Tensor storage, never supplied as an output oracle. -/
theorem tensor_real (c : Checked.Call) (tensors : RegionName→KernelTensor.Tensor)
    (binding : KernelTensor.Binding c tensors) (heap : KernelTensor.Heap)
    (observe : Native.Word→ℝ) (other : State ℝ)
    (accepted : Checked.check c (KernelTensor.bytes tensors heap)=Bool.true)
    (pointers : KernelIndex.pointerCheck (Checked.configuration c (KernelTensor.bytes tensors heap)) tensors=Bool.true)
    (suffix : TailReady (Checked.configuration c (KernelTensor.bytes tensors heap))
      (KernelTensor.lift observe c tensors heap other)) :
    let cfg := Checked.configuration c (KernelTensor.bytes tensors heap)
    let initial := KernelTensor.lift observe c tensors heap other
    ∃ final,TypedGrid.run realOps cfg c.host.abi.cumulative64 c.host.abi.offsets64 initial=some final ∧
      (∀ batch row head col,batch<cfg.shape.batches → row<cfg.length batch →
        head<cfg.shape.heads → col<cfg.shape.width →
        (final.mem (KernelAlgebra.regionName cfg.shape.output)
          (cfg.shape.address (cfg.start batch+row) head col)).read .real=
            realFormula cfg (inputMemory observe tensors heap) batch row head col) ∧
      (∀ r o,TypedGrid.Outside cfg r o → final.mem r o=initial.mem r o) := by
  have result := real _ (Checked.config_valid c _ accepted) _
    (KernelTensor.lift_metadata observe c tensors heap other accepted) suffix
    c.host.abi.cumulative64 c.host.abi.offsets64
  simpa only [real_formula_input] using result

theorem tensor_native (c : Checked.Call) (tensors : RegionName→KernelTensor.Tensor)
    (binding : KernelTensor.Binding c tensors) (heap : KernelTensor.Heap)
    (p : Native.Instructions) (other : State Native.Node)
    (accepted : Checked.check c (KernelTensor.bytes tensors heap)=Bool.true)
    (pointers : KernelIndex.pointerCheck (Checked.configuration c (KernelTensor.bytes tensors heap)) tensors=Bool.true)
    (suffix : TailReady (Checked.configuration c (KernelTensor.bytes tensors heap))
      (KernelTensor.lift Native.atom c tensors heap other)) :
    let cfg := Checked.configuration c (KernelTensor.bytes tensors heap)
    let initial := KernelTensor.lift Native.atom c tensors heap other
    ∃ final,TypedGrid.run (Native.ops p) cfg c.host.abi.cumulative64 c.host.abi.offsets64 initial=some final ∧
      (∀ batch row head col,batch<cfg.shape.batches → row<cfg.length batch →
        head<cfg.shape.heads → col<cfg.shape.width →
        ((final.mem (KernelAlgebra.regionName cfg.shape.output)
          (cfg.shape.address (cfg.start batch+row) head col)).read .real).value=
            nativeFormula cfg p (inputMemory id tensors heap) batch row head col) ∧
      (∀ r o,TypedGrid.Outside cfg r o → final.mem r o=initial.mem r o) := by
  have result := native _ (Checked.config_valid c _ accepted) p _
    (KernelTensor.lift_metadata Native.atom c tensors heap other accepted) suffix
    c.host.abi.cumulative64 c.host.abi.offsets64
  simpa only [native_formula_input] using result

#axiomsClean real
#axiomsClean every_token
#axiomsClean native
#axiomsClean tensor_real
#axiomsClean tensor_native
#specNonCircular inputMemory avoiding [TypedGrid.run,ProgramModel.result,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelEndpoints

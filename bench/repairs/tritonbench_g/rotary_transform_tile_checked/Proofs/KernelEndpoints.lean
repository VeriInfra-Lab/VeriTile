import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedGrid
import bench.repairs.tritonbench_g.rotary_transform.Spec

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelEndpoints
open VeriTile.Bench.Repairs.RotaryTransform
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


#axiomsClean real
#axiomsClean native
#axiomsClean every_token
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelEndpoints

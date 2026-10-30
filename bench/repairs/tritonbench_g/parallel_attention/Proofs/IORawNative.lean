import bench.repairs.tritonbench_g.parallel_attention.Proofs.IONative
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.NativeCombined

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 600000

/-- The original entry image includes every raw cell tag. In particular a
non-Real cell retains the original Real-reader default; no leaf assumption is
imposed on the public symbolic input carrier. -/
def rawInputs (shape : Shape) (s : State NativeStage.Word) : ChannelValues NativeStage.Expr (inputPorts shape) :=
  fun i=>match i with
  | ⟨0,_⟩ => fun j=>((NativeStage.entry s).mem "q" j.1.val).read .real
  | ⟨1,_⟩ => fun j=>((NativeStage.entry s).mem "k" j.1.val).read .real
  | ⟨2,_⟩ => fun j=>((NativeStage.entry s).mem "v" j.1.val).read .real
  | ⟨3,_⟩ => fun j=>((NativeStage.entry s).mem "do" j.1.val).read .real
  | ⟨4,_⟩ => fun j=>((NativeStage.entry s).mem "dz" j.1.val).read .real

theorem raw_binding (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) (s : State NativeStage.Word) :
    (dataIO NativeStage.Expr direction d.shape (nativeJobs direction d scale)).InputBinding
      (rawInputs d.shape s) (NativeStage.entry s) := by
  intro i j _
  fin_cases i <;> rfl

theorem native_laws_refl (a : Scalars NativeStage.Word) : NativeStage.PrimitiveLaws a a := by
  constructor <;> intros <;> rfl

private theorem raw_normalizer_at (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (p : Program d.shape 128) (s : State NativeStage.Word) (r : Fin 128) (active : p.chunk.val*128+r.val<d.shape.T) :
    NativeStage.forwardGridTarget a d scale p s "z" (normalizerAddress d.shape p.tile r.val)=
      Cell.mk .real (NativeStage.normalizerTarget a d scale (withProgram p s) r) := by
  simp only [NativeStage.forwardGridTarget,show ¬(("z" : RegionName)="o") by decide,ite_false]
  exact normalizerTarget_at d.shape p.tile _ (a.literal 0) r active

theorem raw_forward_result (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (s : State NativeStage.Word) :
    ∃ final,runPrograms (fun p before=>NativeStage.forwardRun a d scale (withProgram p before))
        (programs d.shape 128) s=some final ∧
      (∀ o j,outputActive .forward d.shape o j →
        final.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)=
          Cell.mk (outputPorts d.shape)[o].dtype (nativeExpected a d scale (rawInputs d.shape s) o j)) ∧
      (∀ R off,(dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).Outside
        (rawInputs d.shape s) R off → final.mem R off=s.mem R off) := by
  have agree:=inputs_agree .forward d.shape _ _ _ (raw_binding .forward d scale s)
  obtain ⟨final,run,_,reads,frame⟩:=NativeStage.forward_grid a a (native_laws_refl a) d scale s
  refine ⟨final,run,?_,?_⟩
  · intro o j active
    fin_cases o
    · let p:=program d.shape 128 d.shape.BV j
      have ha : (outputRectangle d.shape p.tile 128).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
      have read:=reads p (program_member d.shape 128 p) "o" _ (Or.inl ⟨rfl,_,ha,rfl⟩)
      simp only [NativeStage.forwardGridTarget,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [output_address] at read
      change final.mem "o" (outputWindow d.shape (⟨0,by decide⟩ : Fin 5) j)=Cell.mk .real _
      rw [read]
      apply congrArg (Cell.mk .real)
      rw [NativeStage.outputTarget,program_tile]
      change NativeStage.evaluate a .forward d p.tile
        ((forwardValues (NativeStage.algebra .forward d) (NativeStage.dots .forward d.shape) (forwardConfig d.shape scale) (withProgram p (NativeStage.entry s))).output ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩)=_
      rw [forward_bound _ _ .forward d.shape scale _ _ p agree]
      rfl
    · let p:=normProgram d.shape j
      have read:=reads p (program_member d.shape 128 p) "z" _
        (Or.inr ⟨active.2.1,rfl,_,active.2.2,rfl⟩)
      rw [raw_normalizer_at a d scale p s j.2.2.2.1 active.2.2,normalizer_address] at read
      change final.mem "z" (outputWindow d.shape (⟨1,by decide⟩ : Fin 5) j)=Cell.mk .real _
      rw [read]
      apply congrArg (Cell.mk .real)
      rw [NativeStage.normalizerTarget,program_tile]
      change NativeStage.evaluate a .forward d p.tile
        ((forwardValues (NativeStage.algebra .forward d) (NativeStage.dots .forward d.shape) (forwardConfig d.shape scale) (withProgram p (NativeStage.entry s))).normalizer ⟨j.2.2.2.1,()⟩)=_
      rw [forward_bound _ _ .forward d.shape scale _ _ p agree]
      rfl
    all_goals cases active.1
  · intro R off outside
    exact frame R off (fun p _=>forward_outside d.shape _ _ R off outside p)

theorem raw_backward_result (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (s : State NativeStage.Word) :
    ∃ final,runPrograms (NativeStage.combinedRun a d scale) (programs d.shape 64) s=some final ∧
      (∀ o j,outputActive .backward d.shape o j →
        final.mem (outputPorts d.shape)[o].buf (outputWindow d.shape o j)=
          Cell.mk (outputPorts d.shape)[o].dtype (nativeExpected a d scale (rawInputs d.shape s) o j)) ∧
      (∀ R off,(dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).Outside
        (rawInputs d.shape s) R off → final.mem R off=s.mem R off) := by
  have agree:=inputs_agree .backward d.shape _ _ _ (raw_binding .backward d scale s)
  obtain ⟨final,run,_,reads,frame⟩:=NativeStage.backward_grid a a (native_laws_refl a) d scale s
  refine ⟨final,run,?_,?_⟩
  · intro o j active
    fin_cases o
    · cases active.1
    · cases active.1
    · let p:=program d.shape 64 d.shape.BK j
      have ha : (queryRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
        simpa [queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
      have read:=reads p (program_member d.shape 64 p) "dq" _ (Or.inl ⟨rfl,_,ha,rfl⟩)
      simp only [NativeStage.backwardGridTarget,NativeStage.queryGridTarget,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [query_address] at read
      change final.mem "dq" (outputWindow d.shape (⟨2,by decide⟩ : Fin 5) j)=Cell.mk .real _
      rw [read]
      apply congrArg (Cell.mk .real)
      rw [NativeStage.queryTarget,query_bound _ _ d.shape scale _ _ p agree]
      rfl
    · let p:=program d.shape 64 d.shape.BK j
      have ha : (keyRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
        simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
      have read:=reads p (program_member d.shape 64 p) "dk" _ (Or.inr (Or.inl ⟨rfl,_,ha,rfl⟩))
      simp only [NativeStage.backwardGridTarget,NativeStage.keyValueGridTarget,
        show ¬(("dk" : RegionName)="dq") by decide,ite_false,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [key_address] at read
      change final.mem "dk" (outputWindow d.shape (⟨3,by decide⟩ : Fin 5) j)=Cell.mk .real _
      rw [read]
      apply congrArg (Cell.mk .real)
      rw [NativeStage.keyTarget,keyValue_bound _ _ d.shape scale _ _ p agree]
      rfl
    · let p:=program d.shape 64 d.shape.BV j
      have ha : (valueRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
      have read:=reads p (program_member d.shape 64 p) "dv" _ (Or.inr (Or.inr ⟨rfl,_,ha,rfl⟩))
      simp only [NativeStage.backwardGridTarget,NativeStage.keyValueGridTarget,
        show ¬(("dv" : RegionName)="dq") by decide,
        show ¬(("dv" : RegionName)="dk") by decide,ite_false] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [value_address] at read
      change final.mem "dv" (outputWindow d.shape (⟨4,by decide⟩ : Fin 5) j)=Cell.mk .real _
      rw [read]
      apply congrArg (Cell.mk .real)
      rw [NativeStage.valueTarget,keyValue_bound _ _ d.shape scale _ _ p agree]
      rfl
  · intro R off outside
    exact frame R off (fun p _=>backward_outside d.shape _ _ R off outside p)

#axiomsClean raw_binding
#axiomsClean raw_forward_result
#axiomsClean raw_backward_result
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData

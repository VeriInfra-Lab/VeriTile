import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsGeometry
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeLaunch

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
open VeriTile Triton FP.Structural
open FPStages (Stage)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def cellDtype {α : Type} : Cell α → TileDType
  | .mk d _ => d

theorem scatter_dtype {α β : Type} (xs : List β)
    (active : β → Prop) [DecidablePred active] (address : β → Nat)
    (d : TileDType) (value : β → Value α d) (s : State α) (r : RegionName) (off : Nat) :
    cellDtype ((xs.foldl (fun t i =>
      if active i then t.write r (address i) (.mk d (value i)) else t) s).mem r off)=
      xs.foldl (fun tag i => if active i ∧ off=address i then d else tag) (cellDtype (s.mem r off)) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    congr 1
    by_cases ha : active i <;> by_cases he : off=address i <;>
      simp [ha,he,State.write,cellDtype]

theorem range_dtype {α : Type} (f : Nat → State α → State α)
    (g : Nat → TileDType → TileDType) (r : RegionName) (off : Nat)
    (step : ∀ i s,cellDtype ((f i s).mem r off)=g i (cellDtype (s.mem r off)))
    (start stop : Nat) (s : State α) :
    cellDtype ((foldRange f start stop 1 s).mem r off)=
      foldRange g start stop 1 (cellDtype (s.mem r off)) := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  simp only [Nat.one_ne_zero,↓reduceDIte]
  split
  · rw [range_dtype f g r off step (start+1) stop,step]
  · rfl
termination_by stop-start
decreasing_by omega

variable {α : Type} [Inhabited α]

theorem program_attention_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : Stage) (s : State α) (off : Nat) :
    cellDtype ((FPStages.program M D stage s).mem "attention" off)=
      stageTag stage s.pids off (cellDtype (s.mem "attention" off)) := by
  cases stage with
  | inter c scale =>
    simp only [FPStages.program,stageTag,FPInter.enabled,FPInter.rowStart]
    split
    · rw [FPInter.written,scatter_dtype]
      simp only [FPInter.active,FPInter.outputAddress,FPInter.row,FPInter.column,FPInter.rowStart,and_assoc]
    · rfl
  | intra c BK scale =>
    simp only [FPStages.program,stageTag,FPDiagonal.enabled,FPDiagonal.rowStart,
      FPDiagonal.chunk,FPDiagonal.block,FPStages.diagonalLayout,Bool.false_eq_true,↓reduceIte]
    split
    · unfold FPDiagonal.written
      apply range_dtype
      intro j current
      simp only [FPDiagonal.columnWritten,FPDiagonal.out,Bool.false_eq_true,↓reduceIte]
      rw [scatter_dtype]
      rfl
    · rfl
  | split c scale =>
    rw [FPStages.program_frame M D (.split c scale) s "attention" off
      (by change ("attention" : RegionName)≠"scratch"; decide)]
    rfl
  | merge c =>
    simp only [FPStages.program,stageTag,FPMerge.enabled,FPMerge.row,FPStages.mergeLayout]
    split
    · rw [FPMerge.written,scatter_dtype]
      simp only [FPMerge.active,FPMerge.outputAddress,FPMerge.row,FPMerge.column,FPStages.mergeLayout,and_assoc]
    · rfl
  | output c scale =>
    rw [FPStages.program_frame M D (.output c scale) s "attention" off
      (by change ("attention" : RegionName)≠"o"; decide)]
    rfl

theorem source_attention_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : Stage) (idx : GridIndex stage.grid) (s t : State α)
    (ran : (NativeLaunch.job M D stage idx).run s=some t) (off : Nat) :
    cellDtype (t.mem "attention" off)=stageTag stage idx.toPids off (cellDtype (s.mem "attention" off)) := by
  obtain ⟨final,exec,mem⟩ := NativeLaunch.program_run M D stage idx s
  have same : t=final := Option.some.inj (ran.symm.trans exec)
  subst final
  rw [mem]
  exact program_attention_dtype M D stage (NativeLaunch.start stage idx s) off

#axiomsClean program_attention_dtype
#axiomsClean source_attention_dtype
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags

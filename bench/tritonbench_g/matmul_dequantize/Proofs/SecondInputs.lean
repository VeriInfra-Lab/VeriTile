import bench.tritonbench_g.matmul_dequantize.IO
import bench.tritonbench_g.matmul_dequantize.Proofs.FirstInputs

namespace VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open FPTypedSecond (Parameters Dot)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

open FirstIO (readAt_eq)

@[simp] theorem inner_at (p : Parameters) (s : State A) (pid : Nat) (i : Fin p.blockInner) :
    FPTypedSecond.inputInner p (GridExecution.atProgram s pid) i=i.val := by
  simp [FPTypedSecond.inputInner,GridExecution.atProgram]

theorem a_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p))
    (k : Nat) (hk : k<p.steps) (i : TileIndex [p.blockRows,p.blockInner])
    (hm : innerActive p k i.2.1.val) :
    (s.mem p.input (aAddress p pid k i.1.val i.2.1.val)).read .fp32=
      (operands zero p vals).a (aAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.input off).read .fp32)
  · intro j hj
    exact bound ⟨0,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),hm,rfl⟩

theorem b_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p))
    (k : Nat) (hk : k<p.steps) (i : TileIndex [p.blockInner,p.blockColumns])
    (hm : innerActive p k i.1.val) :
    (s.mem p.weight (bAddress p pid k i.1.val i.2.1.val)).read .int=
      (operands zero p vals).weight (bAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.weight off).read .int)
  · intro j hj
    exact bound ⟨1,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),hm,rfl⟩

theorem scale_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p))
    (k : Nat) (hk : k<p.steps) (i : TileIndex [p.blockInner,p.blockColumns])
    (hm : innerActive p k i.1.val) :
    (s.mem p.scale (sAddress p pid k i.1.val i.2.1.val)).read .fp32=
      (operands zero p vals).scale (sAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.scale off).read .fp32)
  · intro j hj
    exact bound ⟨2,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),hm,rfl⟩

theorem zero_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p))
    (k : Nat) (hk : k<p.steps) (i : TileIndex [p.blockInner,p.blockColumns])
    (hm : innerActive p k i.1.val) :
    (s.mem p.zero (zAddress p pid k i.1.val i.2.1.val)).read .int=
      (operands zero p vals).zero (zAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.zero off).read .int)
  · intro j hj
    exact bound ⟨3,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),hm,rfl⟩

theorem a_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p)) (k : Nat) (hk : k<p.steps) :
    FPTypedSecond.aValue M p (GridExecution.atProgram s pid) k=
      aValue M p (operands (M.literal none .real 0) p vals) pid k := by
  funext i
  simp only [FPTypedSecond.aValue,FPTypedSecond.innerActive,FPTypedSecond.aOffset,inner_at]
  norm_num only
  change M.cast (some .fp32) .fp32 .fp16
    (if innerActive p k i.2.1.val then
      (s.mem p.input (aAddress p pid k i.1.val i.2.1.val)).read .fp32 else _)=_
  unfold aValue
  congr 1
  split
  · exact a_read M D p jobs vals s bound _ pid hp k hk i ‹_›
  · rfl

theorem b_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (pid : Nat) (hp : pid<Geometry.count (SecondGrid.geometry p)) (k : Nat) (hk : k<p.steps) :
    FPTypedSecond.bValue M p (GridExecution.atProgram s pid) k=
      bValue M p (operands (M.literal none .real 0) p vals) pid k := by
  funext i
  simp only [FPTypedSecond.bValue,FPTypedSecond.difference,FPTypedSecond.packedLoad,
    FPTypedSecond.innerActive,FPTypedSecond.bOffset,FPTypedSecond.scaleOffset,FPTypedSecond.zeroOffset,inner_at]
  norm_num only
  change M.cast none .fp32 .fp16
    (M.binary (some .fp32) .fp32 .mul
      (M.cast (some .fp32) .real .fp32 (M.fromInt (some .fp32)
        (Real.signedNibble (if innerActive p k i.1.val then
          (s.mem p.weight (bAddress p pid k i.1.val i.2.1.val)).read .int else 0) i.1.val-
         Real.signedNibble (if innerActive p k i.1.val then
          (s.mem p.zero (zAddress p pid k i.1.val i.2.1.val)).read .int else 0) (column p pid i.2.1.val%p.columns))))
      (if innerActive p k i.1.val then
        (s.mem p.scale (sAddress p pid k i.1.val i.2.1.val)).read .fp32
       else M.cast none .real .fp32 (M.literal none .real 0)))=_
  by_cases hm : innerActive p k i.1.val
  · rw [if_pos hm,if_pos hm,if_pos hm,
      b_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i hm,
      scale_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i hm,
      zero_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i hm]
    simp only [bValue,difference,if_pos hm]
  · simp only [if_neg hm,bValue,difference]

theorem expression_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (j : OutputLane p) :
    FPTypedSecond.outputValue M D p (GridExecution.atProgram s j.1.val) j.2=
      expression M D p vals ⟨0,by simp [outputChannels]⟩ j := by
  have accum : ∀ n,n≤p.steps →
      FPTypedSecond.accumulation M D p (GridExecution.atProgram s j.1.val) n=
        accumulation M D p (operands (M.literal none .real 0) p vals) j.1.val n := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro hn
      simp only [FPTypedSecond.accumulation,accumulation,ih (by omega),
        a_binding M D p jobs vals s bound j.1.val j.1.isLt n (by omega),
        b_binding M D p jobs vals s bound j.1.val j.1.isLt n (by omega)]
  change M.cast none .real .fp16 (FPTypedSecond.accumulation M D p (GridExecution.atProgram s j.1.val) (p.steps) j.2)=_
  rw [accum (p.steps) (by omega)]
  rfl

#axiomsClean expression_binding
end
end VeriTile.Bench.Repairs.MatmulDequantize.SecondIO

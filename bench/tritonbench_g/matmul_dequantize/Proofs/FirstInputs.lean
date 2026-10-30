import bench.tritonbench_g.matmul_dequantize.IO

namespace VeriTile.Bench.Repairs.MatmulDequantize.FirstIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.MatmulDequantize
open FPTypedFirst (Parameters steps Dot)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem readAt_eq {C : Type} {shape : TileShape} (zero : C) (values : TileIndex shape → C)
    (mask : TileIndex shape → Prop) (address : TileIndex shape → Nat)
    (read : Nat → C) (bound : ∀ j,mask j → read (address j)=values j)
    (off : Nat) (witness : ∃ j,mask j ∧ address j=off) :
    read off=readAt zero values mask address off := by
  classical
  rw [readAt,dif_pos witness]
  have chosen := Classical.choose_spec witness
  exact (congrArg read chosen.2).symm.trans (bound (Classical.choose witness) chosen.1)

theorem a_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p))
    (k : Nat) (hk : k<steps p) (i : TileIndex [p.blockRows,p.blockInner])
    (hm : row p pid i.1.val<p.rows) :
    (s.mem p.input (aAddress p pid k i.1.val i.2.1.val)).read .fp16=
      (operands zero p vals).a (aAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.input off).read .fp16)
  · intro j hj
    exact bound ⟨0,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),hm,rfl⟩

theorem b_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p))
    (k : Nat) (hk : k<steps p) (i : TileIndex [p.blockInner,p.blockColumns]) :
    (s.mem p.weight (bAddress p pid k i.1.val i.2.1.val)).read .int=
      (operands zero p vals).weight (bAddress p pid k i.1.val i.2.1.val) := by
  apply readAt_eq (read := fun off => (s.mem p.weight off).read .int)
  · intro j hj
    exact bound ⟨1,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,hk⟩,i),trivial,rfl⟩

theorem scale_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p))
    (k : Nat) (hk : k<steps p) (i : Fin p.blockColumns) :
    (s.mem p.scale (sAddress p pid k i.val)).read .fp16=
      (operands zero p vals).scale (sAddress p pid k i.val) := by
  apply readAt_eq (read := fun off => (s.mem p.scale off).read .fp16)
  · intro j hj
    exact bound ⟨2,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,lt_of_lt_of_le hk (le_max_left _ _)⟩,i,()),Or.inl hk,rfl⟩

theorem zero_read (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (zero : A) (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p))
    (k : Nat) (hk : k<steps p) (i : Fin p.blockColumns) :
    (s.mem p.zero (zAddress p pid k i.val)).read .int=
      (operands zero p vals).zero (zAddress p pid k i.val) := by
  apply readAt_eq (read := fun off => (s.mem p.zero off).read .int)
  · intro j hj
    exact bound ⟨3,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  · exact ⟨(⟨pid,hp⟩,⟨k,lt_of_lt_of_le hk (le_max_left _ _)⟩,i,()),Or.inl hk,rfl⟩

theorem a_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p)) (k : Nat) (hk : k<steps p) :
    FPTypedFirst.aValue M p (GridExecution.atProgram s pid) k=
      aValue M p (operands (M.literal none .real 0) p vals) pid k := by
  funext i
  change (if row p pid i.1.val<p.rows then
    (s.mem p.input (aAddress p pid k i.1.val i.2.1.val)).read .fp16 else _)=_
  unfold aValue
  split
  · exact a_read M D p jobs vals s bound _ pid hp k hk i ‹_›
  · rfl

theorem b_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (pid : Nat) (hp : pid<Geometry.count (FirstGrid.geometry p)) (k : Nat) (hk : k<steps p) :
    FPTypedFirst.bValue M p (GridExecution.atProgram s pid) k=
      bValue M p (operands (M.literal none .real 0) p vals) pid k := by
  funext i
  have bw := b_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i
  have bs := scale_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i.2.1
  have bz := zero_read M D p jobs vals s bound (M.literal none .real 0) pid hp k hk i.2.1
  change M.binary none .fp16 .sub
    (M.binary (some .fp32) .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32)
        (Real.signedNibble ((s.mem p.weight (bAddress p pid k i.1.val i.2.1.val)).read .int) i.1.val)))
      ((s.mem p.scale (sAddress p pid k i.2.1.val)).read .fp16))
    (M.binary none .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32)
        (Real.signedNibble ((s.mem p.zero (zAddress p pid k i.2.1.val)).read .int) (column p pid i.2.1.val))))
      ((s.mem p.scale (sAddress p pid k i.2.1.val)).read .fp16))=_
  rw [bw,bs,bz]
  rfl

theorem expression_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M D p jobs).InputBinding vals s)
    (j : OutputLane p) :
    FPTypedFirst.outputValue M D p (GridExecution.atProgram s j.1.val) j.2=
      expression M D p vals ⟨0,by simp [outputChannels]⟩ j := by
  have accum : ∀ n,n≤steps p →
      FPTypedFirst.accumulation M D p (GridExecution.atProgram s j.1.val) n=
        accumulation M D p (operands (M.literal none .real 0) p vals) j.1.val n := by
    intro n
    induction n with
    | zero => intro _; rfl
    | succ n ih =>
      intro hn
      simp only [FPTypedFirst.accumulation,accumulation,ih (by omega),
        a_binding M D p jobs vals s bound j.1.val j.1.isLt n (by omega),
        b_binding M D p jobs vals s bound j.1.val j.1.isLt n (by omega)]
  change M.cast none .real .fp16 (FPTypedFirst.accumulation M D p (GridExecution.atProgram s j.1.val) (steps p) j.2)=_
  rw [accum (steps p) (by omega)]
  rfl

#axiomsClean expression_binding
end
end VeriTile.Bench.Repairs.MatmulDequantize.FirstIO

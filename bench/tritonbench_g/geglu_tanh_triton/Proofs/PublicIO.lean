import bench.tritonbench_g.geglu_tanh_triton.IO
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Grid
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Rewrite
import bench.tritonbench_g.geglu_tanh_triton.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Reference
open VeriTile Triton FP.Structural
noncomputable section

def mathematical (_candidate backward : Bool) (count : Nat) (_order : List Nat) :
    ChannelValues ℝ (inputs count) → ChannelValues ℝ (outputs count) :=
  formula backward count RealExecution.formula

end
end VeriTile.Bench.Repairs.GegluTanhTriton.Reference

namespace VeriTile.Bench.Repairs.GegluTanhTriton.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α] [Inhabited β]

theorem launch_initial (M : Algebra β) (candidate backward : Bool) (count pid : Nat) (s : State β) :
    (IO.launch M candidate backward count pid).initial s=Tile.rowState pid s := by rfl

theorem run_eq (M : Algebra β) (candidate backward : Bool) (count : Nat) (order : List Nat) (s : State β) :
    runSchedule (order.map (IO.launch M candidate backward count)) s=Grid.runRows M backward 128 count order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Grid.runRows,Invocation.run,launch_initial,FP.Structural.exec]
    change (run M (selectedSource backward candidate 128 count).surfaceBody (Tile.rowState pid s)).bind _=_
    rw [Rewrite.selected_run]
    congr 1
    funext t
    exact ih t

theorem implements (M : Algebra β) (candidate backward : Bool) (count : Nat) (order : List Nat)
    (legal : order.Perm (List.range (Grid.grid count))) (encode : α → β) (observe : β → α)
    (formula : Channel → α → α → α → α)
    (scalar : ∀ ch a b dc,observe (Tile.scalar M ch (encode a) (encode b) (encode dc))=formula ch a b dc) :
    (IO.program M candidate backward count order encode observe).Implements
      (Reference.formula backward count formula) := by
  intro vals _ bounds bounded initial binding
  dsimp only [ProgramIO.InBounds,ProgramIO.InputBinding,IO.program] at bounded binding
  have inputBounds (i : Nat) (hi : i<count) :
      i<bounds "A" ∧ i<bounds "B" ∧ (backward=Bool.true → i<bounds "DC") := by
    refine ⟨?_,?_,?_⟩
    · simpa using bounded.2.1 ⟨0,by simp [IO.program]⟩ (⟨i,hi⟩,()) (Or.inl (by simp))
    · simpa using bounded.2.1 ⟨1,by simp [IO.program]⟩ (⟨i,hi⟩,()) (Or.inl (by simp))
    · intro enabled
      simpa using bounded.2.1 ⟨2,by simp [IO.program]⟩ (⟨i,hi⟩,()) (Or.inr enabled)
  have outputBounds (ch : Channel) (on : enabled backward ch) (i : Nat) (hi : i<count) :
      i<bounds (outputRegion ch) := by
    cases ch
    · simpa [outputRegion] using bounded.2.2.1 ⟨0,by simp [IO.program]⟩ (⟨i,hi⟩,()) (by simpa [IO.channel] using on)
    · simpa [outputRegion] using bounded.2.2.1 ⟨1,by simp [IO.program]⟩ (⟨i,hi⟩,()) (by simpa [IO.channel] using on)
    · simpa [outputRegion] using bounded.2.2.1 ⟨2,by simp [IO.program]⟩ (⟨i,hi⟩,()) (by simpa [IO.channel] using on)
  have safe (jobs : List Nat) (s : State β) :
      ScheduleSafe bounds (jobs.map (IO.launch M candidate backward count)) s := by
    induction jobs generalizing s with
    | nil => trivial
    | cons pid rest ih =>
      exact ⟨AccessSafety.source_safe M candidate backward count 128 bounds _ inputBounds outputBounds,
        fun t _=>ih t⟩
  refine ⟨safe order initial,?_⟩
  obtain ⟨final,ran,values,frame⟩ := Grid.whole M backward count order legal initial
  refine ⟨final,(run_eq M candidate backward count order initial).trans ran,?_,?_⟩
  · have output (ch : Channel) (on : enabled backward ch) (i : Fin count) :
        mapCell observe (final.mem (outputRegion ch) i.val)=Cell.mk .real
          (formula ch (vals ⟨0,by simp [IO.program]⟩ (i,()))
            (vals ⟨1,by simp [IO.program]⟩ (i,()))
            (if backward then vals ⟨2,by simp [IO.program]⟩ (i,()) else default)) := by
      rw [values ch on i.val i.isLt]
      have a := binding ⟨0,by simp [IO.program]⟩ (i,()) (Or.inl (by simp))
      have b := binding ⟨1,by simp [IO.program]⟩ (i,()) (Or.inl (by simp))
      change (initial.mem "A" i.val).read .real=encode (vals ⟨0,by simp [IO.program]⟩ (i,())) at a
      change (initial.mem "B" i.val).read .real=encode (vals ⟨1,by simp [IO.program]⟩ (i,())) at b
      simp only [mapCell,mapValue,a,b]
      cases backward
      · have hc : ch=.c := on
        subst ch
        exact congrArg (Cell.mk .real) (scalar .c _ _ default)
      · have dc := binding ⟨2,by simp [IO.program]⟩ (i,()) (Or.inr rfl)
        change (initial.mem "DC" i.val).read .real=encode (vals ⟨2,by simp [IO.program]⟩ (i,())) at dc
        rw [dc,scalar]
        rfl
    intro o j on
    fin_cases o
    all_goals
      simpa [Reference.formula,IO.program,IO.channel] using output _ on j.1
  · intro region address outside
    dsimp only [ProgramIO.Outside,IO.program] at outside
    apply frame region address
    intro ch on
    by_cases active : address<count
    · left
      cases ch
      · simpa [outputRegion] using outside.1 ⟨0,by simp [IO.program]⟩ (⟨address,active⟩,()) (by simpa [IO.channel] using on)
      · have absent := outside.1 ⟨1,by simp [IO.program]⟩ (⟨address,active⟩,()) (by simpa [IO.channel] using on)
        change region≠outputRegion .da ∨ address≠address at absent
        exact absent.resolve_right (fun h=>h rfl)
      · have absent := outside.1 ⟨2,by simp [IO.program]⟩ (⟨address,active⟩,()) (by simpa [IO.channel] using on)
        change region≠outputRegion .db ∨ address≠address at absent
        exact absent.resolve_right (fun h=>h rfl)
    · exact Or.inr (by omega)

theorem real (candidate backward : Bool) (count : Nat) (order : List Nat)
    (legal : order.Perm (List.range (Grid.grid count))) :
    (Reference.realIO candidate backward count order).Implements
      (Reference.mathematical candidate backward count order) :=
  implements RealExecution.algebra candidate backward count order legal id id
    RealExecution.formula RealExecution.scalar_real

#axiomsClean real
end
end VeriTile.Bench.Repairs.GegluTanhTriton.PublicIO

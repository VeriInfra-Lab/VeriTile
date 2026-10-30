import bench.tritonbench_g.fused_layernorm_triton.IO
import bench.tritonbench_g.fused_layernorm_triton.Proofs.SourceSafety
import bench.tritonbench_g.fused_layernorm_triton.Proofs.KernelValues

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem initial_eq (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (row : Nat) (s : State α) :
    (invocation M eq32 X W Bias Y Mean Rstd rows D c row).initial s=atRow s row := rfl

theorem schedule_eq (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (invocation M eq32 X W Bias Y Mean Rstd rows D c)) s=
      runRows (withEquality M eq32) (Contract.source X W Bias Y Mean Rstd rows D c) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons row jobs ih =>
    simp only [List.map_cons,runSchedule,runRows]
    change (exec (withEquality M eq32) (Contract.source X W Bias Y Mean Rstd rows D c)
      (atRow s row)).bind _ = _
    cases h : exec (withEquality M eq32) (Contract.source X W Bias Y Mean Rstd rows D c) (atRow s row) <;>
      simp [ih]

theorem schedule_safe (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (bounds : RegionName → Nat)
    (bx : ∀ r j,r<rows → j<D → j+D*r<bounds X)
    (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ r j,r<rows → j<D → j+D*r<bounds Y)
    (bm : ∀ r,r<rows → r<bounds Mean) (br : ∀ r,r<rows → r<bounds Rstd)
    (jobs : List Nat) (legal : ∀ r∈jobs,r<rows) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation M eq32 X W Bias Y Mean Rstd rows D c)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons row jobs ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => legal r (by simp [hr])) t⟩
    have active := legal row (by simp)
    exact SourceSafety.source_safe M eq32 (atRow s row) X W Bias Y Mean Rstd rows D c bounds
      (fun j hj => bx row j active hj) bw bb (fun j hj => byOut row j active hj)
      (bm row active) (br row active)

/-- Connect an established complete row-grid value theorem to shared IO.
The successful execution callback is a proved helper premise, never public IO data. -/
theorem implements (M : Algebra α) (eq32 : α → α → Bool) (observe : α → α)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (f : (Nat → Nat → α) → (Nat → α) → (Nat → α) → Nat → α×α×(Nat → α))
    (result : ∀ (s : State α) (xs : Nat → Nat → α) (ws bs : Nat → α),
      (∀ row,row<rows → ∀ col,col<D → (s.mem X (col+D*row)).read .real=xs row col) →
      (∀ col,col<D → (s.mem W col).read .real=ws col) →
      (∀ col,col<D → (s.mem Bias col).read .real=bs col) →
      ∃ final, KernelProof.launch M eq32 X W Bias Y Mean Rstd rows D c s=some final ∧
        (∀ row,row<rows →
          mapCell observe (final.mem Mean row)=.mk .real (f xs ws bs row).1 ∧
          mapCell observe (final.mem Rstd row)=.mk .real (f xs ws bs row).2.1 ∧
          ∀ col,col<D → mapCell observe (final.mem Y (col+D*row))=.mk .bf16 ((f xs ws bs row).2.2 col)) ∧
        (∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off)) :
    (io M eq32 observe X W Bias Y Mean Rstd rows D c).Implements (expected X W Bias Y Mean Rstd rows D f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bound initial _
    apply schedule_safe M eq32 X W Bias Y Mean Rstd rows D c bounds
    · intro r j hr hj
      exact bound.2.1 ⟨0,by simp [io,inputs]⟩ (⟨r,hr⟩,⟨j,hj⟩,()) trivial
    · intro j hj
      exact bound.2.1 ⟨1,by simp [io,inputs]⟩ (⟨j,hj⟩,()) trivial
    · intro j hj
      exact bound.2.1 ⟨2,by simp [io,inputs]⟩ (⟨j,hj⟩,()) trivial
    · intro r j hr hj
      exact bound.2.2.1 ⟨2,by simp [io,outputs]⟩ (⟨r,hr⟩,⟨j,hj⟩,()) trivial
    · intro r hr
      exact bound.2.2.1 ⟨0,by simp [io,outputs]⟩ (⟨r,hr⟩,()) trivial
    · intro r hr
      exact bound.2.2.1 ⟨1,by simp [io,outputs]⟩ (⟨r,hr⟩,()) trivial
    · exact fun r hr => List.mem_range.mp hr
  · intro vals initial hin
    have hx : ∀ row,row<rows → ∀ col,col<D →
        (initial.mem X (col+D*row)).read .real=arrayX X W Bias rows D vals row col := by
      intro row hr col hc
      simpa [ProgramIO.InputBinding,io,inputs,arrayX,hr,hc,mapValue] using
        hin ⟨0,by simp [io,inputs]⟩ (⟨row,hr⟩,⟨col,hc⟩,()) trivial
    have hw : ∀ col,col<D → (initial.mem W col).read .real=arrayW X W Bias rows D vals col := by
      intro col hc
      simpa [ProgramIO.InputBinding,io,inputs,arrayW,hc,mapValue] using
        hin ⟨1,by simp [io,inputs]⟩ (⟨col,hc⟩,()) trivial
    have hb : ∀ col,col<D → (initial.mem Bias col).read .real=arrayB X W Bias rows D vals col := by
      intro col hc
      simpa [ProgramIO.InputBinding,io,inputs,arrayB,hc,mapValue] using
        hin ⟨2,by simp [io,inputs]⟩ (⟨col,hc⟩,()) trivial
    obtain ⟨final,ran,value,frame⟩ := result initial _ _ _ hx hw hb
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_eq M eq32 X W Bias Y Mean Rstd rows D c (List.range rows) initial).trans ran
    · intro port lane _
      fin_cases port
      · exact (value lane.1.val lane.1.isLt).1
      · exact (value lane.1.val lane.1.isLt).2.1
      · exact (value lane.1.val lane.1.isLt).2.2 lane.2.1.val lane.2.1.isLt
    · intro q off outside
      apply frame q off
      intro row hr
      refine ⟨?_,?_,?_⟩
      · exact outside.1 ⟨0,by simp [io,outputs]⟩ (⟨row,hr⟩,()) trivial
      · exact outside.1 ⟨1,by simp [io,outputs]⟩ (⟨row,hr⟩,()) trivial
      · by_cases hq : q≠Y
        · exact .inl hq
        · right
          intro col hc
          exact (outside.1 ⟨2,by simp [io,outputs]⟩ (⟨row,hr⟩,⟨col,hc⟩,()) trivial).resolve_left hq

#axiomsClean implements
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO

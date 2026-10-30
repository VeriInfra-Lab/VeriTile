import bench.tritonbench_g.attention_score.Proofs.Reused.FPGrid

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPReturned
open VeriTile Triton FP.Structural FP.ProfiledRewrite Geometry FPEngine FPGrid
open WholeGrid (Job jobs)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def ownerJob (c : Shape) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : Job c :=
  (z,h,⟨j.val/c.B,(owner c j).1⟩)
def ownerLane (c : Shape) (j : Fin c.NK) : Fin c.B := ⟨j.val%c.B,(owner c j).2.1⟩
def target (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (s : State α) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : α :=
  values M dot c offset size scale window complement s (ownerJob c z h j) (ownerLane c j)

theorem grid_tensor (M : Algebra α) (dot : Dot α) (c : Shape) (s : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ t, run M dot c offset size scale window complement (jobs c) s=some t ∧ InputsSame s t ∧
      ∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        t.mem "out" (c.outputBase z.val h.val+j.val)=.mk .real (target M dot c offset size scale window complement s z h j) := by
  obtain ⟨t,ht,hs,ho⟩ := whole_grid M dot c s offset size scale window complement
  refine ⟨t,ht,hs,?_⟩
  intro z h j
  have he := (owner c j).2.2
  have hv := ho (ownerJob c z h j) (ownerLane c j) (by
    simpa [WholeGrid.active,ownerJob,ownerLane,he] using j.isLt)
  simpa [target,WholeGrid.address,ownerJob,ownerLane,Nat.add_assoc,he] using hv

theorem invoke_frame (M : Algebra α) (dot : Dot α) (c : Shape) (s t : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (job : Job c)
    (hrun : invoke M dot c offset size scale window complement job s=some t)
    (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) : t.mem r a=s.mem r a := by
  let p := parameters c offset size scale window complement
  let initial := programState c s job
  obtain ⟨u,hu,hmem⟩ := FPOutput.source_memory M dot c.B c.D p initial
  have he : u=t := Option.some.inj (hu.symm.trans hrun)
  subst t
  rw [hmem]
  apply FPOutput.frame
  rcases hf with hr|ha
  · exact Or.inl hr
  · right
    intro i hi heq
    rcases i with ⟨i,⟨⟩⟩
    have hb := output_bound c job.1 job.2.1 ⟨job.2.2.val*c.B+i.val,hi⟩
    have ha' : c.outputSize≤WholeGrid.address c job i := by
      have he := heq.trans (output_address c s job offset size scale window complement i)
      simpa only [he] using ha
    exact Nat.not_lt_of_ge ha' (by simpa [WholeGrid.address,Nat.add_assoc] using hb)

theorem run_frame (M : Algebra α) (dot : Dot α) (c : Shape) (offset size : Int)
    (scale : ℝ) (window complement : Bool) (order : List (Job c)) (s t : State α)
    (hrun : run M dot c offset size scale window complement order s=some t)
    (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) : t.mem r a=s.mem r a := by
  induction order generalizing s with
  | nil => cases hrun; rfl
  | cons job rest ih =>
      simp only [FPGrid.run] at hrun
      cases hi : invoke M dot c offset size scale window complement job s with
      | none => simp [hi] at hrun
      | some u =>
          have hu : run M dot c offset size scale window complement rest u=some t := by simpa [hi] using hrun
          exact (ih u hu).trans (invoke_frame M dot c s u offset size scale window complement job hi r a hf)

def zeroOutput (M : Algebra α) (c : Shape) (s : State α) : State α :=
  {s with mem := fun r a => if r="out" ∧ a<c.outputSize then .mk .real (FPContext.zero M) else s.mem r a}

theorem zero_inputs (M : Algebra α) (c : Shape) (s : State α) : InputsSame s (zeroOutput M c s) := by
  intro r a hr
  simp [zeroOutput,hr]

def hostRun (M : Algebra α) (dot : Dot α) (c : Shape) (s : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (State α) :=
  let initialized := zeroOutput M c s
  if c.N=0 ∨ c.outputSize=0 then some initialized
  else run M dot c offset size scale window complement (jobs c) initialized

/-- Empty query returns the freshly allocated FP32 zero. Nonempty queries
return the source's explicit FP32-profile cast of the primitive recurrence. -/
def returnedValue (M : Algebra α) (dot : Dot α) (c : Shape) (s : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : α :=
  if c.N=0 then FPContext.zero M else target M dot c offset size scale window complement s z h j

theorem host_returned (M : Algebra α) (dot : Dot α) (c : Shape) (s : State α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ t, hostRun M dot c s offset size scale window complement=some t ∧ InputsSame s t ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        t.mem "out" (c.outputBase z.val h.val+j.val)=.mk .real (returnedValue M dot c s offset size scale window complement z h j)) ∧
      (∀ (r : RegionName) a, (r≠"out" ∨ c.outputSize≤a) → t.mem r a=s.mem r a) := by
  have zframe (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) :
      (zeroOutput M c s).mem r a=s.mem r a := by
    rcases hf with hr|ha
    · simp [zeroOutput,hr]
    · simp [zeroOutput,Nat.not_lt.mpr ha]
  by_cases hempty : c.N=0 ∨ c.outputSize=0
  · refine ⟨zeroOutput M c s,by simp [hostRun,hempty],zero_inputs M c s,?_,zframe⟩
    intro z h j
    have hb := output_bound c z h j
    rcases hempty with hn|ho
    · simp [returnedValue,hn,zeroOutput,hb]
    · rw [ho] at hb
      omega
  · obtain ⟨t,ht,hs,hv⟩ := grid_tensor M dot c (zeroOutput M c s) offset size scale window complement
    refine ⟨t,by simp [hostRun,hempty,ht],?_,?_,?_⟩
    · intro r a hr
      exact (hs r a hr).trans (zero_inputs M c s r a hr)
    · intro z h j
      rw [hv]
      simp only [returnedValue,if_neg (fun h => hempty (Or.inl h)),target,
        values_same M dot c offset size scale window complement s (zeroOutput M c s) (zero_inputs M c s)]
    · intro r a hf
      exact (run_frame M dot c offset size scale window complement (jobs c) (zeroOutput M c s) t ht r a hf).trans (zframe r a hf)

#specNonCircular returnedValue avoiding [FP.Structural.exec,FPGrid.run,hostRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPReturned

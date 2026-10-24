import bench.repairs.tritonbench_g.attention_score.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.AttentionScore.RealReturned
open VeriTile.Triton Geometry DenseSource WholeGrid
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def target (c : Shape) (s : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : ℝ :=
  LogicalFormula.column (inputs c s z.val h.val) scale c.N window complement offset size j.val

theorem grid_tensor (c : Shape) (s : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) :
    ∃ t, run c offset size scale window complement (jobs c) s=some t ∧ InputsSame s t ∧
      ∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        t.readMem "out" (c.outputBase z.val h.val+j.val)=target c s offset size scale window complement z h j := by
  obtain ⟨t,ht,hs,ho⟩ := whole_grid c s offset size scale window complement
  refine ⟨t,ht,hs,?_⟩
  intro z h j
  obtain ⟨hb,hl,he⟩ := owner c j
  let job : Job c := (z,h,⟨j.val/c.B,hb⟩)
  let lane : Fin c.B := ⟨j.val%c.B,hl⟩
  have hv := ho job lane (by simpa [active,job,lane,he] using j.isLt)
  simpa [target,address,job,lane,Nat.add_assoc,he] using hv

theorem invoke_frame (c : Shape) (s t : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (job : Job c)
    (hrun : invoke c offset size scale window complement job s=some t)
    (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) : t.mem r a=s.mem r a := by
  obtain ⟨u,hu,_,hframe⟩ := DenseSource.executes c s job.2.2.val job.1.val job.2.1
    offset size scale window complement
  have he : u=t := Option.some.inj (hu.symm.trans hrun)
  subst t
  apply hframe r a
  rcases hf with hr | ha
  · exact Or.inl hr
  · right
    intro lane hl heq
    have hb := output_bound c job.1 job.2.1 ⟨job.2.2.val*c.B+lane.val,hl⟩
    rw [heq] at ha
    simp only [Nat.add_assoc] at ha
    exact Nat.not_lt_of_ge ha hb

theorem run_frame (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order : List (Job c)) (s t : BlockState)
    (hrun : run c offset size scale window complement order s=some t)
    (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) : t.mem r a=s.mem r a := by
  induction order generalizing s with
  | nil => cases hrun; rfl
  | cons job rest ih =>
      simp only [run] at hrun
      cases hi : invoke c offset size scale window complement job s with
      | none => simp [hi] at hrun
      | some u =>
          have hu : run c offset size scale window complement rest u=some t := by simpa [hi] using hrun
          exact (ih u hu).trans (invoke_frame c s u offset size scale window complement job hi r a hf)

def zeroOutput (c : Shape) (s : BlockState) : BlockState :=
  {s with mem := fun r a => if r="out" ∧ a<c.outputSize then MemCell.real 0 else s.mem r a}

theorem zero_inputs (c : Shape) (s : BlockState) : InputsSame s (zeroOutput c s) := by
  intro r a hr
  simp [zeroOutput,hr]

theorem zero_value (c : Shape) (s : BlockState) (a : Nat) (ha : a<c.outputSize) :
    (zeroOutput c s).readMem "out" a=0 := by
  have hm : (zeroOutput c s).mem "out" a=MemCell.real 0 := by simp [zeroOutput,ha]
  unfold BlockState.readMem
  rw [hm]
  rfl

/-- Ordinary host allocation/dispatch model: allocate zeroed output, return
immediately on empty query/output, otherwise invoke the entire selected grid. -/
def hostRun (c : Shape) (s : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : Option BlockState :=
  let initialized := zeroOutput c s
  if c.N=0 ∨ c.outputSize=0 then some initialized
  else run c offset size scale window complement (jobs c) initialized

theorem host_returned (c : Shape) (s : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) :
    ∃ t, hostRun c s offset size scale window complement=some t ∧ InputsSame s t ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        t.readMem "out" (c.outputBase z.val h.val+j.val)=target c s offset size scale window complement z h j) ∧
      (∀ (r : RegionName) a, (r≠"out" ∨ c.outputSize≤a) → t.mem r a=s.mem r a) := by
  have zframe (r : RegionName) (a : Nat) (hf : r≠"out" ∨ c.outputSize≤a) :
      (zeroOutput c s).mem r a=s.mem r a := by
    rcases hf with hr | ha
    · simp [zeroOutput,hr]
    · simp [zeroOutput,Nat.not_lt.mpr ha]
  by_cases hempty : c.N=0 ∨ c.outputSize=0
  · refine ⟨zeroOutput c s,by simp [hostRun,hempty],zero_inputs c s,?_,zframe⟩
    intro z h j
    have hb := output_bound c z h j
    rcases hempty with hn | ho
    · rw [zero_value c s _ hb]
      unfold target LogicalFormula.column
      rw [Fin.sum_univ_eq_sum_range (fun row => LogicalFormula.weight (inputs c s z.val h.val) scale window complement offset size row j.val),hn]
      simp
    · rw [ho] at hb
      omega
  · obtain ⟨t,ht,hs,hv⟩ := grid_tensor c (zeroOutput c s) offset size scale window complement
    refine ⟨t,by simp [hostRun,hempty,ht],?_,?_,?_⟩
    · intro r a hr
      exact (hs r a hr).trans (zero_inputs c s r a hr)
    · intro z h j
      have ho := hv z h j
      simpa only [target,slice_same c s (zeroOutput c s) (zero_inputs c s)] using ho
    · intro r a hf
      exact (run_frame c offset size scale window complement (jobs c) (zeroOutput c s) t ht r a hf).trans
        (zframe r a hf)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.RealReturned

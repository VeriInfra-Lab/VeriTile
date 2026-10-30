import bench.repairs.tritonbench_g.attention_score.Proofs.DenseSafety

namespace VeriTile.Bench.Repairs.AttentionScore.WholeGrid
open VeriTile.Triton Geometry DenseSource
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

abbrev Job (c : Shape) := Fin c.Z × Fin c.H × Fin c.programs
def jobs (c : Shape) : List (Job c) := Finset.univ.toList
def address (c : Shape) (job : Job c) (lane : Fin c.B) : Nat :=
  c.outputBase job.1.val job.2.1.val + job.2.2.val*c.B+lane.val
def active (c : Shape) (job : Job c) (lane : Fin c.B) : Prop :=
  job.2.2.val*c.B+lane.val<c.NK

theorem address_owner (c : Shape) (a b : Job c) (i j : Fin c.B)
    (ha : active c a i) (hb : active c b j) (he : address c a i = address c b j) : a=b := by
  have hn : 0<c.NK := Nat.zero_lt_of_lt ha
  have hH : 0<c.H := Nat.zero_lt_of_lt a.2.1.isLt
  have aa := owner_unique c.NK (address c a i) (a.1.val*c.H+a.2.1.val)
    (a.2.2.val*c.B+i.val) hn ha (by simp [address,Shape.outputBase,Nat.add_assoc])
  have bb := owner_unique c.NK (address c a i) (b.1.val*c.H+b.2.1.val)
    (b.2.2.val*c.B+j.val) hn hb (by simpa [address,Shape.outputBase,Nat.add_assoc] using he.symm)
  have hzh := aa.1.trans bb.1.symm
  have hkey := aa.2.trans bb.2.symm
  have az := owner_unique c.H (a.1.val*c.H+a.2.1.val) a.1.val a.2.1.val hH a.2.1.isLt rfl
  have bz := owner_unique c.H (a.1.val*c.H+a.2.1.val) b.1.val b.2.1.val hH b.2.1.isLt hzh.symm
  have hB := supported_positive c.B c.block
  have ab := owner_unique c.B (a.2.2.val*c.B+i.val) a.2.2.val i.val hB i.isLt rfl
  have bb' := owner_unique c.B (a.2.2.val*c.B+i.val) b.2.2.val j.val hB j.isLt hkey.symm
  exact Prod.ext (Fin.ext (az.1.trans bz.1.symm))
    (Prod.ext (Fin.ext (az.2.trans bz.2.symm)) (Fin.ext (ab.1.trans bb'.1.symm)))

def invoke (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : Job c) (s : BlockState) : Option BlockState :=
  exec (kernel c offset size scale window complement).eraseDType.toAlgKernel
    (programState c s job.2.2.val job.1.val job.2.1.val)

def run (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    List (Job c) → BlockState → Option BlockState
  | [], s => some s
  | job::rest, s => (invoke c offset size scale window complement job s).bind
      (run c offset size scale window complement rest)

def InputsSame (initial current : BlockState) : Prop :=
  ∀ (r : RegionName) a, r≠"out" → current.mem r a=initial.mem r a

theorem slice_same (c : Shape) (initial current : BlockState) (h : InputsSame initial current)
    (z head : Nat) : inputs c current z head=inputs c initial z head := by
  have hq : ∀ a, current.mem "q" a=initial.mem "q" a := fun a => h _ a (by decide)
  have hk : ∀ a, current.mem "k" a=initial.mem "k" a := fun a => h _ a (by decide)
  have hm : ∀ a, current.mem "m" a=initial.mem "m" a := fun a => h _ a (by decide)
  simp [inputs,RealLoads.inputs,BlockState.readMem,hq,hk,hm]

def CorrectAt (c : Shape) (initial current : BlockState) (offset size : Int)
    (scale : ℝ) (window complement : Bool) (job : Job c) : Prop :=
  ∀ lane : Fin c.B, active c job lane → current.readMem "out" (address c job lane)=
    LogicalFormula.column (inputs c initial job.1.val job.2.1.val) scale c.N window complement offset size
      (job.2.2.val*c.B+lane.val)

theorem step (c : Shape) (initial current : BlockState) (same : InputsSame initial current)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (job : Job c) :
    ∃ t, invoke c offset size scale window complement job current=some t ∧
      InputsSame initial t ∧ CorrectAt c initial t offset size scale window complement job ∧
      (∀ old, old≠job → CorrectAt c initial current offset size scale window complement old →
        CorrectAt c initial t offset size scale window complement old) := by
  obtain ⟨t,ht,ho,hf⟩ := DenseSource.executes c current job.2.2.val job.1.val job.2.1
    offset size scale window complement
  refine ⟨t,ht,?_,?_,?_⟩
  · intro r a hr
    exact (hf r a (Or.inl hr)).trans (same r a hr)
  · intro lane hl
    have hv := ho lane hl
    rw [slice_same c initial current same] at hv
    exact hv
  · intro old hne hold lane hl
    have hcell := hf "out" (address c old lane) (Or.inr (by
      intro other hother heq
      exact hne (address_owner c old job lane other hl hother heq)))
    have hread : t.readMem "out" (address c old lane)=current.readMem "out" (address c old lane) := by
      unfold BlockState.readMem
      rw [hcell]
    exact hread.trans (hold lane hl)

/-- All programs actually execute. The induction keeps initial input cells and
every column already produced; repeated jobs are harmless because their values
depend only on preserved inputs. The public launch enumerates every job once. -/
theorem run_correct (c : Shape) (initial current : BlockState) (same : InputsSame initial current)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (order done : List (Job c))
    (correct : ∀ job∈done, CorrectAt c initial current offset size scale window complement job) :
    ∃ t, run c offset size scale window complement order current=some t ∧ InputsSame initial t ∧
      ∀ job∈done++order, CorrectAt c initial t offset size scale window complement job := by
  induction order generalizing current done with
  | nil => exact ⟨current,rfl,same,by simpa using correct⟩
  | cons job rest ih =>
      obtain ⟨next,hn,hs,hnew,hkeep⟩ := step c initial current same offset size scale window complement job
      have hc : ∀ old∈done++[job], CorrectAt c initial next offset size scale window complement old := by
        intro old ho
        by_cases he : old=job
        · simpa [he] using hnew
        · apply hkeep old he
          exact correct old (by simpa [he] using ho)
      obtain ⟨t,ht,hi,hout⟩ := ih next hs (done++[job]) hc
      refine ⟨t,?_,hi,?_⟩
      · simp only [run,hn,Option.bind_some]; exact ht
      · simpa only [List.append_assoc,List.singleton_append] using hout

theorem whole_grid (c : Shape) (initial : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) :
    ∃ t, run c offset size scale window complement (jobs c) initial=some t ∧
      InputsSame initial t ∧ ∀ job, CorrectAt c initial t offset size scale window complement job := by
  obtain ⟨t,ht,hs,ho⟩ := run_correct c initial initial (by intro _ _ _; rfl)
    offset size scale window complement (jobs c) [] (by simp)
  exact ⟨t,ht,hs,fun job => ho job (by simp [jobs])⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.WholeGrid

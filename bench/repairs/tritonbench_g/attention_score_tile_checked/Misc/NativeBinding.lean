import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIO

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.Binding
open VeriTile Triton FP.Structural Reused Geometry Physical NativeInstructions NativeStaging
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem full_row (N B row : Nat) (hB : 0<B) (aligned : row%B=0) (live : row<N)
    (even : N%B=0) (i : Fin B) : row+i.val<N := by
  have hr := Nat.mod_add_div row B
  have hn := Nat.mod_add_div N B
  rw [aligned] at hr
  rw [even] at hn
  simp only [Nat.zero_add] at hr hn
  have hquot : row/B<N/B := by nlinarith
  have hm := Nat.mul_le_mul_left B (Nat.succ_le_of_lt hquot)
  have hi := i.isLt
  nlinarith

theorem query_active (c : Shape) (row : Nat) (aligned : row%c.B=0) (live : row<c.N)
    (i : Fin c.B) :
    (decide (c.N%c.B=0) || decide (row+i.val<c.N))=Bool.true ↔ row+i.val<c.N := by
  simp only [Bool.or_eq_true,decide_eq_true_eq]
  exact ⟨fun h => h.elim (fun even => full_row c.N c.B row (supported_positive c.B c.block) aligned live even i) id,Or.inr⟩

theorem key_active (c : Shape) (block : Fin c.programs) (j : Fin c.B) :
    (decide (c.NK%c.B=0) || decide (block.val*c.B+j.val<c.NK))=Bool.true ↔
      block.val*c.B+j.val<c.NK := by
  simp only [Bool.or_eq_true,decide_eq_true_eq]
  exact ⟨fun h => h.elim
    (fun even => DenseSafety.full_row _ _ _ _ (supported_positive c.B c.block) block.isLt j.isLt even) id,Or.inr⟩

theorem read (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (i : Fin 4) (j : Fin (RealInterface.arity c i)) (active : i.val≠3 ∨ RealInterface.empty c) :
    lower ((s.mem (RealInterface.region i) j.val).read .real)=flatInputs c xs i j := by
  have bound := binding.toInputBinding
  fin_cases i
  all_goals
    have h := bound (⟨_,by change _ < 4; decide⟩) (j,()) active
    change (s.mem _ j.val).read .real=encode c _ at h
    exact (congrArg lower h).trans (lower_encode c _)

theorem bases (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (job : WholeGrid.Job c) :
    let p := parameters c offset size scale window complement
    let state := FPGrid.programState c s job
    FPContext.baseQ p state=c.queryBase job.1.val job.2.1.val ∧
      FPContext.baseK p state=c.keyBase job.1.val job.2.1.val ∧
      FPContext.baseM p state=c.normalizerBase job.1.val job.2.1.val := by
  obtain ⟨hz,hh⟩ := DenseSource.batch_head c job.1.val job.2.1
  simp [FPContext.baseQ,FPContext.baseK,FPContext.baseM,FPGrid.programState,parameters,
    Shape.queryBase,Shape.keyBase,Shape.normalizerBase,hz,hh,Nat.add_mul,Nat.mul_assoc]

theorem query (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (job : WholeGrid.Job c) (row : Nat) (aligned : row%c.B=0) (live : row<c.N)
    (i : Fin c.B) (d : Fin c.D) :
    NativeFormula.query c.B c.D (parameters c offset size scale window complement)
      (FPGrid.programState c s job) row (i,d,()) =
      NativeReference.query c (RealInterface.logicalInputs c (flatInputs c xs)) job.1 job.2.1 (row+i.val) d := by
  have hactive := query_active c row aligned live i
  have hbase := (bases c offset size scale window complement s job).1
  change (if (decide (c.N%c.B=0) || decide (row+i.val<c.N))=Bool.true then
    lower ((s.mem "q" (FPContext.baseQ (parameters c offset size scale window complement)
      (FPGrid.programState c s job)+(row+i.val)*c.D+d.val*1)).read .real)
    else NativeFormula.zero32)=_
  by_cases hi : row+i.val<c.N
  · simp only [if_pos (hactive.mpr hi),NativeReference.query,dif_pos hi]
    rw [hbase]
    simpa [FPGrid.programState,RealInterface.logicalInputs,Nat.mul_one,Nat.add_assoc] using
      read c offset size scale window complement s xs binding (0 : Fin 4)
        ⟨_,query_bound c job.1 job.2.1 ⟨row+i.val,hi⟩ d⟩ (Or.inl (by decide))
  · simp only [if_neg (fun h => hi (hactive.mp h)),NativeReference.query,dif_neg hi]

theorem key (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (job : WholeGrid.Job c) (d : Fin c.D) (j : Fin c.B) :
    NativeFormula.key c.B c.D (parameters c offset size scale window complement)
      (FPGrid.programState c s job) (d,j,()) =
      NativeReference.key c (RealInterface.logicalInputs c (flatInputs c xs)) job.1 job.2.1
        (job.2.2.val*c.B+j.val) d := by
  have hactive := key_active c job.2.2 j
  have hbase := (bases c offset size scale window complement s job).2.1
  change (if (decide (c.NK%c.B=0) || decide (job.2.2.val*c.B+j.val<c.NK))=Bool.true then
    lower ((s.mem "k" (FPContext.baseK (parameters c offset size scale window complement)
      (FPGrid.programState c s job)+d.val*1+(job.2.2.val*c.B+j.val)*c.D)).read .real)
    else NativeFormula.zero32)=_
  by_cases hj : job.2.2.val*c.B+j.val<c.NK
  · simp only [if_pos (hactive.mpr hj),NativeReference.key,dif_pos hj]
    rw [hbase]
    simpa [FPGrid.programState,RealInterface.logicalInputs,Shape.keyBase,
      Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      read c offset size scale window complement s xs binding (1 : Fin 4)
        ⟨_,RealInterface.key_all_heads_bound c job.1
          ⟨job.2.1.val/(c.H/c.HK),grouped_head c job.2.1⟩ ⟨job.2.2.val*c.B+j.val,hj⟩ d⟩ (Or.inl (by decide))
  · simp only [if_neg (fun h => hj (hactive.mp h)),NativeReference.key,dif_neg hj]

theorem normalizer (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (job : WholeGrid.Job c) (row : Nat) (i : Fin c.B) :
    NativeFormula.normalizer c.B (parameters c offset size scale window complement)
      (FPGrid.programState c s job) row i =
      NativeReference.normalizer c (RealInterface.logicalInputs c (flatInputs c xs)) job.1 job.2.1 (row+i.val) := by
  have hbase := (bases c offset size scale window complement s job).2.2
  change (if row+i.val<c.N then
    lower ((s.mem "m" (FPContext.baseM (parameters c offset size scale window complement)
      (FPGrid.programState c s job)+row+i.val)).read .real)
    else NativeFormula.zero32)=_
  unfold NativeReference.normalizer
  split_ifs with hi
  · rw [hbase]
    simpa [FPGrid.programState,RealInterface.logicalInputs,Shape.normalizerBase,Nat.add_assoc] using
      read c offset size scale window complement s xs binding (2 : Fin 4)
        ⟨_,normalizer_bound c job.1 job.2.1 ⟨row+i.val,hi⟩⟩ (Or.inl (by decide))
  · rfl

theorem update (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (job : WholeGrid.Job c) (row : Nat) (aligned : row%c.B=0) (live : row<c.N) :
    NativeFormula.update c.B c.D (parameters c offset size scale window complement)
      (FPGrid.programState c s job) row =
      NativeReference.update c (RealInterface.logicalInputs c (flatInputs c xs)) job.1 job.2.1
        offset size scale window complement job.2.2.val row := by
  have hp (i j : Fin c.B) :
      NativeFormula.probability c.B c.D (parameters c offset size scale window complement)
        (FPGrid.programState c s job) row (i,j,()) =
        NativeReference.probability c (RealInterface.logicalInputs c (flatInputs c xs)) job.1 job.2.1
          offset size scale window complement job.2.2.val row i j := by
    simp only [NativeFormula.probability,NativeFormula.raw]
    simp only [query c offset size scale window complement s xs binding job row aligned live,
      key c offset size scale window complement s xs binding job,
      normalizer c offset size scale window complement s xs binding job,
      NativeReference.probability,NativeReference.raw,NativeFormula.scale]
    simp only [parameters,FPGrid.programState]
    simp only [ite_true]
    by_cases hi : row+i.val<c.N
    · simp only [if_pos hi]
      split
      · rfl
      · rename_i hn
        exact False.elim (hn ((query_active c row aligned live i).mpr hi))
    · simp only [if_neg hi]
      split
      · rename_i hp
        exact False.elim (hi ((query_active c row aligned live i).mp hp))
      · rfl
  funext acc col
  simp only [NativeFormula.update,NativeFormula.contribution,NativeReference.update,
    NativeReference.contribution,hp]

theorem fold_binding (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (job : WholeGrid.Job c) (row : Nat) (aligned : row%c.B=0) (acc : Fin c.B → Graph) :
    foldRange (NativeFormula.update c.B c.D (parameters c offset size scale window complement)
      (FPGrid.programState c s job)) row c.N c.B acc =
      foldRange (NativeReference.update c (RealInterface.logicalInputs c (flatInputs c xs))
        job.1 job.2.1 offset size scale window complement job.2.2.val) row c.N c.B acc := by
  have hB := Nat.ne_of_gt (supported_positive c.B c.block)
  by_cases live : row<c.N
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [hB,if_false,live,if_true]
    rw [update c offset size scale window complement s xs binding job row aligned live]
    exact fold_binding c offset size scale window complement s xs binding job (row+c.B)
      (by simpa using aligned) _
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [dif_neg hB,dif_neg live]
termination_by c.N-row
decreasing_by have hb := supported_positive c.B c.block; omega

theorem returned (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : State NativeStaging.Expr) (xs : Inputs c)
    (binding : (programIO c offset size scale window complement).InputCells xs s)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    NativeLoop.returnedGraph c s offset size scale window complement z h j =
      NativeReference.column c (RealInterface.logicalInputs c (flatInputs c xs)) z h
        offset size scale window complement j := by
  unfold NativeLoop.returnedGraph NativeFormula.accumulated NativeReference.column
  exact congrFun (fold_binding c offset size scale window complement s xs binding
    (FPReturned.ownerJob c z h j) 0 (by simp) (fun _ => NativeFormula.zero32)) _

#axiomsClean returned
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.Binding

import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedMemory

namespace VeriTile.Bench.Repairs.LayerNormLiger.Ordered
open VeriTile.Triton FP.Structural LayerIO
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def BMatch (initial : State α) (a : Backward.Acc α p) (z : Carries α p) : Prop :=
  Match initial a.state z.writes ∧ a.dw=z.dw ∧ a.db=z.db

private theorem b_loads (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (s : State α)
    (hin : (backwardIO M R p pid).InputBinding v s) (row : Nat) (hr : rowMask p pid row)
    (a : Backward.Acc α p) (z : Carries α p)
    (h : Match ((bInvocation M R p pid).initial s) a.state z.writes) :
    let base := bInitial p pid v
    Backward.load M p a p.x (row*p.xStride)=bLoad M p base z p.x (row*p.xStride) ∧
    Backward.load M p a p.weight 0=bLoad M p base z p.weight 0 ∧
    Backward.load M p a p.dy (row*p.dyStride)=bLoad M p base z p.dy (row*p.dyStride) ∧
    Backward.mean p a row=bMean p base z row ∧ Backward.rstd p a row=bRstd p base z row := by
  have rowBound : row<p.rows := lt_of_lt_of_le hr.2 (Nat.min_le_right _ _)
  refine ⟨?_,?_,?_,?_,?_⟩
  · funext i
    by_cases hi : i.1.val<p.columns
    · simp only [Backward.load,bLoad,if_pos hi]
      apply match_read h
      simpa only [Invocation.initial_mem] using b_initial_eq M R p pid v s hin p.x
        (row*p.xStride+i.1.val) ⟨0,(⟨row,rowBound⟩,i)⟩ ⟨⟨hr,hi⟩,rfl,rfl⟩
    · simp [Backward.load,bLoad,hi]
  · funext i
    by_cases hi : i.1.val<p.columns
    · simp only [Backward.load,bLoad,if_pos hi]
      apply match_read h
      simpa only [Invocation.initial_mem,Nat.zero_add] using b_initial_eq M R p pid v s hin p.weight
        i.1.val ⟨1,i⟩ ⟨⟨lt_of_le_of_lt hr.1 hr.2,hi⟩,rfl,rfl⟩
    · simp [Backward.load,bLoad,hi]
  · funext i
    by_cases hi : i.1.val<p.columns
    · simp only [Backward.load,bLoad,if_pos hi]
      apply match_read h
      simpa only [Invocation.initial_mem] using b_initial_eq M R p pid v s hin p.dy
        (row*p.dyStride+i.1.val) ⟨2,(⟨row,rowBound⟩,i)⟩ ⟨⟨hr,hi⟩,rfl,rfl⟩
    · simp [Backward.load,bLoad,hi]
  · apply match_read h
    simpa only [Invocation.initial_mem] using b_initial_eq M R p pid v s hin p.mean row
      ⟨3,(⟨row,rowBound⟩,PUnit.unit)⟩ ⟨hr,rfl,rfl⟩
  · apply match_read h
    simpa only [Invocation.initial_mem] using b_initial_eq M R p pid v s hin p.rstd row
      ⟨4,(⟨row,rowBound⟩,PUnit.unit)⟩ ⟨hr,rfl,rfl⟩

private theorem b_step_match (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (s : State α)
    (hin : (backwardIO M R p pid).InputBinding v s) (row : Nat) (hr : rowMask p pid row)
    (a : Backward.Acc α p) (z : Carries α p)
    (h : BMatch ((bInvocation M R p pid).initial s) a z) :
    BMatch ((bInvocation M R p pid).initial s) (Backward.update M R p row a)
      (bStep M R p (bInitial p pid v) row z) := by
  obtain ⟨hx,hw,hdy,hm,hrstd⟩ := b_loads M R p pid v s hin row hr a z h.1
  let base := bInitial p pid v
  have xhat : Backward.xhat M p a row=bXhat M p base z row := by
    funext i
    simp only [Backward.xhat,bXhat,hx,hm,hrstd] <;> rfl
  have wdy : Backward.wdy M p a row=bWdy M p base z row := by
    funext i
    simp only [Backward.wdy,bWdy,hw,hdy] <;> rfl
  have c1 : Backward.c1 M p a row=bC1 M p base z row := by
    simp only [Backward.c1,bC1,xhat,wdy]
  have c2 : Backward.c2 M R p a row=bC2 M R p base z row := by
    simp only [Backward.c2,bC2,wdy]
  have dx : Backward.dxValue M R p a row=bDX M R p base z row := by
    funext i
    simp only [Backward.dxValue,bDX,wdy,xhat,c1,c2,hrstd] <;> rfl
  refine ⟨?_,?_,?_⟩
  · change Match _ ((TileShape.allIndices [p.block]).foldl _ a.state) _
    simp only [Backward.stored,dx]
    exact match_scatter h.1 _ _ _ _
  · funext i
    simp only [Backward.update,bStep,h.2.1,hdy,xhat] <;> rfl
  · funext i
    simp only [Backward.update,bStep,h.2.2,hdy] <;> rfl

private theorem b_range_match (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (s : State α)
    (hin : (backwardIO M R p pid).InputBinding v s) (cur : Nat) (lo : first p pid≤cur)
    (a : Backward.Acc α p) (z : Carries α p)
    (h : BMatch ((bInvocation M R p pid).initial s) a z) :
    BMatch ((bInvocation M R p pid).initial s)
      (foldRange (Backward.update M R p) cur (last p pid) 1 a)
      (foldRange (bStep M R p (bInitial p pid v)) cur (last p pid) 1 z) := by
  by_cases hi : cur<last p pid
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hi,if_false,if_true,dite_false,dite_true]
    exact b_range_match M R p pid v s hin (cur+1) (by omega) _ _
      (b_step_match M R p pid v s hin cur ⟨lo,hi⟩ a z h)
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simpa only [Nat.one_ne_zero,hi,if_false,dite_false] using h
termination_by last p pid-cur
decreasing_by omega

theorem backward_match (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (s : State α)
    (hin : (backwardIO M R p pid).InputBinding v s) :
    let initial := (bInvocation M R p pid).initial s
    Match initial (Backward.reference M R p initial) (backwardWrites M R p pid v) := by
  let initial := (bInvocation M R p pid).initial s
  have loop := b_range_match M R p pid v s hin (first p pid) (by rfl)
    (Backward.initial M p initial) (bZero M p) ⟨match_initial _,rfl,rfl⟩
  change BMatch initial (Backward.accumulated M R p initial) (bAccumulated M R p pid v) at loop
  dsimp only
  unfold Backward.reference Backward.finish backwardWrites bFinish
  simp only [initial,b_pid,Backward.stored,loop.2.1,loop.2.2]
  exact match_scatter (match_scatter loop.1 _ _ _ _) _ _ _ _

theorem b_range_preserves (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (base : RegionName → Nat → α) (cur stop : Nat) (a : Carries α p)
    (q : RegionName) (i : Nat) (h : Present a.writes q i) :
    Present (foldRange (bStep M R p base) cur stop 1 a).writes q i := by
  apply Folds.preserves (bStep M R p base) (fun a => Present a.writes q i) _ cur stop a h
  intro row z hz
  exact scatter_preserves _ _ _ _ _ _ _ hz

theorem b_range_present (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (base : RegionName → Nat → α) (cur stop row : Nat) (a : Carries α p)
    (i : TileIndex [p.block]) (lo : cur≤row) (hi : row<stop) (hc : i.1.val<p.columns) :
    Present (foldRange (bStep M R p base) cur stop 1 a).writes p.dx (row*p.dxStride+i.1.val) := by
  have hc' : cur<stop := by omega
  rw [foldRange]
  simp only [Nat.one_ne_zero,hc',if_false,if_true,dite_false,dite_true]
  by_cases eq : cur=row
  · subst cur
    exact b_range_preserves M R p base (row+1) stop _ _ _ (scatter_present _ _ _ _ _ i hc)
  · exact b_range_present M R p base (cur+1) stop row _ i (by omega) hi hc
termination_by stop-cur
decreasing_by omega

theorem backward_present (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (o : Fin 3)
    (i : TileIndex ((bOutputs p)[o]).shape) (active : (bMemory α p pid).outputMask o v i) :
    Present (backwardWrites M R p pid v) (bOutputs p)[o].buf ((bMemory α p pid).write o v i) := by
  fin_cases o
  · apply scatter_preserves
    apply scatter_preserves
    exact b_range_present M R p _ _ _ i.1.val _ i.2 active.1.1 active.1.2 active.2
  · apply scatter_preserves
    exact scatter_present _ _ _ _ _ i active
  · exact scatter_present _ _ _ _ _ i active

theorem backward_implements (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat) :
    (backwardIO M R p pid).Implements (backward M R p pid) := by
  apply ProgramIO.Implements.intro
  · intro v bounds hb s _
    exact backward_safe _ _ _ _ _ _ hb s
  · intro v s hin
    let initial := (bInvocation M R p pid).initial s
    obtain ⟨t,ran,memory⟩ := Backward.complete_run M R p initial
    have matched := backward_match M R p pid v s hin
    refine ⟨t,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans ran
    · intro o i active
      have cell := match_output matched (bOutputs p)[o].buf ((bMemory α p pid).write o v i)
        (backward_present M R p pid v o i active)
      fin_cases o
      all_goals
        change mapCell id (t.mem _ _) = _
        rw [memory]
        simpa only [backwardIO,bMemory,bOutputs,backward,mapCell,mapValue,initial] using congrArg (mapCell id) cell
    · intro r o outside
      rw [memory]
      exact (Backward.reference_frame _ _ _ _ r o
        (backward_outside _ _ _ _ _ _ _ _ outside)).trans
          (congrFun (congrFun (Invocation.initial_mem _ _) r) o)

#axiomsClean backward_implements
end
end VeriTile.Bench.Repairs.LayerNormLiger.Ordered

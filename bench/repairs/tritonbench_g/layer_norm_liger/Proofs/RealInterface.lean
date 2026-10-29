import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.IOBindings
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardResult
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalInputs

namespace VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

private theorem f_inputs (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat)
    (v : ChannelValues ℝ (fInputs p)) (s : State ℝ)
    (hin : (forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).InputBinding v s)
    (hb : p.columns≤p.block) :
    let t := (fInvocation (Logical.algebra rest) Logical.sumPrimitive p pid).initial s
    (∀ j,j<p.columns → Logical.rowX p t j=fAt p v 0 j) ∧
    (∀ j,j<p.columns → Logical.weight p t j=fAt p v 1 j) ∧
    (∀ j,j<p.columns → Logical.bias p t j=fAt p v 2 j) := by
  refine ⟨?_,?_,?_⟩
  · intro j hj
    have bound := lt_of_lt_of_le hj hb
    have cell := hin (0 : Fin 3) (⟨j,bound⟩,PUnit.unit) hj
    simpa [Logical.rowX,fAt,fInput,bound,forwardIO,fMemory,fInputs,mapValue] using cell
  · intro j hj
    have bound := lt_of_lt_of_le hj hb
    have cell := hin (1 : Fin 3) (⟨j,bound⟩,PUnit.unit) hj
    simpa [Logical.weight,fAt,fInput,bound,forwardIO,fMemory,fInputs,mapValue] using cell
  · intro j hj
    have bound := lt_of_lt_of_le hj hb
    have cell := hin (2 : Fin 3) (⟨j,bound⟩,PUnit.unit) hj
    simpa [Logical.bias,fAt,fInput,bound,forwardIO,fMemory,fInputs,mapValue] using cell

theorem forward_implements (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat)
    (hb : p.columns≤p.block) (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y) :
    (forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements (forwardFormula p) := by
  apply ProgramIO.Implements.intro
  · intro v bounds h s _
    exact forward_safe _ _ _ _ _ _ h s
  · intro v s hin
    let initial := (fInvocation (Logical.algebra rest) Logical.sumPrimitive p pid).initial s
    obtain ⟨t,ran,mean,rstd,y,frame⟩ := Logical.forward_all_exec rest p initial hb hmy hmr hry
    obtain ⟨hx,hw,hb⟩ := f_inputs rest p pid v s hin hb
    refine ⟨t,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans ran
    · intro o i active
      fin_cases o
      · change mapCell id (t.mem p.y (pid*p.yStride+i.1.val)) = .mk .fp32 _
        have cell := y i active
        have formula := LogicalInputs.y_congr p.columns i.1.val p.epsilon
          (Logical.rowX p initial) (Logical.weight p initial) (Logical.bias p initial)
          (fAt p v 0) (fAt p v 1) (fAt p v 2) active hx hw hb
        simp only [Forward.outputOffset,initial,f_pid] at cell
        rw [cell,formula]
        rfl
      · change mapCell id (t.mem p.mean (pid*p.meanStride)) = .mk .fp32 _
        simp only [Forward.meanOffset,initial,f_pid] at mean
        rw [mean,LogicalInputs.mean_congr p.columns _ _ hx]
        rfl
      · change mapCell id (t.mem p.rstd (pid*p.rstdStride)) = .mk .fp32 _
        simp only [Forward.rstdOffset,initial,f_pid] at rstd
        rw [rstd,LogicalInputs.rstd_congr p.columns p.epsilon _ _ hx]
        rfl
    · intro r o outside
      exact (frame r o (forward_outside _ _ _ _ _ _ _ _ outside)).trans (congrFun (congrFun (Invocation.initial_mem _ _) r) o)

private theorem b_inputs (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat)
    (v : ChannelValues ℝ (bInputs p)) (s : State ℝ)
    (hin : (backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).InputBinding v s)
    (hb : p.columns≤p.block) (row : Nat) (hr : rowMask p pid row) :
    let initial := (bInvocation (Logical.algebra rest) Logical.sumPrimitive p pid).initial s
    (∀ j,j<p.columns → Logical.bx p (Logical.saved p initial) row j=bx p v row j) ∧
    (∀ j,j<p.columns → Logical.bw p (Logical.saved p initial) j=bw p v j) ∧
    (∀ j,j<p.columns → Logical.bdy p (Logical.saved p initial) row j=bdy p v row j) ∧
    Backward.mean p (Logical.saved p initial) row=bmean p v row ∧
    Backward.rstd p (Logical.saved p initial) row=brstd p v row := by
  have rowBound : row<p.rows := lt_of_lt_of_le hr.2 (Nat.min_le_right _ _)
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro j hj
    have colBound := lt_of_lt_of_le hj hb
    have cell := hin (0 : Fin 5) (⟨row,rowBound⟩,⟨j,colBound⟩,PUnit.unit) ⟨hr,hj⟩
    simpa [Logical.bx,Logical.saved,bx,rowBound,colBound,backwardIO,bMemory,bInputs,mapValue] using cell
  · intro j hj
    have colBound := lt_of_lt_of_le hj hb
    have cell := hin (1 : Fin 5) (⟨j,colBound⟩,PUnit.unit) ⟨lt_of_le_of_lt hr.1 hr.2,hj⟩
    simpa [Logical.bw,Logical.saved,bw,colBound,backwardIO,bMemory,bInputs,mapValue] using cell
  · intro j hj
    have colBound := lt_of_lt_of_le hj hb
    have cell := hin (2 : Fin 5) (⟨row,rowBound⟩,⟨j,colBound⟩,PUnit.unit) ⟨hr,hj⟩
    simpa [Logical.bdy,Logical.saved,bdy,rowBound,colBound,backwardIO,bMemory,bInputs,mapValue] using cell
  · have cell := hin (3 : Fin 5) (⟨row,rowBound⟩,PUnit.unit) hr
    simpa [Backward.mean,Logical.saved,bmean,rowBound,backwardIO,bMemory,bInputs,mapValue] using cell
  · have cell := hin (4 : Fin 5) (⟨row,rowBound⟩,PUnit.unit) hr
    simpa [Backward.rstd,Logical.saved,brstd,rowBound,backwardIO,bMemory,bInputs,mapValue] using cell

private theorem dx_congr (n i : Nat) (x w dy z v gy : Nat → ℝ) (mu inv : ℝ)
    (hi : i<n) (hx : ∀ j,j<n → x j=z j) (hw : ∀ j,j<n → w j=v j)
    (hdy : ∀ j,j<n → dy j=gy j) : Logical.dx n x w dy mu inv i=Logical.dx n z v gy mu inv i := by
  have a : Logical.sum n (fun j => Logical.xhat x mu inv j*(w j*dy j))=
      Logical.sum n (fun j => Logical.xhat z mu inv j*(v j*gy j)) := by
    apply Finset.sum_congr rfl
    intro j hj
    simp only [Finset.mem_range] at hj
    simp only [Logical.xhat,hx j hj,hw j hj,hdy j hj]
  have b : Logical.sum n (fun j => w j*dy j)=Logical.sum n (fun j => v j*gy j) := by
    apply Finset.sum_congr rfl
    intro j hj
    simp only [Finset.mem_range] at hj
    simpa only [hw j hj,hdy j hj]
  unfold Logical.dx
  rw [a,b,hw i hi,hdy i hi]
  simp only [Logical.xhat,hx i hi]

theorem backward_implements (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat)
    (fresh : Logical.FreshDX p) (hwd : p.dw≠p.db) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride) :
    (backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements (backwardFormula p pid) := by
  apply ProgramIO.Implements.intro
  · intro v bounds h s _
    exact backward_safe _ _ _ _ _ _ h s
  · intro v s hin
    let initial := (bInvocation (Logical.algebra rest) Logical.sumPrimitive p pid).initial s
    have start : Backward.start p initial=first p pid := by simp [initial,Backward.start,first]
    have stop : Backward.stop p initial=last p pid := by simp [initial,Backward.stop,last]
    obtain ⟨t,ran,memory⟩ := Backward.complete_run (Logical.algebra rest) Logical.sumPrimitive p initial
    refine ⟨t,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans ran
    · intro o i active
      fin_cases o
      · change mapCell id (t.mem p.dx (i.1.val*p.dxStride+i.2.1.val)) = .mk .fp32 _
        have cell := Logical.reference_dx rest p initial fresh hxd hxb hb hs i.1.val
          (by simpa [start] using active.1.1) (by simpa [stop] using active.1.2) i.2 active.2
        rw [memory,cell]
        obtain ⟨hx,hw,hdy,hm,hr⟩ := b_inputs rest p pid v s hin hb i.1.val active.1
        change Cell.mk .fp32 (Logical.rowDX p initial i.1.val i.2.1.val)=_
        unfold Logical.rowDX
        rw [hm,hr,dx_congr p.columns i.2.1.val _ _ _ _ _ _ _ _ active.2 hx hw hdy]
        rfl
      · change mapCell id (t.mem p.dw (pid*p.dwStride+i.1.val)) = .mk .fp32 _
        have cell := Logical.reference_dw rest p initial fresh hwd i active
        simp only [initial,b_pid] at cell
        rw [memory,cell]
        change Cell.mk .fp32 (Logical.groupDW p initial i.1.val)=.mk .fp32 (dwFormula p pid v i.1.val)
        congr 1
        unfold Logical.groupDW dwFormula
        rw [start,stop]
        apply Finset.sum_congr rfl
        intro row hrow
        obtain ⟨hx,_,hdy,hm,hr⟩ := b_inputs rest p pid v s hin hb row (Finset.mem_Ico.mp hrow)
        rw [hdy _ active,hm,hr]
        unfold Logical.xhat
        rw [hx _ active]
      · change mapCell id (t.mem p.db (pid*p.dbStride+i.1.val)) = .mk .fp32 _
        have cell := Logical.reference_db rest p initial fresh i active
        simp only [initial,b_pid] at cell
        rw [memory,cell]
        change Cell.mk .fp32 (Logical.groupDB p initial i.1.val)=.mk .fp32 (dbFormula p pid v i.1.val)
        congr 1
        unfold Logical.groupDB dbFormula
        rw [start,stop]
        apply Finset.sum_congr rfl
        intro row hrow
        exact (b_inputs rest p pid v s hin hb row (Finset.mem_Ico.mp hrow)).2.2.1 _ active
    · intro r o outside
      rw [memory]
      exact (Backward.reference_frame _ _ _ _ r o
        (backward_outside _ _ _ _ _ _ _ _ outside)).trans
          (congrFun (congrFun (Invocation.initial_mem _ _) r) o)

#axiomsClean forward_implements
#axiomsClean backward_implements
end
end VeriTile.Bench.Repairs.LayerNormLiger.LayerIO

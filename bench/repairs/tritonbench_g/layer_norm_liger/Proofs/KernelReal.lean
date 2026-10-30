import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex

namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
noncomputable section
set_option maxHeartbeats 4000000

 theorem mean_congr (n : Nat) (x z : Nat → ℝ) (h : ∀ i, i < n → x i = z i) :
    Logical.mean n x = Logical.mean n z := by
  unfold Logical.mean Logical.sum
  congr 1
  exact Finset.sum_congr rfl (fun i hi => h i (Finset.mem_range.mp hi))

 theorem rstd_congr (n : Nat) (eps : ℝ) (x z : Nat → ℝ) (h : ∀ i, i < n → x i = z i) :
    Logical.rstd n eps x = Logical.rstd n eps z := by
  have mean := mean_congr n x z h
  have variance : Logical.variance n x = Logical.variance n z := by
    unfold Logical.variance Logical.sum
    rw [mean]
    congr 1
    exact Finset.sum_congr rfl (fun i hi => by dsimp only; rw [h i (Finset.mem_range.mp hi)])
  rw [Logical.rstd,Logical.rstd,variance]

 theorem y_congr (n i : Nat) (eps : ℝ) (x w b z v c : Nat → ℝ) (hi : i < n)
    (hx : ∀ j, j < n → x j = z j) (hw : ∀ j, j < n → w j = v j) (hb : ∀ j, j < n → b j = c j) :
    Logical.y n eps x w b i = Logical.y n eps z v c i := by
  unfold Logical.y
  rw [hx i hi,hw i hi,hb i hi,mean_congr n x z hx,rstd_congr n eps x z hx]

 theorem dx_congr (n i : Nat) (x w dy z v dz : Nat → ℝ) (mu inv : ℝ) (hi : i < n)
    (hx : ∀ j, j < n → x j = z j) (hw : ∀ j, j < n → w j = v j)
    (hd : ∀ j, j < n → dy j = dz j) :
    Logical.dx n x w dy mu inv i = Logical.dx n z v dz mu inv i := by
  have a : Logical.sum n (fun j => Logical.xhat x mu inv j*(w j*dy j)) =
      Logical.sum n (fun j => Logical.xhat z mu inv j*(v j*dz j)) := by
    apply Finset.sum_congr rfl
    intro j hj
    simp only [Logical.xhat,hx j (Finset.mem_range.mp hj),hw j (Finset.mem_range.mp hj),hd j (Finset.mem_range.mp hj)]
  have b : Logical.sum n (fun j => w j*dy j) = Logical.sum n (fun j => v j*dz j) := by
    apply Finset.sum_congr rfl
    intro j hj
    simp only [hw j (Finset.mem_range.mp hj),hd j (Finset.mem_range.mp hj)]
  unfold Logical.dx
  rw [a,b]
  simp only [Logical.xhat,hx i hi,hw i hi,hd i hi]

 theorem forward_inputs (c : ForwardCall) (h : ForwardDomain c pointers) (memory : Nat → ℝ) :
    (∀ j, j < c.columns → Logical.rowX c.parameters (entry pointers memory c.pid) j =
      c.x.logical memory (c.pid*c.columns+j)) ∧
    (∀ j, j < c.columns → Logical.weight c.parameters (entry pointers memory c.pid) j = c.weight.logical memory j) ∧
    (∀ j, j < c.columns → Logical.bias c.parameters (entry pointers memory c.pid) j = c.bias.logical memory j) := by
  have bx := h.bound c.x (by simp [ForwardCall.buffers,ForwardCall.inputs])
  have bw := h.bound c.weight (by simp [ForwardCall.buffers,ForwardCall.inputs])
  have bb := h.bound c.bias (by simp [ForwardCall.buffers,ForwardCall.inputs])
  refine ⟨?_,?_,?_⟩
  · intro j hj
    have stride := active_row_stride h.x h.xStride h.columns_pos c.pid h.pid_bound
    change ((entry pointers memory c.pid).mem c.x.region (c.pid*c.x.stride0+j)).read .fp32 = _
    rw [stride]
    exact entry_read h.x bx memory c.pid _ (input_bound h.x _ _ h.pid_bound hj)
  · intro j hj
    exact entry_read h.weight bw memory c.pid _ (vector_bound h.weight j hj)
  · intro j hj
    exact entry_read h.bias bb memory c.pid _ (vector_bound h.bias j hj)

/-- Population statistics and every affine result use the actual logical
Tensor values. The three outputs are read through their own storage offsets. -/
def ForwardReal (c : ForwardCall) (initial final : Nat → ℝ) : Prop :=
  c.mean.logical final c.pid = Logical.mean c.columns (fun j => c.x.logical initial (c.pid*c.columns+j)) ∧
  c.rstd.logical final c.pid = Logical.rstd c.columns c.epsilon (fun j => c.x.logical initial (c.pid*c.columns+j)) ∧
  (∀ j, j < c.columns → c.y.logical final (c.pid*c.columns+j) =
    Logical.y c.columns c.epsilon (fun k => c.x.logical initial (c.pid*c.columns+k))
      (c.weight.logical initial) (c.bias.logical initial) j)

 theorem forward_real (rest : Algebra ℝ) (c : ForwardCall) (h : ForwardDomain c pointers)
    (memory : Nat → ℝ) :
    ∃ t, exec (withSum (Logical.algebra rest) Logical.sumPrimitive) (Forward.kernel c.parameters)
        (entry pointers memory c.pid) = some t ∧
      ForwardReal c memory (commit pointers c.writes memory t) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers memory c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) → commit pointers c.writes memory t a = memory a) := by
  obtain ⟨t,run,mu,inv,y,frame⟩ := Logical.forward_all_exec rest c.parameters (entry pointers memory c.pid)
    h.block_covers h.distinct.1 h.distinct.2.1 h.distinct.2.2
  simp only [Forward.meanOffset,Forward.rstdOffset,ForwardCall.parameters,entry,h.meanStride,h.rstdStride,Nat.mul_one] at mu inv
  obtain ⟨hx,hw,hb⟩ := forward_inputs c h memory
  have bm := h.bound c.mean (by simp [ForwardCall.buffers])
  have br := h.bound c.rstd (by simp [ForwardCall.buffers])
  have by_ := h.bound c.y (by simp [ForwardCall.buffers])
  refine ⟨t,run,⟨?_,?_,?_⟩,?_,?_⟩
  · rw [logical_read h.mean bm _ c.pid (vector_bound h.mean _ h.pid_bound)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inl rfl)]
    rw [mu]
    exact mean_congr _ _ _ hx
  · rw [logical_read h.rstd br _ c.pid (vector_bound h.rstd _ h.pid_bound)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inr (Or.inl rfl))]
    rw [inv]
    exact rstd_congr _ _ _ _ hx
  · intro j hj
    rw [logical_read h.y by_ _ _ (input_bound h.y _ _ h.pid_bound hj)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inr (Or.inr ⟨j,hj,rfl⟩))]
    have jy := y (⟨j,lt_of_lt_of_le hj h.block_covers⟩,PUnit.unit) hj
    have jy : t.mem c.y.region (c.pid*c.columns+j) = .mk .fp32
        (Logical.y c.columns c.epsilon (Logical.rowX c.parameters (entry pointers memory c.pid))
          (Logical.weight c.parameters (entry pointers memory c.pid))
          (Logical.bias c.parameters (entry pointers memory c.pid)) j) := by
      simpa only [Forward.outputOffset,ForwardCall.parameters,entry,h.yStride] using jy
    rw [jy]
    exact y_congr _ _ _ _ _ _ _ _ _ hj hx hw hb
  · intro l hl
    exact frame _ _ (forward_outside c h memory l hl)
  · exact commit_frame _ _ _ _

/-- Complete backward specification for this source program. DW and DB are
its actual partial rows; the subsequent Torch reduction is outside the kernel. -/
def BackwardReal (c : BackwardCall) (initial final : Nat → ℝ) : Prop :=
  (∀ col, col < c.columns →
    c.dw.logical final (c.pid*c.columns+col) =
      ∑ row ∈ Finset.Ico c.first c.last,
        c.dy.logical initial (row*c.columns+col) *
          Logical.xhat (fun j => c.x.logical initial (row*c.columns+j))
            (c.mean.logical initial row) (c.rstd.logical initial row) col) ∧
  (∀ col, col < c.columns → c.db.logical final (c.pid*c.columns+col) =
    ∑ row ∈ Finset.Ico c.first c.last, c.dy.logical initial (row*c.columns+col)) ∧
  (∀ row, c.first ≤ row → row < c.last → ∀ col, col < c.columns →
    c.dx.logical final (row*c.columns+col) =
      Logical.dx c.columns (fun j => c.x.logical initial (row*c.columns+j))
        (c.weight.logical initial) (fun j => c.dy.logical initial (row*c.columns+j))
        (c.mean.logical initial row) (c.rstd.logical initial row) col)

 theorem backward_inputs (c : BackwardCall) (h : BackwardDomain c pointers) (memory : Nat → ℝ)
    (row : Nat) (hr : row < c.rows) :
    (∀ j, j < c.columns → Logical.bx c.parameters (Logical.saved c.parameters (entry pointers memory c.pid)) row j =
      c.x.logical memory (row*c.columns+j)) ∧
    (∀ j, j < c.columns → Logical.bw c.parameters (Logical.saved c.parameters (entry pointers memory c.pid)) j =
      c.weight.logical memory j) ∧
    (∀ j, j < c.columns → Logical.bdy c.parameters (Logical.saved c.parameters (entry pointers memory c.pid)) row j =
      c.dy.logical memory (row*c.columns+j)) ∧
    Backward.mean c.parameters (Logical.saved c.parameters (entry pointers memory c.pid)) row = c.mean.logical memory row ∧
    Backward.rstd c.parameters (Logical.saved c.parameters (entry pointers memory c.pid)) row = c.rstd.logical memory row := by
  have binding (b : Buffer) (hb : b ∈ c.inputs) := h.bound b (List.mem_append_left _ hb)
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro j hj
    have stride := active_row_stride h.x h.xStride h.columns_pos row hr
    change ((entry pointers memory c.pid).mem c.x.region (row*c.x.stride0+j)).read .fp32 = _
    rw [stride]
    exact entry_read h.x (binding _ (by simp [BackwardCall.inputs])) memory c.pid _ (input_bound h.x _ _ hr hj)
  · intro j hj
    exact entry_read h.weight (binding _ (by simp [BackwardCall.inputs])) memory c.pid _ (vector_bound h.weight _ hj)
  · intro j hj
    have stride := active_row_stride h.dy h.dyStride h.columns_pos row hr
    change ((entry pointers memory c.pid).mem c.dy.region (row*c.dy.stride0+j)).read .fp32 = _
    rw [stride]
    exact entry_read h.dy (binding _ (by simp [BackwardCall.inputs])) memory c.pid _ (input_bound h.dy _ _ hr hj)
  · exact entry_read h.mean (binding _ (by simp [BackwardCall.inputs])) memory c.pid _ (vector_bound h.mean _ hr)
  · exact entry_read h.rstd (binding _ (by simp [BackwardCall.inputs])) memory c.pid _ (vector_bound h.rstd _ hr)

 theorem backward_fresh (c : BackwardCall) (h : BackwardDomain c pointers) : Logical.FreshDX c.parameters := by
  intro r hr
  simp only [Logical.Reads,BackwardCall.parameters] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl
  all_goals exact h.fresh _ (by simp [BackwardCall.inputs])

 theorem backward_real (rest : Algebra ℝ) (c : BackwardCall) (h : BackwardDomain c pointers)
    (memory : Nat → ℝ) :
    ∃ t, exec (withSum (Logical.algebra rest) Logical.sumPrimitive) (Backward.kernel c.parameters)
        (entry pointers memory c.pid) = some t ∧
      BackwardReal c memory (commit pointers c.writes memory t) ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers memory c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) → commit pointers c.writes memory t a = memory a) := by
  obtain ⟨t,run,mem⟩ := Backward.complete_run (Logical.algebra rest) Logical.sumPrimitive c.parameters (entry pointers memory c.pid)
  have fresh := backward_fresh c h
  have bind (b : Buffer) (hb : b ∈ c.buffers) := h.bound b hb
  refine ⟨t,run,⟨?_,?_,?_⟩,?_,?_⟩
  · intro col hc
    rw [logical_read h.dw (bind _ (by simp [BackwardCall.buffers])) _ _ (input_bound h.dw _ _ h.pid_bound hc)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inl ⟨col,hc,rfl⟩),mem]
    have got := Logical.reference_dw rest c.parameters (entry pointers memory c.pid) fresh h.distinct.1
      (⟨col,lt_of_lt_of_le hc h.block_covers⟩,PUnit.unit) hc
    have got : (Backward.reference (Logical.algebra rest) Logical.sumPrimitive c.parameters (entry pointers memory c.pid)).mem
      c.dw.region (c.pid*c.columns+col) = .mk .fp32 (Logical.groupDW c.parameters (entry pointers memory c.pid) col) := by
        simpa only [BackwardCall.parameters,entry,h.dwStride] using got
    rw [got]
    change Logical.groupDW c.parameters (entry pointers memory c.pid) col = _
    unfold Logical.groupDW
    apply Finset.sum_congr rfl
    intro row hr
    have rb : row < c.rows := lt_of_lt_of_le (Finset.mem_Ico.mp hr).2 (min_le_right _ _)
    obtain ⟨hx,hw,hd,hm,hi⟩ := backward_inputs c h memory row rb
    rw [hd col hc,hm,hi]
    unfold Logical.xhat
    rw [hx col hc]
  · intro col hc
    rw [logical_read h.db (bind _ (by simp [BackwardCall.buffers])) _ _ (input_bound h.db _ _ h.pid_bound hc)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inr (Or.inl ⟨col,hc,rfl⟩)),mem]
    have got := Logical.reference_db rest c.parameters (entry pointers memory c.pid) fresh
      (⟨col,lt_of_lt_of_le hc h.block_covers⟩,PUnit.unit) hc
    have got : (Backward.reference (Logical.algebra rest) Logical.sumPrimitive c.parameters (entry pointers memory c.pid)).mem
      c.db.region (c.pid*c.columns+col) = .mk .fp32 (Logical.groupDB c.parameters (entry pointers memory c.pid) col) := by
        simpa only [BackwardCall.parameters,entry,h.dbStride] using got
    rw [got]
    change Logical.groupDB c.parameters (entry pointers memory c.pid) col = _
    unfold Logical.groupDB
    apply Finset.sum_congr rfl
    intro row hr
    have rb : row < c.rows := lt_of_lt_of_le (Finset.mem_Ico.mp hr).2 (min_le_right _ _)
    exact (backward_inputs c h memory row rb).2.2.1 col hc
  · intro row lo hr col hc
    have rb : row < c.rows := lt_of_lt_of_le hr (min_le_right _ _)
    rw [logical_read h.dx (bind _ (by simp [BackwardCall.buffers])) _ _ (input_bound h.dx _ _ rb hc)]
    rw [commit_read _ _ _ _ h.physicalWrites _ (Or.inr (Or.inr ⟨row,col,lo,hr,hc,rfl⟩)),mem]
    have got := Logical.reference_dx rest c.parameters (entry pointers memory c.pid) fresh h.distinct.2.1 h.distinct.2.2 h.block_covers
      (by simpa only [BackwardCall.parameters,h.dxStride] using Nat.le_refl c.columns) row lo hr (⟨col,lt_of_lt_of_le hc h.block_covers⟩,PUnit.unit) hc
    have got : (Backward.reference (Logical.algebra rest) Logical.sumPrimitive c.parameters (entry pointers memory c.pid)).mem
      c.dx.region (row*c.columns+col) = .mk .fp32 (Logical.rowDX c.parameters (entry pointers memory c.pid) row col) := by
        simpa only [BackwardCall.parameters,entry,h.dxStride] using got
    rw [got]
    change Logical.rowDX c.parameters (entry pointers memory c.pid) row col = _
    unfold Logical.rowDX
    obtain ⟨hx,hw,hd,hm,hi⟩ := backward_inputs c h memory row rb
    rw [hm,hi]
    exact dx_congr _ _ _ _ _ _ _ _ _ _ hc hx hw hd
  · intro l hl
    rw [mem]
    exact Backward.reference_frame _ _ _ _ _ _ (backward_outside c h memory l hl)
  · exact commit_frame _ _ _ _

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract

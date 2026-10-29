import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.MemoryProof
noncomputable section
set_option maxHeartbeats 8000000
def Preserves {α : Type} (S : Addr → Prop) (before after : Memory α) :=
  ∀ a,¬S a → after a=before a
theorem preserves_refl {α : Type} (S : Addr → Prop) (m : Memory α) : Preserves S m m := by
  intro a _; rfl
theorem preserves_trans {α : Type} {S : Addr → Prop} {a b c : Memory α}
    (h : Preserves S a b) (k : Preserves S b c) : Preserves S a c := by
  intro p hp; rw [k p hp,h p hp]
theorem write_preserves {α : Type} (S mask : Addr → Prop) (m : Memory α) (value : Addr → α)
    (h : ∀ a,mask a → S a) : Preserves S m (write m mask value) := by
  intro a ha
  simp only [write]
  exact if_neg (fun hm => ha (h a hm))
theorem row_preserves {α : Type} (S : Addr → Prop) (p : Ops α) (m : Memory α)
    (r : Region) (plane row lo hi : Nat) (value : Nat → α)
    (h : ∀ col,lo≤col → col<hi → S ⟨r,plane,row,col⟩) :
    Preserves S m (rowStore p m r plane row lo hi value) := by
  apply write_preserves
  rintro ⟨r',p',t',j⟩ ⟨hr,hp,ht,hlo,hhi⟩
  dsimp at hr hp ht hlo hhi
  subst r'; subst p'; subst t'
  exact h j hlo hhi
theorem state_preserves {α : Type} (S : Addr → Prop) (s : Shape) (p : Ops α) (m : Memory α)
    (r : Region) (bh iv BV : Nat) (hval : Nat → Nat → α)
    (h : ∀ k v,k<s.K → iv*BV≤v → v<min ((iv+1)*BV) s.V → S ⟨r,bh,k,v⟩) :
    Preserves S m (stateStore s p m r bh iv BV hval) := by
  apply write_preserves
  rintro ⟨r',b',k,v⟩ ⟨hr,hb,hk,hlo,hhi⟩
  dsimp at hr hb hk hlo hhi
  subst r';subst b'
  exact h k v hk hlo hhi
theorem fold_preserves {α β : Type} (S : Addr → Prop) (step : β → Memory α → Memory α)
    (xs : List β) (m : Memory α) (h : ∀ x∈xs,∀ m,Preserves S m (step x m)) :
    Preserves S m (xs.foldl (fun m x => step x m) m) := by
  induction xs generalizing m with
  | nil => exact preserves_refl S m
  | cons x xs ih =>
    apply preserves_trans (h x (by simp) m)
    exact ih _ (fun y hy => h y (by simp [hy]))

def ForwardWrites (s : Shape) (f : Flags) (a : Addr) :=
  a.plane<s.BH ∧ (match a.region with
  | .v | .o => a.row<s.T ∧ a.col<s.V
  | .ht => f.final=Bool.true ∧ a.row<s.K ∧ a.col<s.V
  | _ => False)
def BackwardWrites (s : Shape) (f : Flags) (a : Addr) :=
  match a.region with
  | .dq | .dk => a.plane<s.BNV*s.BH ∧ a.row<s.T ∧ a.col<s.K
  | .dv => a.plane<s.BH ∧ a.row<s.T ∧ a.col<s.V
  | .dbeta => a.plane<s.BNV*s.BH ∧ a.row<s.T ∧ a.col<(if f.vectorBeta then s.V else 1)
  | .dh0 => f.initialGrad=Bool.true ∧ a.plane<s.BH ∧ a.row<s.K ∧ a.col<s.V
  | .retq | .retk => a.plane<s.BH ∧ a.row<s.T ∧ a.col<s.K
  | .retv => a.plane<s.BH ∧ a.row<s.T ∧ a.col<s.V
  | .retbeta => a.plane<s.BH ∧ a.row<s.T ∧ a.col<(if f.vectorBeta then s.V else 1)
  | _ => False

theorem forward_step_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv t : Nat) (hb : bh<s.BH) (ht : t<s.T) (st : Machine α) :
    Preserves (ForwardWrites s f) st.mem (forwardStep s f p g scale bh iv t st).mem := by
  have row (r : Region) (hr : r=.v ∨ r=.o) (m : Memory α) (value : Nat → α) :
      Preserves (ForwardWrites s f) m (rowStore p m r bh t (iv*s.FBV) (min ((iv+1)*s.FBV) s.V) value) := by
    apply row_preserves
    intro j _ hj
    rcases hr with rfl|rfl <;> exact ⟨hb,ht,lt_of_lt_of_le hj (Nat.min_le_right _ _)⟩
  unfold forwardStep
  dsimp only
  exact preserves_trans (row .v (Or.inl rfl) _ _) (row .o (Or.inr rfl) _ _)

theorem forward_loop_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (hb : bh<s.BH) (hn : n≤s.T) (st : Machine α) :
    Preserves (ForwardWrites s f) st.mem (forwardLoop s f p g scale bh iv n st).mem := by
  induction n with
  | zero => exact preserves_refl _ _
  | succ n ih =>
    exact preserves_trans (ih (by omega))
      (forward_step_frame s f p g scale bh iv n hb (by omega) _)
theorem forward_program_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (hb : bh<s.BH) (m : Memory α) :
    Preserves (ForwardWrites s f) m (forwardProgram s f p g scale bh iv m) := by
  unfold forwardProgram
  dsimp only
  split
  · rename_i hf
    apply preserves_trans (forward_loop_frame s f p g scale bh iv s.T hb le_rfl ⟨seed s p m .h0 f.initial bh iv s.FBV,m⟩)
    apply state_preserves
    intro k v hk _ hv
    exact ⟨hb,hf,hk,lt_of_lt_of_le hv (Nat.min_le_right _ _)⟩
  · exact forward_loop_frame s f p g scale bh iv s.T hb le_rfl ⟨seed s p m .h0 f.initial bh iv s.FBV,m⟩

theorem job_member (BH NV : Nat) (bh iv : Nat) : (bh,iv)∈jobs BH NV ↔ bh<BH ∧ iv<NV := by
  simp [jobs,List.mem_flatMap,List.mem_map]
theorem forward_grid_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : Preserves (ForwardWrites s f) m (forwardGrid s f p g scale m) := by
  apply fold_preserves
  intro job hj mem
  exact forward_program_frame s f p g scale job.1 job.2 ((job_member _ _ _ _).mp hj).1 mem

theorem scratch_plane (s : Shape) (bh iv : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) :
    iv*s.BH+bh<s.BNV*s.BH := by
  calc
    _ < iv*s.BH+s.BH := Nat.add_lt_add_left hb _
    _ = (iv+1)*s.BH := by rw [Nat.add_mul];simp
    _ ≤ s.BNV*s.BH := Nat.mul_le_mul_right _ (by omega)

theorem reverse_step_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (hn : n<s.T) (st : Machine α) :
    Preserves (BackwardWrites s f) st.mem (reverseStep s f p g scale bh iv n st).mem := by
  have hp:=scratch_plane s bh iv hb hv
  have ht : s.T-1-n<s.T := by omega
  have hd (mem : Memory α) (value : Nat → α) : Preserves (BackwardWrites s f) mem
      (rowStore p mem .dk (iv*s.BH+bh) (s.T-1-n) 0 s.K value) := by
    apply row_preserves
    intro j _ hj
    exact ⟨hp,ht,hj⟩
  have hvv (mem : Memory α) (value : Nat → α) : Preserves (BackwardWrites s f) mem
      (rowStore p mem .dv bh (s.T-1-n) (iv*s.BBV) (min ((iv+1)*s.BBV) s.V) value) := by
    apply row_preserves
    intro j _ hj
    exact ⟨hb,ht,lt_of_lt_of_le hj (Nat.min_le_right _ _)⟩
  unfold reverseStep
  dsimp only
  apply preserves_trans (preserves_trans (hd _ _) (hvv _ _))
  split
  · rename_i hbeta
    apply row_preserves
    intro j _ hj
    exact ⟨hp,ht,by simpa [hbeta] using lt_of_lt_of_le hj (Nat.min_le_right _ _)⟩
  · rename_i hbeta
    apply row_preserves
    intro j _ hj
    exact ⟨hp,ht,by simpa [hbeta] using hj⟩

theorem reverse_loop_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (hn : n≤s.T) (st : Machine α) :
    Preserves (BackwardWrites s f) st.mem (reverseLoop s f p g scale bh iv n st).mem := by
  induction n with
  | zero => exact preserves_refl _ _
  | succ n ih =>
    exact preserves_trans (ih (by omega))
      (reverse_step_frame s f p g scale bh iv n hb hv (by omega) _)
theorem recompute_step_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv t : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (ht : t<s.T) (st : Machine α) :
    Preserves (BackwardWrites s f) st.mem (recomputeStep s f p g scale bh iv t st).mem := by
  have row (r : Region) (hr : r=.dk ∨ r=.dq) (m : Memory α) (value : Nat → α) :
      Preserves (BackwardWrites s f) m (rowStore p m r (iv*s.BH+bh) t 0 s.K value) := by
    apply row_preserves
    intro j _ hj
    rcases hr with rfl|rfl <;> exact ⟨scratch_plane s bh iv hb hv,ht,hj⟩
  unfold recomputeStep
  dsimp only
  exact preserves_trans (row .dk (Or.inl rfl) _ _) (row .dq (Or.inr rfl) _ _)

theorem recompute_loop_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (hn : n≤s.T) (st : Machine α) :
    Preserves (BackwardWrites s f) st.mem (recomputeLoop s f p g scale bh iv n st).mem := by
  induction n with
  | zero => exact preserves_refl _ _
  | succ n ih =>
    exact preserves_trans (ih (by omega))
      (recompute_step_frame s f p g scale bh iv n hb hv (by omega) _)
theorem backward_program_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (hb : bh<s.BH) (hv : iv<s.BNV) (m : Memory α) :
    Preserves (BackwardWrites s f) m (backwardProgram s f p g scale bh iv m) := by
  let st:=reverseLoop s f p g scale bh iv s.T ⟨seed s p m .dht f.terminalGrad bh iv s.BBV,m⟩
  have hrev : Preserves (BackwardWrites s f) m st.mem :=
    reverse_loop_frame s f p g scale bh iv s.T hb hv le_rfl ⟨seed s p m .dht f.terminalGrad bh iv s.BBV,m⟩
  let mem:=if f.initialGrad then stateStore s p st.mem .dh0 bh iv s.BBV st.h else st.mem
  have hstore : Preserves (BackwardWrites s f) st.mem mem := by
    dsimp only [mem]
    split
    · rename_i hf
      apply state_preserves
      intro k v hk _ hval
      exact ⟨hf,hb,hk,lt_of_lt_of_le hval (Nat.min_le_right _ _)⟩
    · exact preserves_refl _ _
  exact preserves_trans (preserves_trans hrev hstore)
    (recompute_loop_frame s f p g scale bh iv s.T hb hv le_rfl ⟨seed s p mem .h0 f.initial bh iv s.BBV,mem⟩)

theorem backward_grid_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : Preserves (BackwardWrites s f) m (backwardGrid s f p g scale m) := by
  apply fold_preserves
  intro job hj mem
  obtain ⟨hb,hv⟩:=(job_member _ _ _ _).mp hj
  exact backward_program_frame s f p g scale job.1 job.2 hb hv mem

#axiomsClean forward_grid_frame
#axiomsClean backward_grid_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.MemoryProof

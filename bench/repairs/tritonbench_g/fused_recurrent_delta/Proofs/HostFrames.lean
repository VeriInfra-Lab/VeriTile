import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Memory
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Native
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.MemoryProof
noncomputable section
set_option maxHeartbeats 5000000
def ForwardCallWrites (s : Shape) (f : Flags) (a : Addr) := ForwardWrites s f a ∨
  (f.defaultBeta=Bool.true ∧ a.region=.beta ∧ a.plane<s.BH ∧ a.row<s.T ∧ a.col=0)
theorem preserves_mono {α : Type} (S T : Addr → Prop) (m n : Memory α)
    (h : ∀ a,S a → T a) (hm : Preserves S m n) : Preserves T m n := by
  intro a ha
  exact hm a (fun hs => ha (h a hs))
theorem prepare_beta_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α) :
    Preserves (ForwardCallWrites s f) m (prepareBeta s f p m) := by
  unfold prepareBeta
  split
  · rename_i hf
    apply write_preserves
    intro a ha
    exact Or.inr ⟨hf,ha⟩
  · exact preserves_refl _ _
theorem prepare_dbeta_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α) :
    Preserves (BackwardWrites s f) m (prepareDbeta s f p m) := by
  unfold prepareDbeta
  split
  · rename_i hf
    apply write_preserves
    rintro ⟨r,b,t,v⟩ ⟨hr,hb,ht,hv⟩
    dsimp at hr hb ht hv
    subst r
    exact ⟨hb,ht,by simpa [hf] using hv⟩
  · exact preserves_refl _ _
theorem host_reduce_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α) :
    Preserves (BackwardWrites s f) m (hostReduce s f p g m) := by
  apply write_preserves
  rintro ⟨r,b,t,v⟩ ha
  cases r <;> simp_all [BackwardWrites]
theorem forward_call_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : Preserves (ForwardCallWrites s f) m (forwardCall s f p g scale m) := by
  exact preserves_trans (prepare_beta_frame s f p m)
    (preserves_mono _ _ _ _ (fun _ h => Or.inl h) (forward_grid_frame s f p g scale _))
theorem backward_call_frame {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : Preserves (BackwardWrites s f) m (backwardCall s f p g scale m) := by
  exact preserves_trans (preserves_trans (prepare_dbeta_frame s f p m) (backward_grid_frame s f p g scale _))
    (host_reduce_frame s f p g _)
theorem native_forward_frame (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (p : Native.Instructions) (policy : Native.Policy) (m : Memory Native.Word) :
    Preserves (ForwardCallWrites s f) m (Native.forward s f g scale p policy m) := by
  intro a ha
  have h:=forward_call_frame s f Native.graphOps g (.bits scale) (Native.graphMemory m) a ha
  change (forwardCall s f Native.graphOps g (.bits scale) (Native.graphMemory m) a).eval p policy=m a
  rw [h]
  simp [Native.graphMemory,Native.Expr.eval]
theorem native_backward_frame (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (p : Native.Instructions) (policy : Native.Policy) (m : Memory Native.Word) :
    Preserves (BackwardWrites s f) m (Native.backward s f g scale p policy m) := by
  intro a ha
  have h:=backward_call_frame s f Native.graphOps g (.bits scale) (Native.graphMemory m) a ha
  change (backwardCall s f Native.graphOps g (.bits scale) (Native.graphMemory m) a).eval p policy=m a
  rw [h]
  simp [Native.graphMemory,Native.Expr.eval]

#axiomsClean forward_call_frame
#axiomsClean backward_call_frame
#axiomsClean native_forward_frame
#axiomsClean native_backward_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.MemoryProof

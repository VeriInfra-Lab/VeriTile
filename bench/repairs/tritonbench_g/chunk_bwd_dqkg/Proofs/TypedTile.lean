import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedOutput
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
open Scatter (masked_readback_active)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def prepare (c : Config) (s : State α) (a : CTA c) : State α :=
  {s with regs:=fun _ _ _=>none
          pids:=fun i=>if i=0 then a.key.val else if i=1 then a.time.val else if i=2 then a.head.val else 0
          numPids:=fun i=>if i=0 then c.NK else if i=1 then c.NT else if i=2 then c.heads else 1}

def resultQ (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) (e : Fin c.BK) : α :=
  M.cast (some .fp32) .real .fp32
    (answerQ M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c)) (r,e,()))
def resultK (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) (e : Fin c.BK) : α :=
  M.cast (some .fp32) .real .fp32
    (answerK M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c)) (r,e,()))
def resultG (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) : α :=
  M.cast (some .fp32) .real .real
    (answerG M dot c (prepare c s a) (accumulated M dot c (prepare c s a) (valueBlocks c)) (r,()))

theorem q_active_prepare (c : Config) (s : State α) (a : CTA c) (i : TileIndex [64,c.BK]) :
    qActive c (prepare c s a) i ↔ activeQK c a i.1 i.2.1 := by rfl

theorem q_offset_prepare (c : Config) (s : State α) (a : CTA c) (i : TileIndex [64,c.BK]) :
    qOffset c (prepare c s a) i=qAddress c a i.1 i.2.1 := by
  simp [qOffset,prepare,qAddress,Config.layout,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem g_active_prepare (c : Config) (s : State α) (a : CTA c) (i : TileIndex [64]) :
    gActive c (prepare c s a) i ↔ activeG c a i.1 := by rfl

theorem g_offset_prepare (c : Config) (s : State α) (a : CTA c) (i : TileIndex [64]) :
    gOffset c (prepare c s a) i=gAddress c a i.1 := by
  simp [gOffset,prepare,gAddress,Nat.add_assoc]

theorem q_offset_injective (c : Config) (s : State α) (a : CTA c)
    (i j : TileIndex [64,c.BK]) (hi : qActive c (prepare c s a) i) (hj : qActive c (prepare c s a) j)
    (he : qOffset c (prepare c s a) i=qOffset c (prepare c s a) j) : i=j := by
  rw [q_offset_prepare,q_offset_prepare] at he
  obtain ⟨_,hr,hk⟩:=q_unique c a a i.1 j.1 i.2.1 j.2.1 hi hj he
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  simp_all

theorem g_offset_injective (c : Config) (s : State α) (a : CTA c)
    (i j : TileIndex [64]) (hi : gActive c (prepare c s a) i) (hj : gActive c (prepare c s a) j)
    (he : gOffset c (prepare c s a) i=gOffset c (prepare c s a) j) : i=j := by
  rw [g_offset_prepare,g_offset_prepare] at he
  have hr: i.1=j.1 := (g_unique c a a i.1 j.1 hi hj he).2
  rcases i with ⟨i,⟨⟩⟩
  rcases j with ⟨j,⟨⟩⟩
  simp_all

theorem writeQK_frame (M : Algebra α) (c : Config) (initial current : State α)
    (region : RegionName) (value : TileIndex [64,c.BK] → α) (r : RegionName) (off : Nat)
    (h : r≠region ∨ ∀ i,qActive c initial i → off≠qOffset c initial i) :
    (writeQK M c initial current region value).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases h with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _=>h i)

theorem writeG_frame (M : Algebra α) (c : Config) (initial current : State α)
    (value : TileIndex [64] → α) (r : RegionName) (off : Nat)
    (h : r≠("dg" : RegionName) ∨ ∀ i,gActive c initial i → off≠gOffset c initial i) :
    (writeG M c initial current value).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases h with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _=>h i)

theorem tile_q (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) (e : Fin c.BK) (h : activeQK c a r e) :
    (finalState M dot c (prepare c s a)).mem "dq" (qAddress c a r e)=.mk .fp32 (resultQ M dot c s a r e) := by
  rw [source_memory]
  unfold written
  rw [writeG_frame M c _ _ _ _ _ (Or.inl (by decide)),writeQK_frame M c _ _ _ _ _ _ (Or.inl (by decide))]
  rw [←q_offset_prepare c s a (r,e,())]
  exact masked_readback_active _ _ _ _ _ (r,e,()) h (q_offset_injective c s a)

theorem tile_k (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) (e : Fin c.BK) (h : activeQK c a r e) :
    (finalState M dot c (prepare c s a)).mem "dk" (qAddress c a r e)=.mk .fp32 (resultK M dot c s a r e) := by
  rw [source_memory]
  unfold written
  rw [writeG_frame M c _ _ _ _ _ (Or.inl (by decide)),←q_offset_prepare c s a (r,e,())]
  exact masked_readback_active _ _ _ _ _ (r,e,()) h (q_offset_injective c s a)

theorem tile_g (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (r : Fin 64) (h : activeG c a r) :
    (finalState M dot c (prepare c s a)).mem "dg" (gAddress c a r)=.mk .real (resultG M dot c s a r) := by
  rw [source_memory]
  unfold written
  rw [←g_offset_prepare c s a (r,())]
  exact masked_readback_active _ _ _ _ _ (r,()) h (g_offset_injective c s a)

theorem tile_frame (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (region : RegionName) (off : Nat) (h : ¬writes c a (region,off)) :
    (finalState M dot c (prepare c s a)).mem region off=s.mem region off := by
  rw [writes_cases] at h
  rw [source_memory]
  unfold written
  rw [writeG_frame,writeQK_frame,writeQK_frame]
  · rfl
  · by_cases hr : region=("dq" : RegionName)
    · right;intro i hi he;apply h;left;exact ⟨hr,i.1,i.2.1,hi,(q_offset_prepare c s a i).symm.trans he.symm⟩
    · exact Or.inl hr
  · by_cases hr : region=("dk" : RegionName)
    · right;intro i hi he;apply h;right;left;exact ⟨hr,i.1,i.2.1,hi,(q_offset_prepare c s a i).symm.trans he.symm⟩
    · exact Or.inl hr
  · by_cases hr : region=("dg" : RegionName)
    · right;intro i hi he;apply h;right;right;exact ⟨hr,i.1,hi,(g_offset_prepare c s a i).symm.trans he.symm⟩
    · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed

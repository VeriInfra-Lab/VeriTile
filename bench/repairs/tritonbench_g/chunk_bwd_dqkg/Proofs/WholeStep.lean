import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.Geometry
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFrame

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def kernel (c : Config) : ComputeKernel :=
  TypedSource.original .fp32 .fp32 .fp32 .fp32
    "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
    c.layout.s_k_h c.layout.s_k_t c.layout.s_v_h c.layout.s_v_t
    c.layout.s_h_h c.layout.s_h_t c.scale c.T c.K c.V 64 c.BK c.BV c.NT

/-- A CTA receives the preceding CTA's complete current memory. Its private
register file is fresh. The zero undef channel is internal: every checked-off
source load explicitly requests zero padding. -/
def prepare (c : Config) (s : BlockState) (a : CTA c) : BlockState :=
  {s with
    regs := fun _ _ _ => none
    pids := fun i => if i = 0 then a.key.val else if i = 1 then a.time.val else if i = 2 then a.head.val else 0
    numPids := fun i => if i = 0 then c.NK else if i = 1 then c.NT else if i = 2 then c.heads else 1
    undef := fun _ _ => 0}

def inputs (s : BlockState) : RealMath.Inputs :=
  RealPost.inputs s "q" "k" "v" "h" "g" "do" "dh"

def writes (c : Config) (a : CTA c) : WriteFootprint :=
  RealFrame.footprint c.layout a.launch "dq" "dk" "dg"

theorem launch_prepare (c : Config) (s : BlockState) (a : CTA c) :
    RealLoop.launch (prepare c s a) = a.launch := by
  rfl

theorem inputs_prepare (c : Config) (s : BlockState) (a : CTA c) :
    inputs (prepare c s a) = inputs s := rfl

theorem source_q_address (c : Config) (a : CTA c) (r : Fin 64) (e : Fin c.BK) :
    RealFrame.qkAddress c.layout a.launch (r,e,PUnit.unit) = qAddress c a r e := by
  simp only [RealFrame.qkAddress,Config.layout,CTA.launch,qAddress,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem source_g_address (c : Config) (a : CTA c) (r : Fin 64) :
    RealFrame.dgAddress c.layout a.launch (r,PUnit.unit) = gAddress c a r := rfl

theorem writes_cases (c : Config) (a : CTA c) (region : RegionName) (off : Nat) :
    writes c a (region,off) ↔
      (region = "dq" ∧ ∃ r e, activeQK c a r e ∧ qAddress c a r e = off) ∨
      (region = "dk" ∧ ∃ r e, activeQK c a r e ∧ qAddress c a r e = off) ∨
      (region = "dg" ∧ ∃ r, activeG c a r ∧ gAddress c a r = off) := by
  simp [writes,RealFrame.footprint,WriteFootprint.activeTileImage,
    Prod.exists,RealFrame.qkAddress,RealFrame.dgAddress,RealFrame.qkActive,
    RealFrame.dgActive,Config.layout,CTA.launch,activeQK,activeG,qAddress,gAddress,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

theorem writes_unique (c : Config) (a b : CTA c) (region : RegionName) (off : Nat)
    (ha : writes c a (region,off)) (hb : writes c b (region,off)) : a = b := by
  rw [writes_cases] at ha hb
  rcases ha with ⟨hr,r,e,ha,he⟩ | ⟨hr,r,e,ha,he⟩ | ⟨hr,r,ha,he⟩
  · subst region
    rcases hb with ⟨_,s,f,hb,hf⟩ | ⟨hbad,_⟩ | ⟨hbad,_⟩
    · exact (q_unique c a b r s e f ha hb (he.trans hf.symm)).1
    all_goals contradiction
  · subst region
    rcases hb with ⟨hbad,_⟩ | ⟨_,s,f,hb,hf⟩ | ⟨hbad,_⟩
    · contradiction
    · exact (q_unique c a b r s e f ha hb (he.trans hf.symm)).1
    · contradiction
  · subst region
    rcases hb with ⟨hbad,_⟩ | ⟨hbad,_⟩ | ⟨_,s,hb,hf⟩
    · contradiction
    · contradiction
    · exact (g_unique c a b r s ha hb (he.trans hf.symm)).1

theorem input_unwritten (c : Config) (a : CTA c) (region : RegionName) (off : Nat)
    (hr : region ∈ (["q","k","v","h","g","do","dh"] : List RegionName)) : ¬writes c a (region,off) := by
  rw [writes_cases]
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp

/-- Successful complete repaired-source execution, all three mathematical
tiles and exact tagged-cell frame. No program result or success premise. -/
theorem step_exists (c : Config) (s : BlockState) (a : CTA c) :
    ∃ final, exec (kernel c).eraseDType.toAlgKernel (prepare c s a) = some final ∧
      (∀ r e, activeQK c a r e → final.readMem "dq" (qAddress c a r e) =
        RealMath.dq c.layout a.launch (inputs s) r e) ∧
      (∀ r e, activeQK c a r e → final.readMem "dk" (qAddress c a r e) =
        RealMath.dk c.layout a.launch (inputs s) r e) ∧
      (∀ r, activeG c a r → final.readMem "dg" (gAddress c a r) =
        RealMath.dg c.layout a.launch (inputs s) r) ∧
      BlockState.WriteWithin (writes c a) s final := by
  have hinj : ∀ i j : TileIndex [c.layout.BT,c.layout.BK],
      ((prepare c s a).pids 1 * c.layout.BT + i.1.val < c.layout.T ∧
        (prepare c s a).pids 0 * c.layout.BK + i.2.1.val < c.layout.K) →
      ((prepare c s a).pids 1 * c.layout.BT + j.1.val < c.layout.T ∧
        (prepare c s a).pids 0 * c.layout.BK + j.2.1.val < c.layout.K) →
      cbdQkAddr (prepare c s a) c.layout.s_k_h c.layout.s_k_t c.layout.BT c.layout.BK i =
        cbdQkAddr (prepare c s a) c.layout.s_k_h c.layout.s_k_t c.layout.BT c.layout.BK j → i = j := by
    intro i j hi hj he
    apply RealFrame.qk_injective_on c.layout a.launch (Nat.le_refl _) i j hi hj
    simpa only [RealFrame.qkAddress,RealLoop.launch,cbdQkAddr,prepare,CTA.launch,
      Nat.mul_one,Nat.add_assoc] using he
  obtain ⟨final,he,hq,hk,hg⟩ := RealPost.original_executes
    (prepare c s a) c.layout "q" "k" "v" "h" "g" "do" "dh"
    .fp32 .fp32 .fp32 .fp32 "dq" "dk" "dg" rfl (width_pos c.V)
    (by decide) (by decide) (by decide) hinj
  have hf := RealFrame.original_frame (prepare c s a) final c.layout
    .fp32 .fp32 .fp32 .fp32 "q" "k" "v" "h" "g" "do" "dh"
    "dq" "dk" "dg" rfl he
  refine ⟨final,he,?_,?_,?_,?_⟩
  · intro r e ha
    have hh := hq (r,e,PUnit.unit) ha
    simpa only [cbdQkAddr,prepare,Config.layout,CTA.launch,RealLoop.launch,
      qAddress,inputs,RealPost.inputs,BlockState.readMem,Nat.add_mul,Nat.mul_one,
      Nat.mul_assoc,Nat.add_assoc] using hh
  · intro r e ha
    have hh := hk (r,e,PUnit.unit) ha
    simpa only [cbdQkAddr,prepare,Config.layout,CTA.launch,RealLoop.launch,
      qAddress,inputs,RealPost.inputs,BlockState.readMem,Nat.add_mul,Nat.mul_one,
      Nat.mul_assoc,Nat.add_assoc] using hh
  · intro r ha
    have hh := hg (r,PUnit.unit) ha
    simpa only [cbdDgAddr,prepare,Config.layout,CTA.launch,RealLoop.launch,
      gAddress,inputs,RealPost.inputs,BlockState.readMem,Nat.mul_one,Nat.add_assoc] using hh
  · intro region off hm
    exact hf region off hm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole

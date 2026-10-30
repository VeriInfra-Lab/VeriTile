import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DirectLayout

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
open VeriTile.Triton Whole
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Independent mathematical inputs range over all Reals. -/
abbrev RealData := RegionName → Nat → ℝ
def original (data : RealData) : BlockState where
  mem := fun r off => .real (data r off)
  regs := fun _ _ _ => none
  pids := fun _ => 0

/-- The wrapper's existing DG fill; no input copy or normalization occurs. -/
def initialized (p : Metadata) (data : RealData) : BlockState :=
  {original data with mem := fun r off =>
    if r="dg" ∧ off<extent (config p) "dg" then .real (-1000000000) else .real (data r off)}

def realAllocation (p : Metadata) (data : RealData) : ChunkBwdDqkg.Allocation (original data) where
  state := initialized p data
  frame := by intro r off hq hk hg; simp [initialized,original,hg]

/-- Both warp configurations execute the same erased arithmetic. Every tuner
trial and the selected launch use the state produced by their predecessor. -/
def trace (c : Config) (trials : List (Fin 2)) (chosen : Fin 2) : List (CTA c) :=
  (trials++[chosen]).flatMap fun _ => allCTAs c

theorem trace_covers (c : Config) (trials : List (Fin 2)) (chosen : Fin 2) (a : CTA c) :
    a∈trace c trials chosen := by
  exact List.mem_flatMap.mpr ⟨chosen,by simp,mem_allCTAs c a⟩

def executeReal (p : Metadata) (a : Allocation p) (trials : List (Fin 2)) (chosen : Fin 2)
    (data : RealData) : Option BlockState :=
  runFlat (config p) (flat p a) (trace (config p) trials chosen) ((flat p a).flattenState (initialized p data))

structure ReturnedTensor where
  object : Nat
  shape : List Nat
  strides : List Nat
  pointer : Nat
  device : VeriTile.Torch.Device
  dtype : VeriTile.Torch.DType

def returned (p : Metadata) (a : Allocation p) (r : Role) : ReturnedTensor :=
  ⟨a.objects r,shape p r,canonicalStrides (shape p r),pointer p a r,(p.tensor .q).device,.floating .fp32⟩

theorem real_complete (p : Metadata) (h : InputContract p) (a : Allocation p)
    (trials : List (Fin 2)) (chosen : Fin 2) (data : RealData) :
    ∃ final,executeReal p a trials chosen data=some final ∧
      (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
        final.readMem (flat p a).flat ((flat p a).addr "dq" ((hd.val*(config p).T+t.val)*(config p).K+e.val))=
          Spec.dq (config p) (Spec.views (config p) data) hd.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
        final.readMem (flat p a).flat ((flat p a).addr "dk" ((hd.val*(config p).T+t.val)*(config p).K+e.val))=
          Spec.dk (config p) (Spec.views (config p) data) hd.val (t.val/64) (t.val%64) e.val) ∧
      (∀ (k : Fin (config p).NK) (hd : Fin (config p).heads) (t : Fin (config p).T),
        final.readMem (flat p a).flat ((flat p a).addr "dg" ((k.val*(config p).heads+hd.val)*(config p).T+t.val))=
          Spec.dg (config p) (Spec.views (config p) data) k.val hd.val (t.val/64) (t.val%64)) ∧
      BlockState.WriteWithin (flatWrites (config p) (flat p a))
        ((flat p a).flattenState (initialized p data)) final := by
  let c := config p
  let A := flat p a
  let s := initialized p data
  have ha := denseAllocation p h a
  obtain ⟨final,run,hq,hk,hg,frame⟩ := grid_exists c (trace c trials chosen) s
  have inputs_eq := allocation_inputs (original data) (realAllocation p data)
  have fullframe : BlockState.WriteWithin (gridWrites c (allCTAs c)) s final := by
    intro r off outside
    apply frame r off
    rintro ⟨cta,_,hw⟩
    exact outside ⟨cta,mem_allCTAs c cta,hw⟩
  refine ⟨A.flattenState final,?_,?_,?_,?_,whole_flat_frame c A s final fullframe⟩
  · rw [executeReal,grid_flatten c A ha,run]
    rfl
  · intro hd t e
    rw [A.flattenState_readMem ha.apart final (by simp [A,flat,regions])
      (by change _<extent c "dq"; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ hd.isLt t.isLt) e.isLt)]
    have value := hq (qOwner c hd t e) (trace_covers c trials chosen _) (rowLane t.val) (keyLane c e.val) (q_owner_active c hd t e)
    rw [q_owner_address] at value
    change final.readMem "dq" _=expectedQ c s hd t e at value
    have same : expectedQ c s hd t e=expectedQ c (original data) hd t e := by
      unfold expectedQ
      exact congrArg (fun x=>RealMath.dq c.layout (qOwner c hd t e).launch x (rowLane t.val) (keyLane c e.val)) inputs_eq
    exact value.trans (same.trans (Spec.logical_q c (original data) hd t e))
  · intro hd t e
    rw [A.flattenState_readMem ha.apart final (by simp [A,flat,regions])
      (by change _<extent c "dk"; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ hd.isLt t.isLt) e.isLt)]
    have value := hk (qOwner c hd t e) (trace_covers c trials chosen _) (rowLane t.val) (keyLane c e.val) (q_owner_active c hd t e)
    rw [q_owner_address] at value
    change final.readMem "dk" _=expectedK c s hd t e at value
    have same : expectedK c s hd t e=expectedK c (original data) hd t e := by
      unfold expectedK
      exact congrArg (fun x=>RealMath.dk c.layout (qOwner c hd t e).launch x (rowLane t.val) (keyLane c e.val)) inputs_eq
    exact value.trans (same.trans (Spec.logical_k c (original data) hd t e))
  · intro k hd t
    rw [A.flattenState_readMem ha.apart final (by simp [A,flat,regions])
      (by change _<extent c "dg"; simpa [extent] using pack_bound _ _ _ _ (pack_bound _ _ _ _ k.isLt hd.isLt) t.isLt)]
    have value := hg (gOwner c k hd t) (trace_covers c trials chosen _) (rowLane t.val) (g_owner_active c k hd t)
    rw [g_owner_address] at value
    change final.readMem "dg" _=expectedG c s k hd t at value
    have same : expectedG c s k hd t=expectedG c (original data) k hd t := by
      unfold expectedG
      exact congrArg (fun x=>RealMath.dg c.layout (gOwner c k hd t).launch x (rowLane t.val)) inputs_eq
    exact value.trans (same.trans (Spec.logical_g c (original data) k hd t))

#axiomsClean real_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct

import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReverseInit
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReverseLoop
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelPost

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelMemory KernelReverseLoop
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem initial_holds (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    Holds M c s p u 0 (KernelPrefix.backward M c s seed u).val := by
  obtain ⟨hs,hi,hx,hd,hg,ha⟩ := KernelReverseInit.registers M c s seed p u hh hk hv
  refine ⟨Nat.zero_le _,KernelReverseInit.fixed M c s seed p u hh hk hv,
    hs,hi,hx,hd,hg,ha,?_,?_,?_,?_,?_,?_⟩
  · intro o; rfl
  · intro o; rfl
  · intro o; rfl
  · intro m hm; omega
  · intro m hm; omega
  · intro r o h; rfl

theorem source_correct (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (nonempty : 0<c.N)
    (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    ∃ v, FP.Structural.exec (engine M) (source .backward c s seed) u=some v ∧
      (∀ (t : Fin (c.N-1)) (e : Fin 64) (j : Fin 16),
        v.mem "DI" (address c p t.val e.val j.val)=
          KernelReference.stored M .real (carry M c s p u (c.N-1-t.val) (e,j,PUnit.unit))) ∧
      (∀ t : Fin (c.N-1),v.mem "DG" (gradientAddress c p t.val)=
        KernelReference.stored M .real
          (KernelReference.contraction M (carry M c s p u (c.N-1-t.val))
            ((KernelReference.read c s p u).saved t.val))) ∧
      (∀ (e : Fin 64) (j : Fin 16),v.mem "DL" (dseedAddress c p e.val j.val)=
        KernelReference.stored M .fp32 (carry M c s p u c.N (e,j,PUnit.unit))) ∧
      Frame .backward c p u v := by
  obtain ⟨w,he,inv⟩ := KernelReverseLoop.executes M c s p u 0
    (KernelPrefix.backward M c s seed u).val (initial_holds M c s seed p u hh hk hv)
  have incomingOff : KernelReversePointers.off c p (c.N-1) (c.N-1)=
      (fun e j => address c p 0 e.val j.val) := by
    funext e j
    simpa using KernelReversePointers.live c p (c.N-1) (c.N-1) (Nat.le_refl _) e j
  have gateOff : KernelReversePointers.gate c p (c.N-1)=gateAddress c p 0 := by
    simpa using KernelReversePointers.gate_live c p (c.N-1) (Nat.le_refl _)
  have hx : w.regs .ptr [64,16] "DS"=some (ptr "DS" (fun e j => address c p 0 e.val j.val)) := by
    simpa [incomingOff] using inv.incomingPtr
  have hd : w.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gateAddress c p 0)) := by
    simpa [gateOff] using inv.gatePtr
  let acc := carry M c s p u (c.N-1)
  let v := KernelPost.result M c p acc w
  have post := KernelPost.executes M c s seed p acc w inv.fixed hx hd inv.acc
  have values : KernelReverseStep.value M (fun e j => address c p 0 e.val j.val)
      (gateAddress c p 0) acc w=carry M c s p u c.N := by
    have Nnext : c.N=(c.N-1)+1 := by omega
    rw [Nnext]
    funext i
    simp [KernelReverseStep.value,KernelStep.loadedMatrix,KernelStep.loadedScalar,KernelOps.next,
      carry,KernelReference.reverse,KernelReference.update,KernelReference.read,raw,
      inv.readsIncoming,inv.readsGate,acc]
  refine ⟨v,?_,?_,?_,?_,?_⟩
  · change run (engine M) (source .backward c s seed).surfaceBody u=some v
    rw [source_body,List.append_assoc,run_append,(KernelPrefix.backward M c s seed u).property]
    simp only [Option.bind_some,List.singleton_append,run]
    change (step (engine M) (loop .backward c s seed)
      (KernelPrefix.backward M c s seed u).val >>= run (engine M) (after .backward c s seed))=some v
    have hloop : step (engine M) (loop .backward c s seed)
        (KernelPrefix.backward M c s seed u).val=some w := by
      simpa [KernelProgram.loop,KernelOps.backward_body,step,FP.Structural.evalOp] using he
    rw [hloop]
    exact post
  · intro t e j
    simp only [v,KernelPost.result]
    change (scatter "DL" .fp32 (fun a b => dseedAddress c p a.val b.val)
      (fun i => M.cast (some .fp32) .real .fp32
        (KernelReverseStep.value M (fun a b => address c p 0 a.val b.val) (gateAddress c p 0) acc w i))
      (KernelPost.pre M c p acc w)).mem "DI" _=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide)]
    have ti : c.N-2-(c.N-2-t.val)=t.val := by omega
    have ci : (c.N-2-t.val)+1=c.N-1-t.val := by omega
    simpa [ti,ci,KernelPost.pre] using inv.output (c.N-2-t.val) (by omega) e j
  · intro t
    simp only [v,KernelPost.result]
    change (scatter "DL" .fp32 (fun a b => dseedAddress c p a.val b.val)
      (fun i => M.cast (some .fp32) .real .fp32
        (KernelReverseStep.value M (fun a b => address c p 0 a.val b.val) (gateAddress c p 0) acc w i))
      (KernelPost.pre M c p acc w)).mem "DG" _=_
    rw [scatter_other _ _ _ _ _ _ _ (by decide)]
    have ti : c.N-2-(c.N-2-t.val)=t.val := by omega
    have ci : (c.N-2-t.val)+1=c.N-1-t.val := by omega
    simpa [contraction,ti,ci,KernelPost.pre] using inv.gateOutput (c.N-2-t.val) (by omega)
  · intro e j
    simp only [v,KernelPost.result]
    change (scatter "DL" .fp32 (fun a b => dseedAddress c p a.val b.val)
      (fun i => M.cast (some .fp32) .real .fp32
        (KernelReverseStep.value M (fun a b => address c p 0 a.val b.val) (gateAddress c p 0) acc w i))
      (KernelPost.pre M c p acc w)).mem "DL" _=_
    rw [KernelStore.seed_cell,values]
    rfl
  · intro r o outside
    simp only [v,KernelPost.result]
    change (scatter "DL" .fp32 (fun a b => dseedAddress c p a.val b.val)
      (fun i => M.cast (some .fp32) .real .fp32
        (KernelReverseStep.value M (fun a b => address c p 0 a.val b.val) (gateAddress c p 0) acc w i))
      (KernelPost.pre M c p acc w)).mem r o=_
    exact (KernelStore.backward_seed_frame c p _ _ r o outside).trans (inv.frame r o outside)

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBackward

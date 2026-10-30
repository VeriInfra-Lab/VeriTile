import bench.tritonbench_g.chunk_gated_attention.Sources.Materialize

/-! One actual legal flattened program, from materialized logical inputs to
logical history/final readback in finite allocations. All full-tile, bounds and
collision conditions are derived from host padding and launch membership. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

structure Outputs (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (x : Host.InnerInputs) (T K V : Nat) (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT →
    keyIndex p l e.val < K → valueIndex p l j.val < V →
    v.readMem A.flat (A.addr r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))) =
      Host.innerState x T p.BT gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV),
    keyIndex p l e.val < K → valueIndex p l j.val < V →
    v.readMem A.flat (A.addr r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val))) =
      Host.innerState x T p.BT gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

theorem executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (heads T K V : Nat) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool)
    (x : Nat → Host.InnerInputs) (u : BlockState)
    (hp : StagedLayout.Canonical p) (hk : K ≤ p.K) (hv : V ≤ p.V)
    (hpid : u.pids 0+base < heads*(p.K/p.BK)*(p.V/p.BV))
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (ha : StagedLayout.Allocated p heads r gateK hasSeed storeFinal A.extent)
    (hkey : ∀ off, u.readMem r.key off = (Materialize.inputs p T K V gateK x).key off)
    (hvalue : ∀ off, u.readMem r.value off = (Materialize.inputs p T K V gateK x).value off)
    (hgate : ∀ off, u.readMem r.gate off = (Materialize.inputs p T K V gateK x).gate off)
    (hseed : ∀ off, u.readMem r.seed off = (Materialize.inputs p T K V gateK x).seed off) :
    ∃ v, exec (A.flattenKernel (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      Outputs A p (RebasedInit.decoded p base u) r (x (RebasedInit.decoded p base u).head)
        T K V gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (RebasedFlat.footprint A p (RebasedInit.decoded p base u) r storeFinal) (A.flattenState u) v := by
  let l := RebasedInit.decoded p base u
  let a := Materialize.inputs p T K V gateK x
  have hl := StagedLayout.decoded_valid p heads base u hpid
  have ht := Materialize.staged p l T K V gateK x hp hk hv
  obtain ⟨v,he,ho,hf⟩ := RebasedFlat.source_executes A hd hcov p r dot gate history seed final base
    gateK hasSeed storeFinal swapped a u (StagedLayout.full_tile p heads l hp hl)
    (StagedLayout.key_full p heads l hp hl) (StagedLayout.value_full p heads l hp hl)
    hs hfsep (StagedLayout.history_noCollision p l hp) (fun _ => StagedLayout.state_noCollision p l)
    hkey hvalue hgate hseed (StagedLayout.bounds p heads l r gateK hasSeed storeFinal hp hl A.extent ha)
  refine ⟨v,he,⟨?_,?_⟩,hf⟩
  · intro c e j hc hek hjv
    exact (ho.history c e j hc ⟨lt_of_lt_of_le hek hk,lt_of_lt_of_le hjv hv⟩).trans
      (HostStaging.closed_eq p l a (x l.head) T K V gateK hasSeed ht c _ _ hc.le hek hjv)
  · intro hon e j hek hjv
    exact (ho.final hon e j ⟨lt_of_lt_of_le hek hk,lt_of_lt_of_le hjv hv⟩).trans
      (HostStaging.closed_eq p l a (x l.head) T K V gateK hasSeed ht p.NT _ _ le_rfl hek hjv)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostProgram

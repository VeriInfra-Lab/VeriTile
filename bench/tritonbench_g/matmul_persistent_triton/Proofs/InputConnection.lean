import bench.tritonbench_g.matmul_persistent_triton.Proofs.PublicData

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The existing regional execution theorem does not use physical storage IDs.
Distinct internal IDs encode its two separately named regional input memories;
physical aliases remain covered by the retained Tensor bridge. -/
def isolated (b : RawKernel.Buffers) : RawKernel.Buffers :=
  ⟨{b.a with storage := 0},{b.b with storage := 1},{b.c with storage := 2}⟩

theorem isolated_layout (p : Core.Parameters) (b : RawKernel.Buffers) (h : RawKernel.Layout p b) :
    RawKernel.Layout p (isolated b) := by
  refine ⟨?_,?_,?_,h.outputStrides,by simp [isolated],by simp [isolated]⟩
  · exact ⟨h.aBounds.rowsBound,h.aBounds.columnsBound,h.aBounds.rowStrideBound,
      h.aBounds.columnStrideBound,h.aBounds.offsetBound,h.aBounds.span,h.aBounds.capacity,
      h.aBounds.physical,h.aBounds.pointer⟩
  · exact ⟨h.bBounds.rowsBound,h.bBounds.columnsBound,h.bBounds.rowStrideBound,
      h.bBounds.columnStrideBound,h.bBounds.offsetBound,h.bBounds.span,h.bBounds.capacity,
      h.bBounds.physical,h.bBounds.pointer⟩
  · exact ⟨h.cBounds.rowsBound,h.cBounds.columnsBound,h.cBounds.rowStrideBound,
      h.cBounds.columnStrideBound,h.cBounds.offsetBound,h.cBounds.span,h.cBounds.capacity,
      h.cBounds.physical,h.cBounds.pointer⟩

def entry (p : Core.Parameters) (b : RawKernel.Buffers) (s : State α) : Memory α :=
  fun a => if a.1=0 then Core.readCell p.problem.profile (s.mem "A" (a.2-b.a.offset))
    else Core.readCell p.problem.profile (s.mem "B" (a.2-b.b.offset))

theorem entry_inputs (p : Core.Parameters) (b : RawKernel.Buffers) (s : State α) :
    RawKernel.Inputs p (isolated b) s (entry p b s) := by
  constructor <;> intro i <;> simp [isolated,entry]

theorem input_a (P : Source.Arithmetic α α) (p : Core.Parameters)
    (xs : (io P p).Inputs) (s : State α) (hin : (io P p).InputBinding xs s)
    (r : Fin p.problem.M) (k : Fin p.problem.K) :
    Core.readCell p.problem.profile (s.mem "A" (r.val*p.am+k.val*p.ak))=
      payload p.problem.profile (xs 0 (r,k,PUnit.unit)) := by
  have h := congrArg (payload p.problem.profile) (hin 0 (r,k,PUnit.unit) trivial)
  change payload p.problem.profile ((s.mem "A" (r.val*p.am+k.val*p.ak)).read (Core.dtype p.problem.profile))=
    payload p.problem.profile (mapValue id (Core.dtype p.problem.profile) (xs 0 (r,k,PUnit.unit))) at h
  simpa only [payload_read,payload_map] using h

theorem input_b (P : Source.Arithmetic α α) (p : Core.Parameters)
    (xs : (io P p).Inputs) (s : State α) (hin : (io P p).InputBinding xs s)
    (k : Fin p.problem.K) (c : Fin p.problem.N) :
    Core.readCell p.problem.profile (s.mem "B" (k.val*p.bk+c.val*p.bn))=
      payload p.problem.profile (xs 1 (k,c,PUnit.unit)) := by
  have h := congrArg (payload p.problem.profile) (hin 1 (k,c,PUnit.unit) trivial)
  change payload p.problem.profile ((s.mem "B" (k.val*p.bk+c.val*p.bn)).read (Core.dtype p.problem.profile))=
    payload p.problem.profile (mapValue id (Core.dtype p.problem.profile) (xs 1 (k,c,PUnit.unit))) at h
  simpa only [payload_read,payload_map] using h

theorem left_connection (P Q : Source.Arithmetic α α) (p : Core.Parameters)
    (b : RawKernel.Buffers) (xs : (io P p).Inputs) (s : State α)
    (hin : (io P p).InputBinding xs s) (hm : 0<p.problem.M) (rb q : Nat) :
    Targets.left Q.inputZero p.problem ((isolated b).descriptors p) (entry p b s) rb q=
      left Q.inputZero p xs rb q := by
  funext r k
  have hr := clamp_bound p.problem.M (rb*128+r.val) hm
  by_cases hk : q*p.problem.profile.blockK+k.val<p.problem.K
  · simp only [Targets.left,left,dif_pos hr,dif_pos hk,if_pos hk]
    simpa [Tensor.address,RawKernel.Buffers.descriptors,RawKernel.Allocation.descriptor,
      isolated,entry,Nat.add_assoc] using
      input_a P p xs s hin ⟨_,hr⟩ ⟨_,hk⟩
  · simp [Targets.left,left,hr,hk]

theorem right_connection (P Q : Source.Arithmetic α α) (p : Core.Parameters)
    (b : RawKernel.Buffers) (xs : (io P p).Inputs) (s : State α)
    (hin : (io P p).InputBinding xs s) (hn : 0<p.problem.N) (cb q : Nat) :
    Targets.right Q.inputZero p.problem ((isolated b).descriptors p) (entry p b s) cb q=
      right Q.inputZero p xs cb q := by
  funext k c
  have hc := clamp_bound p.problem.N (cb*256+c.val) hn
  by_cases hk : q*p.problem.profile.blockK+k.val<p.problem.K
  · simp only [Targets.right,right,dif_pos hc,dif_pos hk,if_pos hk]
    simpa [Tensor.address,RawKernel.Buffers.descriptors,RawKernel.Allocation.descriptor,
      isolated,entry,Nat.add_assoc] using
      input_b P p xs s hin ⟨_,hk⟩ ⟨_,hc⟩
  · simp [Targets.right,right,hc,hk]

theorem accumulator_connection (P Q : Source.Arithmetic α α) (p : Core.Parameters)
    (b : RawKernel.Buffers) (xs : (io P p).Inputs) (s : State α)
    (hin : (io P p).InputBinding xs s) (hm : 0<p.problem.M) (hn : 0<p.problem.N)
    (rb cb q : Nat) :
    Targets.accReference Q p.problem ((isolated b).descriptors p) (entry p b s) rb cb q=
      accumulator Q p xs rb cb q := by
  induction q with
  | zero => rfl
  | succ q ih =>
    simp only [Targets.accReference,accumulator,ih,left_connection P Q p b xs s hin hm,
      right_connection P Q p b xs s hin hn]

theorem result_connection (P Q : Source.Arithmetic α α) (p : Core.Parameters)
    (b : RawKernel.Buffers) (xs : (io P p).Inputs) (s : State α)
    (hin : (io P p).InputBinding xs s) (j : TileIndex [p.problem.M,p.problem.N]) :
    Core.pack p.problem.profile (RawKernel.reference Q p (isolated b) (entry p b s) j.1.val j.2.1.val)=
      Cell.mk (Core.dtype p.problem.profile) (result Q p xs 0 j) := by
  have hm : 0<p.problem.M := lt_of_le_of_lt (Nat.zero_le _) j.1.isLt
  have hn : 0<p.problem.N := lt_of_le_of_lt (Nat.zero_le _) j.2.1.isLt
  change Core.pack p.problem.profile (RawKernel.reference Q p (isolated b) (entry p b s) j.1.val j.2.1.val)=
    Cell.mk (Core.dtype p.problem.profile) (value p.problem.profile
      (Q.cast p.problem.profile (accumulator Q p xs (j.1.val/128) (j.2.1.val/256) p.problem.kTiles _ _)))
  rw [pack_value,RawKernel.reference,Targets.result,accumulator_connection P Q p b xs s hin hm hn]

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData

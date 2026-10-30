import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Execution

namespace VeriTile.Bench.Repairs.L2NormBwd.Machine
open Physical Host Integer
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Index (signed_ofNat)
noncomputable section
set_option maxHeartbeats 4000000

structure Address (m : Matrix) (r c : Nat) : Prop where
  pid : (BitVec.ofNat 32 r).toNat=r
  rowStride : (BitVec.ofNat 64 m.rowStride).toInt=(m.rowStride:Int)
  colStride : (BitVec.ofNat 64 m.colStride).toInt=(m.colStride:Int)
  rowMul : (rowProduct r m.rowStride).toInt=(r*m.rowStride:Nat)
  colMul : (BitVec.ofNat 64 c*BitVec.ofNat 64 m.colStride).toInt=(c*m.colStride:Nat)
  pointer : (Integer.pointer m r c).toNat=(m.key r c).address
  bytes : ∀ byte,byte<4 → (Integer.pointer m r c).toNat+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
    (Integer.pointer m r c).toNat+byte<2^64

/-- Only the launched row number needs the PID bound. A longer DY backing
may have arbitrarily many extra rows that this call never accesses. -/
theorem active_role (m : Matrix) (metadata : Metadata m) (backed : m.backed)
    (r c : Nat) (pid : r<2^32) (hr : r<m.rows) (hc : c<m.cols) : Address m r c := by
  obtain ⟨_,lower,upper⟩ := backed r c hr hc
  have capacity := m.view.storage.representable
  simp only [Matrix.key,Matrix.index,Allocation.key] at lower upper
  have rmul : r*m.rowStride<2^63 := by omega
  have cmul : c*m.colStride<2^63 := by omega
  have arith : m.view.storage.baseByte+4*m.view.storageOffset+r*m.rowStride*4+c*m.colStride*4=
      m.view.storage.baseByte+4*(m.view.storageOffset+r*m.rowStride+c*m.colStride) := by ring
  have exactPointer : (Integer.pointer m r c).toNat=(m.key r c).address := by
    unfold Integer.pointer rowProduct
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_mul,
      ←BitVec.ofNat_add,←BitVec.ofNat_add,BitVec.toNat_ofNat,arith,Nat.mod_eq_of_lt (by omega)]
    rfl
  refine ⟨by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt pid],
    signed_ofNat 64 _ (by decide) metadata.row,signed_ofNat 64 _ (by decide) metadata.col,
    ?_,?_,exactPointer,?_⟩
  · rw [rowProduct,←BitVec.ofNat_mul];exact signed_ofNat 64 _ (by decide) rmul
  · rw [←BitVec.ofNat_mul];exact signed_ofNat 64 _ (by decide) cmul
  · intro byte small
    rw [exactPointer]
    simp only [Matrix.key,Matrix.index,Allocation.key]
    omega

/-- Explicit global launcher compatibility boundary, separated from every
local integer/address theorem and every numerical primitive obligation. -/
def GridContract {M : Type} (rt : Runtime M) : Prop :=
  ∀ p queue before after,rt.perform (.setup p queue) before=.ok () after → GridCompatible p.x.rows

def LaunchSafe {M α β : Type} {ops : Ops M α β} {red : Reduction α} {rt : Runtime M}
    {q : Request} {eps : α} {before after : World M} (path : BodyPath ops red rt q eps before after) : Prop :=
  (∀ r c,r<path.layout.x.rows → c<path.layout.x.cols →
    Address path.layout.x r c ∧ Address path.layout.dy r c ∧ Address path.layout.dx r c) ∧
  ∀ c,c<path.layout.block →
    (BitVec.ofNat 64 c).slt (BitVec.ofNat 64 path.layout.x.cols)=decide (c<path.layout.x.cols)

theorem launch_safe {M α β : Type} {ops : Ops M α β} {red : Reduction α} {rt : Runtime M}
    {q : Request} {eps : α} {before after : World M} (grids : GridContract rt)
    (path : BodyPath ops red rt q eps before after) : LaunchSafe path := by
  have grid := grids path.layout path.launch.queue path.launch path.launch path.setup
  have valid : Valid path.layout := path.valid
  refine ⟨?_,?_⟩
  · intro r c hr hc
    have pid : r<2^32 := by unfold GridCompatible at grid;omega
    refine ⟨active_role path.layout.x path.xMetadata valid.xBacked r c pid hr hc,?_,?_⟩
    · exact active_role path.layout.dy path.dyMetadata valid.dyBacked r c pid
        (lt_of_lt_of_le hr valid.dyRows) (by rwa [valid.dyCols])
    · exact active_role path.layout.dx path.dxMetadata valid.dxBacked r c pid
        (by rwa [valid.dxRows]) (by rwa [valid.dxCols])
  · intro c hc
    exact Integer.mask_exact _ _ _ valid.block path.blockBound hc

def Placement {M α β : Type} {ops : Ops M α β} {red : Reduction α} {rt : Runtime M}
    {q : Request} {eps : α} {before after : World M} (path : BodyPath ops red rt q eps before after) : Prop :=
  path.launch.queue.owner=path.x.view.storage.owner ∧
  path.dy.view.storage.owner=path.x.view.storage.owner ∧
  q.dx.view.storage.owner=path.x.view.storage.owner ∧
  path.launch.queue.stream=path.launch.streams path.x.view.storage.owner

/-- The actual context selects the resolved X/DY and actual emptyLike device;
the actual launch uses that device's current stream. -/
theorem placement {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (eps : α) (initial after : World M)
    (output : View) (domain : InputDomain call initial)
    (path : WholePath ops red rt call eps initial output after) : Placement path.context.body := by
  have norm := normalize_spec ops neg rt lib call initial domain
  rw [path.normalized] at norm
  change Normalized ops neg call initial path.request path.prepared at norm
  let b := path.context.body
  obtain ⟨_,_,_,copyX,_⟩ := lib.resolve _ _ _ _ b.resolvedX
  obtain ⟨_,_,_,copyY,_⟩ := lib.resolve _ _ _ _ b.resolvedDY
  have dx : b.afterX.device=path.context.inside.device := by
    simpa only [b.resolvedX,Result.state] using lib.device (.resolve path.request.x) path.context.inside
  have dy : b.launch.device=b.afterX.device := by
    simpa only [b.resolvedDY,Result.state] using lib.device (.resolve path.request.dy) b.afterX
  have entered := (lib.enter _ _ _ _ path.context.entered).2.2
  have actual : b.launch.device=b.x.view.storage.owner :=
    (dy.trans (dx.trans entered)).trans copyX.owner.symm
  refine ⟨actual,copyY.owner.trans (norm.sameDevice.symm.trans copyX.owner.symm),
    norm.dxDevice.trans copyX.owner.symm,?_⟩
  change b.launch.streams b.launch.device=b.launch.streams b.x.view.storage.owner
  rw [actual]

#specNonCircular GridContract avoiding [Physical.run,Physical.value,Tile.scalar,source]
#axiomsClean active_role
#axiomsClean launch_safe
#axiomsClean placement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Machine

import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Host
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.FlatStorage

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.HostStorage
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary HostModel Storage
noncomputable section
set_option maxHeartbeats 3000000

variable {α : Type} {call : Call α} {entry : PhysicalMemory α} {old : Nat → Prop}
  {g : HostModel.Geometry call} {logical : Logical α}
  {a : AllocationResult α call g.host entry old}

/-- Same-call ordinary tensor metadata and lifetime. Bounds quantify over
logical tensor coordinates and output storage, not kernel results or native
integer terms. Descriptors are the actual prepared inputs and fresh outputs. -/
structure Backings (p : Prepared call g logical a) where
  allocation : Nat → Allocation
  live : Nat → Prop
  /-- The registry contains every caller and retained input allocation, not
  only the descriptors named by the selected launch. -/
  keptLive : ∀ allocation, p.kept allocation → live allocation
  /-- Ordinary same-call allocator separation at actual native byte ranges.
  Equal allocation IDs retain all permitted input aliases. -/
  separated : Storage.Separated allocation live
  dt : Storage.Tensor allocation
  rate : Storage.Tensor allocation
  bias : Storage.Tensor allocation
  out : Storage.Tensor allocation
  sum : Storage.Tensor allocation
  dtView : dt.view = p.dt.view
  rateView : rate.view = p.rate.view
  biasView : bias.view = (p.bias.map RawBoundary.Tensor.view).getD p.dt.view
  outView : out.view = a.out
  sumView : sum.view = a.sum
  dtShape : dt.shape = p.dt.shape
  rateShape : rate.shape = p.rate.shape
  dtStride : dt.stride = p.dt.stride
  rateStride : rate.stride = p.rate.stride
  dtDevice : dt.device = p.dt.device
  rateDevice : rate.device = p.rate.device
  biasDevice : ∀ t, p.bias = some t → bias.device = t.device
  biasMetadata : ∀ t, p.bias = some t → bias.shape = t.shape ∧ bias.stride = t.stride
  outShape : out.shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk]
  sumShape : sum.shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk]
  outPacked : out.stride 0 = p.host.layout.outBatch ∧ out.stride 1 = p.host.layout.outHead ∧
    out.stride 2 = p.host.layout.outChunk ∧ out.stride 3 = 1
  sumPacked : sum.stride 0 = p.host.layout.sumBatch ∧ sum.stride 1 = p.host.layout.sumHead ∧
    sum.stride 2 = p.host.layout.sumChunk ∧ sum.stride 3 = 1
  outObject : out.object = a.outObject
  sumObject : sum.object = a.sumObject
  outDevice : out.device = call.dt.device
  sumDevice : sum.device = call.dt.device
  dtLive : live dt.view.allocation
  rateLive : live rate.view.allocation
  biasLive : call.hasBias = Bool.true → live bias.view.allocation
  outLive : live out.view.allocation
  sumLive : live sum.view.allocation
  dtCovers : ∀ b t h, b < g.host.batches → t < g.host.sequence → h < g.host.heads →
    dt.Covers (b*p.dt.stride 0+t*p.dt.stride 1+h*p.dt.stride 2)
  rateCovers : ∀ h, h < g.host.heads → rate.Covers (h*p.rate.stride 0)
  biasCovers : call.hasBias = Bool.true → ∀ h, h < g.host.heads → bias.Covers (h*p.host.biasHead)
  outCovers : ∀ off, off < g.host.outputSize → out.Covers off
  sumCovers : ∀ off, off < g.host.outputSize → sum.Covers off

variable {p : Prepared call g logical a}

theorem prepared_sizes (p : Prepared call g logical a) (h : Native.Sizes g.host) :
    Native.Sizes p.host := ⟨h.sequence,h.heads,h.chunk⟩

theorem Backings.callerLive (b : Backings p) (allocation : Nat) (caller : old allocation) :
    b.live allocation := b.keptLive allocation (p.callerProtected allocation caller)

theorem Backings.output_bytes_disjoint (b : Backings p) (i j : Nat)
    (hi : i < (b.allocation b.out.view.allocation).bytes)
    (hj : j < (b.allocation b.sum.view.allocation).bytes)
    (device : (b.allocation b.out.view.allocation).device = (b.allocation b.sum.view.allocation).device) :
    (b.allocation b.out.view.allocation).address+i ≠ (b.allocation b.sum.view.allocation).address+j := by
  intro collision
  have same := b.separated _ _ i j b.outLive b.sumLive hi hj device collision
  exact a.distinct (by simpa only [b.outView,b.sumView] using same)

theorem Backings.output_kept_disjoint (b : Backings p) (output caller i j : Nat)
    (selected : output = b.out.view.allocation ∨ output = b.sum.view.allocation)
    (kept : p.kept caller) (hi : i < (b.allocation output).bytes)
    (hj : j < (b.allocation caller).bytes)
    (device : (b.allocation output).device = (b.allocation caller).device) :
    (b.allocation output).address+i ≠ (b.allocation caller).address+j := by
  intro collision
  rcases selected with rfl | rfl
  · have same := b.separated _ _ i j b.outLive (b.keptLive caller kept) hi hj device collision
    have held : p.kept b.out.view.allocation := same.symm ▸ kept
    apply p.outFresh
    simpa only [b.outView] using held
  · have same := b.separated _ _ i j b.sumLive (b.keptLive caller kept) hi hj device collision
    have held : p.kept b.sum.view.allocation := same.symm ▸ kept
    apply p.sumFresh
    simpa only [b.sumView] using held

/-- Any prefix of actual output-byte stores preserves every original caller
byte at its native address, including partial trailing words. Payloads are
unrestricted: their numerical meaning is irrelevant to this frame proof. -/
theorem Backings.output_native_prefix_frame (b : Backings p)
    (events : List (Storage.ByteWrite b.allocation b.live))
    (targets : ∀ e ∈ events, e.owner = b.out.view.allocation ∨ e.owner = b.sum.view.allocation)
    (memory : Storage.ByteHeap) (flat : Storage.FlatHeap)
    (rep : Storage.RepresentsBytes b.allocation b.live memory flat)
    (n caller byte : Nat) (held : old caller) (bounded : byte < (b.allocation caller).bytes) :
    ((events.take n).foldl (fun m e => e.native m) flat)
      (b.allocation caller).device ((b.allocation caller).address+byte) =
        flat (b.allocation caller).device ((b.allocation caller).address+byte) := by
  apply Storage.byte_trace_native_prefix_frame b.separated events _ b.callerLive rep n caller byte held bounded
  intro e member olde
  rcases targets e member with out | sum
  · exact a.outFresh (by simpa only [out,b.outView] using olde)
  · exact a.sumFresh (by simpa only [sum,b.sumView] using olde)

theorem Backings.dt_bound (b : Backings p) (cfg : HeadBlock)
    (l : Launch) (hl : (reblock p.host cfg).valid l) (i j : Nat)
    (active : Math.activeInput (reblock p.host cfg).layout l i j) :
    b.dt.Covers (Math.dtAddress (reblock p.host cfg).layout l i j) := by
  have start := Whole.chunk_start_lt (reblock p.host cfg) l hl
  have time : l.chunk*g.host.chunk+j < g.host.sequence := by
    have ht := active.2
    change j < min g.host.chunk (g.host.sequence-l.chunk*g.host.chunk) at ht
    change l.chunk*g.host.chunk < g.host.sequence at start
    omega
  have bound := b.dtCovers l.batch (l.chunk*g.host.chunk+j)
    (Math.head (reblock p.host cfg).layout l i) hl.1 time active.1
  convert bound using 1
  simp [Math.dtAddress,Whole.Host.layout,reblock,Prepared.host]
  ring

theorem Backings.out_bound (b : Backings p) (cfg : HeadBlock)
    (l : Launch) (hl : (reblock p.host cfg).valid l) (i j : Nat)
    (active : Math.activeOutput (reblock p.host cfg).layout l i j) :
    b.out.Covers (Math.outAddress (reblock p.host cfg).layout l i j) ∧
    b.sum.Covers (Math.outAddress (reblock p.host cfg).layout l i j) := by
  have bound := Whole.output_bound (reblock p.host cfg) l hl i j active
  exact ⟨b.outCovers _ bound,b.sumCovers _ bound⟩

def Span (b : Storage.Tensor allocation) (off : Nat) (word : BitVec 64) : Prop :=
  (Index.address b.dataPtr word).toNat = b.dataPtr+4*off ∧
  b.dataPtr+4*off+4 ≤ (allocation b.view.allocation).address+(allocation b.view.allocation).bytes ∧
  b.dataPtr+4*off+4 ≤ 2^64

def Facts (b : Backings p) : Prop :=
  ∀ cfg : HeadBlock, ∀ l, (reblock p.host cfg).valid l →
    ∀ (i : Fin cfg.size) (j : Fin p.host.BC),
      let c := reblock p.host cfg
      ((Index.head c l i.val).ult (Index.u c.heads) = decide (Math.head c.layout l i.val < c.heads)) ∧
      ((Index.u j.val).ult (Index.u c.chunk) = decide (j.val < c.chunk)) ∧
      ((Index.u j.val).toNat < min (Index.u c.chunk).toNat (Index.remaining c l).toNat ↔
        j.val < min c.chunk (c.sequence-l.chunk*c.chunk)) ∧
      (Math.activeInput c.layout l i.val j.val → Span b.dt (Math.dtAddress c.layout l i.val j.val) (Index.dt c l i.val j.val)) ∧
      (Math.head c.layout l i.val < c.heads →
        Span b.rate (Math.head c.layout l i.val*c.rateHead) (Index.rate c l i.val) ∧
        (call.hasBias = Bool.true → Span b.bias (Math.head c.layout l i.val*c.biasHead) (Index.bias c l i.val))) ∧
      (Math.activeOutput c.layout l i.val j.val →
        Span b.out (Math.outAddress c.layout l i.val j.val) (Index.out c l i.val j.val) ∧
        Span b.sum (Math.outAddress c.layout l i.val j.val) (Index.out c l i.val j.val))

theorem actual (b : Backings p) (checkedSizes : Native.Sizes p.host) : Facts b := by
  intro cfg l hl i j
  let c := reblock p.host cfg
  have sizes : Native.Sizes c := ⟨checkedSizes.sequence,checkedSizes.heads,checkedSizes.chunk⟩
  obtain ⟨hm,hj,hr⟩ := Native.masks c sizes l hl i j
  refine ⟨hm,hj,hr,?_,?_,?_⟩
  · intro active
    exact Native.active_dt c l i.val j.val b.dt (b.dt_bound cfg l hl i.val j.val active)
  · intro active
    have rate := b.rateCovers (Math.head c.layout l i.val) active
    refine ⟨?_,?_⟩
    · unfold Span
      rw [Native.rate_word]
      exact ⟨Storage.native_address b.rate _ rate,Storage.span b.rate _ rate⟩
    · intro bias
      have bound := b.biasCovers bias (Math.head c.layout l i.val) active
      unfold Span
      rw [Native.bias_word]
      exact ⟨Storage.native_address b.bias _ bound,Storage.span b.bias _ bound⟩
  · intro active
    obtain ⟨out,sum⟩ := b.out_bound cfg l hl i.val j.val active
    exact ⟨Native.active_out c l i.val j.val b.out out,Native.active_out c l i.val j.val b.sum sum⟩

/-- The descriptors are provided for this exact successful allocation and
direct input observation; every trial and selected launch reuses those arguments. -/
def OrdinaryStorage (rt : HostModel.Runtime call logical entry old) : Type :=
  ∀ g a p, rt.allocate g = .ok a → rt.inputs g a = p → Backings p

/-- The actual widened source output expression, not merely its abstract
offset, reads the same four native bytes through the returned descriptors. -/
theorem Backings.native_output_bytes (b : Backings p) (memory : PhysicalMemory Word)
    (flat : Storage.FlatHeap) (rep : Storage.Represents b.allocation b.live memory flat)
    (cfg : HeadBlock) (l : Launch) (hl : (reblock p.host cfg).valid l) (i j : Nat)
    (active : Math.activeOutput (reblock p.host cfg).layout l i j) (k : Nat) (hk : k < 4) :
    flat b.out.device ((Index.address b.out.dataPtr (Index.out (reblock p.host cfg) l i j)).toNat+k) =
      Storage.byte (b.out.view.read memory (Math.outAddress (reblock p.host cfg).layout l i j)) k ∧
    flat b.sum.device ((Index.address b.sum.dataPtr (Index.out (reblock p.host cfg) l i j)).toNat+k) =
      Storage.byte (b.sum.view.read memory (Math.outAddress (reblock p.host cfg).layout l i j)) k := by
  obtain ⟨outCovered,sumCovered⟩ := b.out_bound cfg l hl i j active
  have outNative := Native.active_out (reblock p.host cfg) l i j b.out outCovered
  have sumNative := Native.active_out (reblock p.host cfg) l i j b.sum sumCovered
  have outBytes := Storage.represented_native_bytes rep b.out b.outLive _ outCovered k hk
  have sumBytes := Storage.represented_native_bytes rep b.sum b.sumLive _ sumCovered k hk
  rw [Storage.native_address b.out _ outCovered] at outBytes
  rw [Storage.native_address b.sum _ sumCovered] at sumBytes
  rw [outNative.1,sumNative.1]
  exact ⟨outBytes,sumBytes⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.HostStorage

import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostSetup
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeSource
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostBlock

/- The source launcher selects q.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
/-! Whole public Real return bound to caller inputs, returned output metadata,
ordinary preparation/allocation facts, ordered setup, and native pointers. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostWrapper
open VeriTile.Triton Memory Relocation PhysicalCells HostSetup
open VeriTile.Bench.Repairs.LayerNormFwd
open Address (Allocation signed64)
open NativeOffsets
open HostShape (blockD blockD_covers OutputDescriptor)
noncomputable section
set_option maxHeartbeats 2000000

def inputValue (xs : RealProofs.Inputs) (r : Role) (z h n d : Nat) : ℝ :=
  match r with
  | .query => xs.query z h n d | .key => xs.key z h n d | .value => xs.value z h n d
  | .queryScale => xs.queryScale n | .keyScale => xs.keyScale n | .output => 0

def outputDescriptor (p : Parameters) (base : Role → Nat) : OutputDescriptor :=
  ⟨p.Out,base .output,p.batches,p.heads,p.context,p.headDim,
    p.outputBatch,p.outputHead,p.outputRow,p.outputDim,.bf16⟩

structure Prepared (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs) where
  stage : p.stage=3
  owned : Fin a.capacity → Prop
  originalCell : Role → Nat → Nat → Nat → Nat → Fin a.capacity
  negative : Role → Bool
  originalOwned : ∀ z h n d r, r≠Role.output → active p z h n d r → owned (originalCell r z h n d)
  originalValues : ∀ z h n d r, r≠Role.output → active p z h n d r →
    inputValue xs r z h n d=(if negative r then -(Host.readReal (caller (originalCell r z h n d)))
      else Host.readReal (caller (originalCell r z h n d)))
  base : Role → Nat
  views : Views a A p base
  returnedOutput : OutputDescriptor
  outputBinding : returnedOutput=outputDescriptor p base
  alloc : Role → Allocation
  bases : ∀ r, (alloc r).base=base r
  allocated : Allocated p alloc
  initial : AliasMemory.Physical a.toLayout
  preparationWrites : Fin a.capacity → Prop
  preparationFrame : ∀ cell, ¬preparationWrites cell → initial cell=caller cell
  preparationFresh : ∀ cell, owned cell → ¬preparationWrites cell
  outputFresh : ∀ cell, owned cell → ∀ k, GridFootprint.window A p k → a.address k≠cell.val
  agree : LogicalAgreement.Agree p (Host.logicalInputs a A p initial template) xs
  bounds : RegionBounds
  windows : ViewBounds.Windows bounds (parameters p A.flat) (offsets A p)
  nonoverlap : OutputLayout.NonOverlapping p
  fresh : FreshBytes a (GridFootprint.window A p)

/-- Active native source arithmetic addresses the exact same physical view. -/
theorem native_address (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) (z h n d : Nat) (r : Role) (hr : active p z h n d r) :
    machine p z h n d r=offset p z h n d r ∧ signed64 (machine p z h n d r)=(offset p z h n d r : Int) ∧
      bytePointer (ret.alloc r).base (width r) (machine p z h n d r)=a.address (key A p z h n d r) := by
  have hx := all_roles p ret.alloc ret.allocated z h n d r hr
  refine ⟨hx.1,hx.2.1,?_⟩
  rw [hx.2.2,ret.views.address z h n d r hr,ret.bases r]
  simp [Nat.mul_comm]

def NativeCertificate (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) : Prop :=
  (∀ z h n d r, active p z h n d r →
    machine p z h n d r=offset p z h n d r ∧ signed64 (machine p z h n d r)=(offset p z h n d r : Int) ∧
    bytePointer (ret.alloc r).base (width r) (machine p z h n d r)=a.address (key A p z h n d r)) ∧
  (∀ z h lane d lower iterations,
    keyValue p.keyBatch p.keyHead p.keyRow p.keyDim z h lane d lower iterations=
      Address.wrap64 (offset p z h (lower+iterations*64+lane) d .key)) ∧
  (∀ z h lane d lower iterations,
    keyValue p.valueBatch p.valueHead p.valueRow p.valueDim z h lane d lower iterations=
      Address.wrap64 (offset p z h (lower+iterations*64+lane) d .value)) ∧
  (∀ head count lower iterations,
    NativeOffsets.advance (packedScale head count (lower/64)) (Address.wrap64 1) iterations=
      Address.wrap64 (head*count+(lower+iterations*64)/64)) ∧
  (NativeControl.CudaLaunchable p (blockD p.headDim) →
    (signed64 p.context=(p.context : Int) ∧ signed64 (p.context+127)=(p.context+127 : Nat) ∧
      signed64 (p.context+63)=(p.context+63 : Nat)) ∧
    (∀ start row, start*128<p.context → row<128 →
      signed64 (start*128+row)=(start*128+row : Nat) ∧
      signed64 ((start+1)*128)=((start+1)*128 : Nat) ∧
      signed64 (min ((start+1)*128) p.context)=(min ((start+1)*128) p.context : Nat)) ∧
    (∀ cursor lane, cursor<p.context → lane<64 →
      signed64 (p.context-cursor)=(p.context : Int)-(cursor : Int) ∧
      signed64 (cursor+lane)=(cursor+lane : Nat) ∧ signed64 (cursor+64)=(cursor+64 : Nat)) ∧
    (∀ lane, lane<blockD p.headDim → signed64 lane=(lane : Int)) ∧
    (∀ head, head<p.batches*p.heads →
      signed64 (head*((p.context+127)/128))=(head*((p.context+127)/128) : Nat) ∧
      signed64 (head*((p.context+63)/64))=(head*((p.context+63)/64) : Nat)))

theorem native_certificate (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) : NativeCertificate p a A caller template xs ret := by
  refine ⟨native_address p a A caller template xs ret,?_,?_,scale_progression,?_⟩
  · intro z h lane d lower iterations
    exact keyValue_mod p.keyBatch p.keyHead p.keyRow p.keyDim z h lane d lower iterations
  · intro z h lane d lower iterations
    exact keyValue_mod p.valueBatch p.valueHead p.valueRow p.valueDim z h lane d lower iterations
  · intro hl
    exact ⟨NativeControl.cdiv_inputs p _ hl,
      fun start row hs hr => NativeControl.row_arithmetic p _ start row hl hs hr,
      fun cursor lane hc hd => NativeControl.key_arithmetic p _ cursor lane hl hc hd,
      fun lane hd => NativeControl.feature_lanes p _ lane hl hd,
      fun head hh => NativeControl.packed_offsets p _ head hl hh⟩

/-- Returned output descriptor, successful ordinary setup, native source
certificate, independent causal formula on original caller coordinates,
all-grid trace bounds and full caller-owned memory preservation. -/
theorem successful_return (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : p.batches*p.heads*p.context*p.headDim≠0)
    (hlaunch : NativeControl.CudaLaunchable p (blockD p.headDim)) (reciprocal : Bool) :
    ret.returnedOutput=outputDescriptor p ret.base ∧
    HostSetup.run p (blockD p.headDim) ok=.launch hne hlaunch ∧
    NativeCertificate p a A caller template xs ret ∧
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p (blockD p.headDim) reciprocal) (GridCoordinates.grid p)
        (AliasMemory.pull a.toLayout ret.initial template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p ret.base ret.views i))=
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val xs) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList ret.bounds (GridLaunch.kernel A p (blockD p.headDim) reciprocal).body
          ((AliasMemory.pull a.toLayout ret.initial template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.owned cell → final cell=caller cell) := by
  refine ⟨ret.outputBinding,HostSetup.successful_setup p (blockD p.headDim) ok hbefore hne hscales hlaunch hcompile,
    native_certificate p a A caller template xs ret,?_⟩
  obtain ⟨t,final,hexec,hread,hsafe,hframe⟩ := Host.launch a A p (blockD p.headDim) ret.base ret.views
    ret.initial template xs ret.agree ret.bounds ret.windows (blockD_covers _) ret.stage reciprocal ret.nonoverlap ret.fresh
  refine ⟨t,final,hexec,hread,hsafe,fun cell hp hw => (hframe cell hw).trans (ret.preparationFrame cell hp),?_⟩
  intro cell hc
  exact (hframe cell (ret.outputFresh cell hc)).trans (ret.preparationFrame cell (ret.preparationFresh cell hc))

theorem empty_return (p : Parameters) (a : Layout) (A : FlatAlloc)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState) (xs : RealProofs.Inputs)
    (ret : Prepared p a A caller template xs) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (he : p.batches*p.heads*p.context*p.headDim=0) :
    ret.returnedOutput=outputDescriptor p ret.base ∧
    HostSetup.run p (blockD p.headDim) ok=.empty he ∧ IsEmpty (OutputLayout.Coordinate p) ∧
    (∀ cell, ret.owned cell → ret.initial cell=caller cell) :=
  ⟨ret.outputBinding,HostSetup.empty_returns_before_launch p _ ok hbefore he,Host.empty_output p he,
    fun cell hc => ret.preparationFrame cell (ret.preparationFresh cell hc)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostWrapper

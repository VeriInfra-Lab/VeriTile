import bench.tritonbench_g.dequantize_matmul.Proofs.EventGrid
import bench.tritonbench_g.dequantize_matmul.Proofs.JIT
import bench.tritonbench_g.dequantize_matmul.Kernels

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Typed strided, resolved, contiguous FP32 Tensor descriptor. Dynamic supplied
output metadata is admitted by the direct OutputContract; fresh values are
created by the FP32 FreshAllocation interface. -/
structure Descriptor where
  object : Nat
  location : Location
  rows : Nat
  columns : Nat
  device : Nat
  strides : Dense2 rows columns

/-- Exactly the Python's `if out is None` descriptor selection. -/
def selectOutput (provided : Option Descriptor) (fresh : Descriptor) : Descriptor :=
  provided.getD fresh

/-- Two local allocation results: optional c, then fresh fp_b. Allocation
preserves every caller-owned cell; it does not constrain computation results. -/
structure Allocated (s : Shape) (provided : Option Descriptor) (fresh : Descriptor)
    (b : Buffers) (before after : Memory (Cell α)) where
  caller : Nat → Prop
  selectedLocation : (selectOutput provided fresh).location=b.out
  selectedRows : (selectOutput provided fresh).rows=s.M
  selectedColumns : (selectOutput provided fresh).columns=s.N
  scratchFresh : ∀ storage,caller storage → storage≠b.fpb.storage
  freshOutput : provided=none → ∀ storage,caller storage → storage≠b.out.storage
  preserved : ∀ p,caller p.1 → after p=before p
  separate : Separate b
  inputA : caller b.a.storage
  inputB : caller b.b.storage
  inputScale : caller b.scale.storage
  outputA : b.a.storage≠b.out.storage
  outputB : b.b.storage≠b.out.storage
  outputScale : b.scale.storage≠b.out.storage

structure HostReturn (α : Type) where
  tensor : Descriptor
  memory : Memory (Cell α)

def hostRun (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (provided : Option Descriptor) (fresh : Descriptor)
    (allocated : Memory (Cell α)) : HostReturn α :=
  ⟨selectOutput provided fresh,run a s b tile plan allocated⟩

theorem returned_provided (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (provided fresh : Descriptor) (allocated : Memory (Cell α)) :
    (hostRun a s b tile plan (some provided) fresh allocated).tensor=provided := rfl

theorem returned_fresh (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (fresh : Descriptor) (allocated : Memory (Cell α)) :
    (hostRun a s b tile plan none fresh allocated).tensor=fresh := rfl

theorem allocation_represents {before after : Memory (Cell α)} (allocation : Allocated s provided fresh b before after)
    (x : Input α s) (rep : Represents x b before) : Represents x b after := by
  refine ⟨?_,?_,?_⟩
  · intro r k
    rw [allocation.preserved _ allocation.inputA,rep.a]
  · intro k n
    rw [allocation.preserved _ allocation.inputB,rep.b]
  · intro n
    rw [allocation.preserved _ allocation.inputScale,rep.scale]

theorem caller_frame (a : Arithmetic α) (tile : Tile) (plan : LibraryPlan s)
    (allocation : Allocated s provided fresh b before after) (point : Address)
    (caller : allocation.caller point.1) (outside : ¬(b.out.window s.M s.N).inside point) :
    (hostRun a s b tile plan provided fresh after).memory point=before point := by
  have scratch : ¬(b.fpb.window s.K s.N).inside point := by
    intro h
    exact allocation.scratchFresh point.1 caller h.1
  exact (frame a s b tile plan allocation.separate after point outside scratch).trans
    (allocation.preserved point caller)

theorem fresh_caller_frame (a : Arithmetic α) (tile : Tile) (plan : LibraryPlan s)
    (allocation : Allocated s none fresh b before after) (point : Address)
    (caller : allocation.caller point.1) :
    (hostRun a s b tile plan none fresh after).memory point=before point := by
  apply caller_frame a tile plan allocation point caller
  intro h
  exact allocation.freshOutput rfl point.1 caller h.1

theorem host_real (tile : Tile) (plan : LibraryPlan s)
    (allocation : Allocated s provided fresh b before after)
    (x : Input ℝ s) (rep : Represents x b before) (r : Fin s.M) (n : Fin s.N) :
    (hostRun realArithmetic s b tile plan provided fresh after).memory
      ((hostRun realArithmetic s b tile plan provided fresh after).tensor.location.addr (r.val*s.N+n.val))=
        .fp32 (mathematical x r n) := by
  simp only [hostRun,allocation.selectedLocation]
  exact real_complete x b tile plan allocation.separate after
    (allocation_represents allocation x rep) r n

/-- Static typed storage encoding. For any machine state and physical caller
byte, owner/decode identify its cell and byte projection. This is an ABI
representation law, not a relation asserted only for a final execution. -/
structure ByteRepresentation (abi : LocalEventABI α σ a) where
  bytes : σ → Address → BitVec 8
  owner : Address → Address
  decode : Cell α → Address → BitVec 8
  represented : ∀ state byte,bytes state byte=decode (abi.observe state (owner byte)) byte

def Window.byteInside (w : Window) (byte : Address) : Prop :=
  byte.1=w.storage ∧ 4*w.offset≤byte.2 ∧ byte.2<4*(w.offset+w.rows*w.columns)

structure ByteWindows (rep : ByteRepresentation abi) (s : Shape) (b : Buffers) : Prop where
  output : ∀ byte,(b.out.window s.M s.N).inside (rep.owner byte) →
    (b.out.window s.M s.N).byteInside byte
  scratch : ∀ byte,(b.fpb.window s.K s.N).inside (rep.owner byte) →
    byte.1=b.fpb.storage

/-- Physical caller-byte frame is derived from local event execution and a
static codec; prefix/suffix bytes are included by the exact output window. -/
theorem native_byte_frame (abi : LocalEventABI α σ a) (rep : ByteRepresentation abi)
    (windows : ByteWindows rep s b) (tile : Tile) (plan : LibraryPlan s)
    (sep : Separate b) (state : σ) (byte : Address)
    (outside : ¬(b.out.window s.M s.N).byteInside byte)
    (scratchFresh : byte.1≠b.fpb.storage) :
    rep.bytes ((pipelineCode s b plan).native abi state).2 byte=rep.bytes state byte := by
  rw [rep.represented,rep.represented,native_pipeline abi s b tile plan sep state]
  rw [frame a s b tile plan sep (abi.observe state) (rep.owner byte)
    (fun h => outside (windows.output byte h))
    (fun h => scratchFresh (windows.scratch byte h))]

/-- Native host result, from locally bound primitive occurrences. -/
theorem host_native_words (native : Arithmetic Word) (instruction : FixedInstructionProfile)
    (numeric : NumericLeaves native instruction) (abi : LocalEventABI Word σ native)
    (tile : Tile) (plan : LibraryPlan s) (state : σ)
    (allocation : Allocated s provided fresh b before (abi.observe state))
    (x : Input Word s) (rep : Represents x b before) (r : Fin s.M) (n : Fin s.N) :
    abi.observe ((pipelineCode s b plan).native abi state).2
      ((selectOutput provided fresh).location.addr (r.val*s.N+n.val))=
        .fp32 (wordReference (fpArithmetic instruction) x plan r n) := by
  rw [allocation.selectedLocation,native_pipeline abi s b tile plan allocation.separate state]
  exact fp_complete native instruction numeric x b tile plan allocation.separate
    (abi.observe state) (allocation_represents allocation x rep) r n

#axiomsClean returned_provided
#axiomsClean returned_fresh
#axiomsClean caller_frame
#axiomsClean fresh_caller_frame
#axiomsClean host_real
#axiomsClean native_byte_frame
#axiomsClean host_native_words
end
end VeriTile.Bench.Repairs.DequantizeMatmul

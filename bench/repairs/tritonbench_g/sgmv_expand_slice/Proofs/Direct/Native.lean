import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Common
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativePrepared
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Physical

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.NativeCall
open VeriTile.Triton FP.Structural Contract
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000
namespace Raw
abbrev State := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.State
abbrev Word := BitVec 32
abbrev read := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.read
abbrev unpack := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.unpack
abbrev ops := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.ops
abbrev byte_of_read := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.Raw.byte_of_read
end Raw
abbrev Window := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.TensorWindow
abbrev ByteAllocation := VeriTile.Bench.Repairs.SwigluTriton.Memory.Allocation

/-- Ordinary original-view and live-allocation bindings. Read/read aliases are
allowed. The singleton squeeze is an alias with unchanged pointer/backing. -/
structure Binding (t : Inputs) where
  input : Window t.x 4
  weight : Window t.w 2
  output : Window t.y 4
  starts : Window t.starts (indexBytes t.starts)
  lengths : Window t.lengths (indexBytes t.lengths)
  indices : Window t.indices (indexBytes t.indices)
  squeezed : TensorInfo
  squeezeShape : squeezed.shape=[rows t.w,width t.w,rank t.w]
  squeezePointer : squeezed.dataPtr=t.w.dataPtr
  squeezeBacking : squeezed.backing=t.w.backing
  inputSeparate : t.x.backing≠t.y.backing → input.backing.separate output.backing
  weightSeparate : t.w.backing≠t.y.backing → weight.backing.separate output.backing
  startsSeparate : t.starts.backing≠t.y.backing → starts.backing.separate output.backing
  lengthsSeparate : t.lengths.backing≠t.y.backing → lengths.backing.separate output.backing
  indicesSeparate : t.indices.backing≠t.y.backing → indices.backing.separate output.backing

def signed (memory : Raw.State) (tensor : TensorInfo) (window : ByteAllocation) (b : Nat) : Int :=
  let address:=window.baseByte+indexBytes tensor*b
  if tensor.dtype=.other "int64" then
    (Raw.read memory ⟨window.owner,address+4⟩ ++ Raw.read memory ⟨window.owner,address⟩).toInt
  else (Raw.read memory ⟨window.owner,address⟩).toInt

/-- The local tolist API observes the original signed-width metadata bytes. -/
def lists (a : Binding t) (memory : Raw.State) : Lists t where
  starts := (List.range t.starts.numel).map (signed memory t.starts a.starts.window)
  lengths := (List.range t.lengths.numel).map (signed memory t.lengths a.lengths.window)
  indices := (List.range t.indices.numel).map (signed memory t.indices a.indices.window)
  startCount := by simp
  lengthCount := by simp
  indexCount := by simp

def input (t : Inputs) (s : Scalars) (a : Binding t) (memory : Raw.State) :
    Native.Inputs (NativeBackend.profile (abi t s)) where
  x := fun r k => Raw.read memory (a.input.window.key (r*rank t.w+k))
  weight := fun slot c k =>
    let address:=a.weight.window.baseByte+2*((slot*width t.w+c)*rank t.w+k)
    memory a.weight.window.owner (address+1) ++ memory a.weight.window.owner address

def entry (t : Inputs) (s : Scalars) (a : Binding t) (memory : Raw.State) :
    Nat→Nat→Native.Word (NativeBackend.scalar (abi t s).output) :=
  fun r c => Raw.read memory (a.output.window.key (r*columns t.y+c))
def initial (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  NativeBackend.prepared (abi t s) (shape t s) (input t s a before) (entry t s a before)
    (raw t s (lists a before)) ((lists a before).lengths.drop s.batches.toNat)
def execution (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  runSourceJobs (SourceBody.install (NativeBackend.algebra rest ops (NativeBackend.profile (abi t s)))
    (NativeBackend.dot ops .half)) (abi t s) (args t s)
    (jobs (shape t s).columns (segments t s (lists a before))) (initial t s a before)
def finalState (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  (execution rest ops t s a before).getD (initial t s a before)
def reference (ops : Native.Primitives) (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) :=
  Native.result ops (NativeBackend.profile (abi t s)) (shape t s)
    (segments t s (lists a before)) (input t s a before) (entry t s a before)

theorem complete (rest : Algebra Raw.Word) (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before)) :
    execution rest ops t s a before=some (finalState rest ops t s a before) ∧
    SourceSuffix.OutputMemory (abi t s) (args t s) (shape t s) (finalState rest ops t s a before)
      (reference ops t s a before) ∧
    SourceGrid.ReadFrame (args t s) (initial t s a before) (finalState rest ops t s a before) := by
  obtain ⟨final,run,read,frame⟩:=NativeBackend.prepared_result rest ops zeros (abi t s) (shape t s)
    (strides (shape t s)) (input t s a before) (entry t s a before)
    (raw t s (lists a before)) ((lists a before).lengths.drop s.batches.toNat)
    (segments t s (lists a before)) (validated ok) (xlayout _) (wlayout _) (ylayout _) (capacity ok)
  have success : execution rest ops t s a before=some final := run
  have finalEq : finalState rest ops t s a before=final := by simp [finalState,success]
  rw [finalEq]
  refine ⟨success,?_,frame⟩
  simpa only [NativeBackend.pack,abi,NativeBackend.scalar,BitVec.setWidth_eq] using read

def key (a : Binding t) (s : Scalars) (cell : Nat×Nat) := a.output.window.key (offset (shape t s) cell)
def value (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (cell : Nat×Nat) :=
  SourceSuffix.readOutput (abi t s) (args t s) (finalState rest ops t s a before) cell.1 cell.2

/-- Physical interpretation of the completed source's scalar stores. The
payload is read from actual final source memory, not the reference formula.
This introduces no Python copyback or fresh allocation. -/
def sourceStores (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) : Raw.State :=
  stores Raw.ops (key a s) (value rest ops t s a before)
    (cells (shape t s) (segments t s (lists a before))) before

theorem source_read (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before))
    (cell : Nat×Nat) (live : Touched (shape t s) (segments t s (lists a before)) cell) :
    Raw.read (sourceStores rest ops t s a before) (key a s cell)=value rest ops t s a before cell := by
  apply readback Raw.ops _ _ _ before cell (touched_complete ok cell live)
  · intro other member equal
    have h:=(mem_cells _ _ _).mp member
    have hc:=touched_inside ok cell live
    have address:=congrArg Key.address equal
    have same : offset (shape t s) other=offset (shape t s) cell := by simp only [key,Allocation.key] at address;omega
    have coordinates:=linear_injective h.2.1 hc.2 same
    cases other;cases cell;simp_all [value]
  · intro other member different
    apply Allocation.indices_disjoint
    intro same
    exact different (congrArg a.output.window.key same.symm)

theorem source_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (owner address : Nat)
    (outside : ∀ cell,Touched (shape t s) (segments t s (lists a before)) cell → ¬(key a s cell).covers owner address) :
    sourceStores rest ops t s a before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro cell member
  exact outside cell ((mem_cells _ _ _).mp member).2.2

theorem whole (rest : Algebra Raw.Word) (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before)) :
    execution rest ops t s a before=some (finalState rest ops t s a before) ∧
    (∀ cell,Touched (shape t s) (segments t s (lists a before)) cell → ∀ byte : Fin 4,
      sourceStores rest ops t s a before a.output.window.owner ((key a s cell).address+byte.val)=
        Raw.unpack (reference ops t s a before cell.1 cell.2) byte) ∧
    (∀ owner address,(∀ cell,Touched (shape t s) (segments t s (lists a before)) cell →
      ¬(key a s cell).covers owner address) → sourceStores rest ops t s a before owner address=before owner address) := by
  have result:=complete rest ops zeros t s a before ok
  refine ⟨result.1,?_,source_frame rest ops t s a before⟩
  intro cell live byte
  have inside:=touched_inside ok cell live
  calc
    _=Raw.unpack (Raw.read (sourceStores rest ops t s a before) (key a s cell)) byte :=
      Raw.byte_of_read _ (key a s cell) byte
    _=_ := by rw [source_read rest ops t s a before ok cell live];exact congrArg (fun word => Raw.unpack word byte) (result.2.1 _ _ inside.1 inside.2)

#axiomsClean complete
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean whole
#specNonCircular reference avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.NativeCall

import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Common
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativePrepared
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Binding

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
open VeriTile.Triton FP.Structural Contract
open Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000
def signed (memory : Raw.State) (tensor : TensorInfo) (b : Nat) : Int :=
  let address:=tensor.dataPtr.toNat+indexBytes tensor*b
  if tensor.dtype=.other "int64" then
    (Raw.read memory ⟨tensor.device.index,address+4⟩ ++ Raw.read memory ⟨tensor.device.index,address⟩).toInt
  else (Raw.read memory ⟨tensor.device.index,address⟩).toInt

/-- The local tolist API observes the original signed-width metadata bytes. -/
def lists (t : Inputs) (memory : Raw.State) : Lists t where
  starts := (List.range t.starts.numel).map (signed memory t.starts)
  lengths := (List.range t.lengths.numel).map (signed memory t.lengths)
  indices := (List.range t.indices.numel).map (signed memory t.indices)
  startCount := by simp
  lengthCount := by simp
  indexCount := by simp

def input (t : Inputs) (s : Scalars) (memory : Raw.State) :
    Native.Inputs (NativeBackend.profile (abi t s)) where
  x := fun r k => Raw.read memory (tensorKey t.x (r*rank t.w+k))
  weight := fun slot c k =>
    let address:=t.w.dataPtr.toNat+2*((slot*width t.w+c)*rank t.w+k)
    memory t.w.device.index (address+1) ++ memory t.w.device.index address

def entry (t : Inputs) (s : Scalars) (memory : Raw.State) :
    Nat→Nat→Native.Word (NativeBackend.scalar (abi t s).output) :=
  fun r c => Raw.read memory (tensorKey t.y (r*columns t.y+c))
def initial (t : Inputs) (s : Scalars) (before : Raw.State) :=
  NativeBackend.prepared (abi t s) (shape t s) (input t s before) (entry t s before)
    (raw t s (lists t before)) ((lists t before).lengths.drop s.batches.toNat)
def execution (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  runSourceJobs (SourceBody.install (NativeBackend.algebra rest ops (NativeBackend.profile (abi t s)))
    (NativeBackend.dot ops .half)) (abi t s) (args t s)
    (jobs (shape t s).columns (segments t s (lists t before))) (initial t s before)
def finalState (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) :=
  (execution rest ops t s before).getD (initial t s before)
def reference (ops : Native.Primitives) (t : Inputs) (s : Scalars) (before : Raw.State) :=
  Native.result ops (NativeBackend.profile (abi t s)) (shape t s)
    (segments t s (lists t before)) (input t s before) (entry t s before)

theorem complete (rest : Algebra Raw.Word) (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (lists t before)) :
    execution rest ops t s before=some (finalState rest ops t s before) ∧
    SourceSuffix.OutputMemory (abi t s) (args t s) (shape t s) (finalState rest ops t s before)
      (reference ops t s before) ∧
    SourceGrid.ReadFrame (args t s) (initial t s before) (finalState rest ops t s before) := by
  obtain ⟨final,run,read,frame⟩:=NativeBackend.prepared_result rest ops zeros (abi t s) (shape t s)
    (strides (shape t s)) (input t s before) (entry t s before)
    (raw t s (lists t before)) ((lists t before).lengths.drop s.batches.toNat)
    (segments t s (lists t before)) (validated ok) (xlayout _) (wlayout _) (ylayout _) (capacity ok)
  have success : execution rest ops t s before=some final := run
  have finalEq : finalState rest ops t s before=final := by simp [finalState,success]
  rw [finalEq]
  refine ⟨success,?_,frame⟩
  simpa only [NativeBackend.pack,abi,NativeBackend.scalar,BitVec.setWidth_eq] using read

def key (t : Inputs) (s : Scalars) (cell : Nat×Nat) := tensorKey t.y (offset (shape t s) cell)
def value (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) (cell : Nat×Nat) :=
  SourceSuffix.readOutput (abi t s) (args t s) (finalState rest ops t s before) cell.1 cell.2

/-- Physical interpretation of the completed source's scalar stores. The
payload is read from actual final source memory, not the reference formula.
This introduces no Python copyback or fresh allocation. -/
def sourceStores (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) : Raw.State :=
  stores Raw.ops (key t s) (value rest ops t s before)
    (cells (shape t s) (segments t s (lists t before))) before

theorem source_read (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (lists t before))
    (cell : Nat×Nat) (live : Touched (shape t s) (segments t s (lists t before)) cell) :
    Raw.read (sourceStores rest ops t s before) (key t s cell)=value rest ops t s before cell := by
  apply readback Raw.ops _ _ _ before cell (touched_complete ok cell live)
  · intro other member equal
    have h:=(mem_cells _ _ _).mp member
    have hc:=touched_inside ok cell live
    have address:=congrArg Key.address equal
    have same : offset (shape t s) other=offset (shape t s) cell := by simp only [key,tensorKey] at address;omega
    have coordinates:=linear_injective h.2.1 hc.2 same
    cases other;cases cell;simp_all [value]
  · intro other member different
    have neq : offset (shape t s) cell≠offset (shape t s) other := by
      intro same
      exact different (congrArg (tensorKey t.y) same.symm)
    simp only [key,tensorKey,Key.disjoint]
    omega

theorem source_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State) (owner address : Nat)
    (outside : ∀ cell,Touched (shape t s) (segments t s (lists t before)) cell → ¬(key t s cell).covers owner address) :
    sourceStores rest ops t s before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro cell member
  exact outside cell ((mem_cells _ _ _).mp member).2.2

theorem whole (rest : Algebra Raw.Word) (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (lists t before)) :
    execution rest ops t s before=some (finalState rest ops t s before) ∧
    (∀ cell,Touched (shape t s) (segments t s (lists t before)) cell → ∀ byte : Fin 4,
      sourceStores rest ops t s before t.y.device.index ((key t s cell).address+byte.val)=
        Raw.unpack (reference ops t s before cell.1 cell.2) byte) ∧
    (∀ owner address,(∀ cell,Touched (shape t s) (segments t s (lists t before)) cell →
      ¬(key t s cell).covers owner address) → sourceStores rest ops t s before owner address=before owner address) := by
  have result:=complete rest ops zeros t s before ok
  refine ⟨result.1,?_,source_frame rest ops t s before⟩
  intro cell live byte
  have inside:=touched_inside ok cell live
  calc
    _=Raw.unpack (Raw.read (sourceStores rest ops t s before) (key t s cell)) byte :=
      Raw.byte_of_read _ (key t s cell) byte
    _=_ := by rw [source_read rest ops t s before ok cell live];exact congrArg (fun word => Raw.unpack word byte) (result.2.1 _ _ inside.1 inside.2)

#axiomsClean complete
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean whole
#specNonCircular reference avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated

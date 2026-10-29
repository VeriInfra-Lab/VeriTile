import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Reify
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.NativeScale

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.NativeExecution
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def encode (word : BitVec 32) : Native.Expr := fun _=>Native.Stage.ready (.bits word)
def prepare (c : Config) (layout : Reify.Layout c) (payload : Raw.Payload (BitVec 32)) :=
  Reify.prepare c layout encode BitVec.toInt Native.zeroExpr payload
def observed (c : Config) (payload : Raw.Payload (BitVec 32)) := Reify.observed c BitVec.toInt payload

/-- Execute the complete parsed typed source, including fresh zero allocation,
both dynamic loops, casts, and every source store, then read its returned cell. -/
def source (runtime : Native.Runtime) (actual : Native.CompareISA (BitVec 32))
    (c : Config) (layout : Reify.Layout c) (payload : Raw.Payload (BitVec 32)) (q : Raw.Coord c) : BitVec 32 :=
  match Physical.invoke (fun _=>Native.algebra runtime) (fun _=>Native.dotModel c.storage actual)
      c Native.zeroExpr (prepare c layout payload) (programs (grid c)) with
  | none=>0
  | some result=>if hi : Raw.offset q<result.memory.capacity c.out then
      Native.readNative actual c (result.memory.cells c.out ⟨Raw.offset q,hi⟩) (Raw.dim q)
    else 0

/-- Initial-word native recurrence. This definition does not execute source
statements, read returned memory, or call the implementation. -/
def target (runtime : Native.Runtime) (standard : Native.CompareISA (BitVec 32))
    (c : Config) (layout : Reify.Layout c) (payload : Raw.Payload (BitVec 32)) (q : Raw.Coord c) : BitVec 32 :=
  if hbm : 0<c.BM then
    (Native.logicalGraph runtime standard c (prepare c layout payload).extend hbm
      (Raw.hz q) (Raw.row q) (Raw.dim q) q.2.2.isLt).eval standard.toISA
  else 0

theorem source_target (runtime : Native.Runtime) (actual standard : Native.CompareISA (BitVec 32))
    (laws : Native.ComparisonTransport id actual standard) (c : Config) (layout : Reify.Layout c)
    (request : Request) (payload : Raw.Payload (BitVec 32))
    (accepted : Admission c request (observed c payload)) (q : Raw.Coord c) :
    source runtime actual c layout payload q=target runtime standard c layout payload q := by
  have h:=to_contract accepted (Reify.binding c request layout encode BitVec.toInt Native.zeroExpr payload)
  obtain ⟨result,hr,hdesc,hv⟩:=Native.returned_local_transport runtime id actual standard laws c
    (prepare c layout payload) h.fresh h.separated h.shape.contiguous h.metadata.sequence
    (bm_pos h) (bn_pos h) (programs (grid c)) (List.Perm.refl _)
  obtain ⟨i,hi,hval⟩:=hv (Raw.hz q) q.1.isLt (Raw.row q) q.2.1.isLt (Raw.dim q) q.2.2.isLt
  change i.val=Raw.offset q at hi
  have fits : Raw.offset q<result.memory.capacity c.out := by rw [←hi];exact i.isLt
  have index : (⟨Raw.offset q,fits⟩ : Fin (result.memory.capacity c.out))=i := Fin.ext hi.symm
  simp only [source,hr,target,dif_pos fits,dif_pos (bm_pos h)]
  simpa only [index,id_eq] using hval

#axiomsClean source_target
#specNonCircular target avoiding [source,Physical.invoke,Whole.invoke,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.NativeExecution

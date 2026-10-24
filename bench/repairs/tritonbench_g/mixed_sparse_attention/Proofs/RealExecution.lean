import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Reify
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.RealExecution
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def prepare (c : Config) (layout : Reify.Layout c) (integer : ℝ → Int) (payload : Raw.Payload ℝ) :=
  Reify.prepare c layout (some : ℝ → Real.R) integer (some 0) payload
def observed (c : Config) (integer : ℝ → Int) (payload : Raw.Payload ℝ) := Reify.observed c integer payload
def values (c : Config) (layout : Reify.Layout c) (payload : Raw.Payload ℝ) (r : RegionName) (i : Nat) : ℝ :=
  match layout.route r with
  | some role=>if i<role.size c then payload role i else 0
  | none=>0

theorem reads (c : Config) (layout : Reify.Layout c) (integer : ℝ → Int) (payload : Raw.Payload ℝ) :
    Real.Reads c (prepare c layout integer payload).extend (values c layout payload) := by
  intro r hr i
  have hq:=layout.inputs .query (by simp [inputRoles])
  have hk:=layout.inputs .key (by simp [inputRoles])
  have hv:=layout.inputs .value (by simp [inputRoles])
  simp only [Role.region] at hq hk hv
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl
  all_goals simp only [prepare,Reify.prepare,Reify.capacity,Physical.Store.extend,values,hq,hk,hv,
    Reify.cell,Role.floating,ite_true,ofFloat]
  all_goals split_ifs <;> rfl

def source (c : Config) (layout : Reify.Layout c) (integer : ℝ → Int)
    (payload : Raw.Payload ℝ) (q : Raw.Coord c) : ℝ :=
  match Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel)
      c (some 0) (prepare c layout integer payload) (programs (grid c)) with
  | none=>0
  | some result=>if hi : Raw.offset q<result.memory.capacity c.out then
      ((result.memory.cells c.out ⟨Raw.offset q,hi⟩).read .real).unbotD 0
    else 0

def target (c : Config) (layout : Reify.Layout c) (integer : ℝ → Int)
    (payload : Raw.Payload ℝ) (q : Raw.Coord c) : ℝ :=
  Real.logicalExpected c (prepare c layout integer payload).extend (values c layout payload)
    (Raw.hz q) (Raw.row q) (Raw.dim q)

theorem source_target (c : Config) (layout : Reify.Layout c) (integer : ℝ → Int)
    (request : Request) (payload : Raw.Payload ℝ)
    (accepted : Admission c request (observed c integer payload)) (q : Raw.Coord c) :
    source c layout integer payload q=target c layout integer payload q := by
  have h:=to_contract accepted (Reify.binding c request layout (some : ℝ→Real.R) integer (some 0) payload)
  obtain ⟨result,hr,hdesc,hcap,hv,hframe⟩:=Real.returned_finite c (prepare c layout integer payload)
    h.fresh (values c layout payload) (reads c layout integer payload) h.separated h.shape.contiguous
    h.metadata.sequence (bm_pos h) (bn_pos h) (programs (grid c)) (List.Perm.refl _)
  obtain ⟨i,hi,hval⟩:=hv (Raw.hz q) q.1.isLt (Raw.row q) q.2.1.isLt (Raw.dim q) q.2.2.isLt
  change i.val=Raw.offset q at hi
  have fits : Raw.offset q<result.memory.capacity c.out := by rw [←hi];exact i.isLt
  have index : (⟨Raw.offset q,fits⟩ : Fin (result.memory.capacity c.out))=i := Fin.ext hi.symm
  simp only [source,hr,dif_pos fits,index,hval]
  rfl

#axiomsClean reads
#axiomsClean source_target
#specNonCircular target avoiding [source,Physical.invoke,Whole.invoke,FP.Structural.exec,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.RealExecution

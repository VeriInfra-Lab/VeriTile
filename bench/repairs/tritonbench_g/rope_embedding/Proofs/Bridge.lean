import bench.repairs.tritonbench_g.rope_embedding.Proofs.Contract
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def parameters (r : Request) : Source.Parameters :=
  ⟨"Q","C","S",r.heads*r.dimension,r.half,r.half,r.sequence,r.dimension,r.heads,r.block⟩
def programState {α : Type} (s : State α) (a g : Nat) : State α :=
  {s with pids:=fun axisId => if axisId=0 then a else if axisId=1 then g else s.pids axisId}

theorem source_offset (r : Request) (a h : Nat) (side : Bool) (i : Lane r) :
    Source.offset (parameters r) a h side i=r.offset a h (i.1.val+if side then r.half else 0) := by
  simp only [Source.offset,parameters,Request.offset,Request.half]
  ring

theorem scalar_slot (op : Operations α) (r : Request) (backward : Bool) (q c s : Nat→α)
    (a h j : Nat) (hj : j<r.half) (side : Bool) :
    scalarResult op r backward q c s a h (j+if side then r.half else 0)=
      scalarPair op backward (q (r.offset a h j)) (q (r.offset a h (j+r.half)))
        (c (a%r.sequence*r.half+j)) (s (a%r.sequence*r.half+j)) side := by
  cases side
  · have h1 : j<2*r.half := by omega
    have h2 : ¬r.half≤j := by omega
    simp [scalarResult,hj,h1,h2]
  · have h1 : j+r.half<2*r.half := by omega
    have h2 : ¬j+r.half<r.half := by omega
    simp [scalarResult,h1,h2]

def sourceValue {α : Type} [Inhabited α] (op : Operations α) (r : Request) (backward : Bool)
    (s : State α) (l : Location r) : α :=
  let p := parameters r
  let u := programState s (row l) (group l)
  Source.value op.backend p (row l) (head l)
    (Source.coefficient op.backend p u p.cosine p.cosineStride)
    (fun i => if backward then op.backend.unary (some .fp32) (.neg .real)
      (Source.coefficient op.backend p u p.sine p.sineStride i)
      else Source.coefficient op.backend p u p.sine p.sineStride i)
    u (side l) (lane l)

theorem source_value {α : Type} [Inhabited α] (op : Operations α) (r : Request) (backward : Bool)
    (s : State α) (l : Location r) (ha : active l) :
    sourceValue op r backward s l=scalarResult op r backward
      (fun n => (s.mem "Q" n).read .real) (fun n => (s.mem "C" n).read .real)
      (fun n => (s.mem "S" n).read .real) (row l) (head l) (column l) := by
  rw [column,scalar_slot op r backward _ _ _ _ _ _ ha.2.2]
  unfold sourceValue Source.value
  simp only [source_offset]
  have hi : (lane l).1.val<r.dimension/2 := ha.2.2
  simp only [Source.coefficient,↓reduceIte,programState,parameters,Operations.backend,scalarPair]
  cases side l <;> cases backward <;> simp [Source.active,hi,Request.half]

theorem source_execution {α : Type} [Inhabited α] (op : Operations α) (r : Request) (backward : Bool)
    (s : State α) (l : Location r) (hl : l∈schedule r) :
    ∃ final,FP.Structural.exec op.backend (Source.source (parameters r) backward)
      (programState s (row l) (group l))=some final ∧
      (active l → final.mem "Q" (r.offset (row l) (head l) (column l))=
        .mk .real (scalarResult op r backward (fun n => (s.mem "Q" n).read .real)
          (fun n => (s.mem "C" n).read .real) (fun n => (s.mem "S" n).read .real)
          (row l) (head l) (column l))) := by
  have hs := (mem_schedule r l).mp hl
  have hg : (programState s (row l) (group l)).pids 1*4≤(parameters r).heads := by
    simpa [programState,parameters] using Nat.le_of_lt (group_head r (group l) hs.2.1)
  obtain ⟨final,he,hv,hf⟩ := Source.complete_correctness op.backend (parameters r) backward
    (programState s (row l) (group l)) hg
  refine ⟨final,he,?_⟩
  intro ha
  have h := hv (head l) (by simpa [programState] using hs.2.2.1)
    (by simpa [programState,parameters] using hs.2.2.2) (side l) (lane l) ha.2.2
  have h' : final.mem "Q" (r.offset (row l) (head l) (column l))=
      .mk .real (sourceValue op r backward s l) := by
    simpa only [programState,show (0:Nat)=0 from rfl,↓reduceIte,source_offset,column,sourceValue] using h
  rw [source_value op r backward s l ha] at h'
  exact h'

theorem group_coordinate (r : Request) (l : Location r) (hl : l∈schedule r) : head l/4=group l := by
  have h := (mem_schedule r l).mp hl
  have hu := lt_of_lt_of_le h.2.2.2 (min_le_left _ _)
  omega

#axiomsClean source_value
#axiomsClean source_execution
#axiomsClean group_coordinate
end
end VeriTile.Bench.Repairs.RopeEmbedding

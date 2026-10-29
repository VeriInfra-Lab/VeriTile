import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ScaledWordLogical
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GridExecution
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Geometry
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Triton FP.Structural
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def geometry (p : Parameters) : Geometry.Layout :=
  ⟨p.rows,p.columns,p.blockRows,p.blockColumns,p.groupRows,p.outputRowStride,p.outputColumnStride⟩
abbrev LogicalIndex (p : Parameters) := Geometry.Cell (geometry p)

theorem active_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    active p (atProgram s pid) i ↔ Geometry.Active (geometry p) pid i := by rfl

theorem offset_at (p : Parameters) (s : State Word) (pid : Nat) (i : CIndex p) :
    outputOffset p (atProgram s pid) i=Geometry.offset (geometry p) pid i := by
  change Geometry.outColumn (geometry p) pid i*p.outputColumnStride+
    Geometry.outRow (geometry p) pid i*p.outputRowStride=_
  simp only [Geometry.offset,geometry,Nat.add_comm]

theorem logical_preserved (p : Parameters) (initial current : State Word)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hout : Outside p.output initial current) (r c : Nat) :
    logicalDot p current r c=logicalDot p initial r c := by
  unfold logicalDot
  congr 1
  apply congrArg (fun f => LogicalSum.sum f p.inner)
  funext k
  unfold logicalProduct
  rw [hout p.input _ ha,hout p.weight _ hb]

theorem scaled_preserved (cfg : FP.Config) (p : Parameters) (initial current : State Word)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (hout : Outside p.output initial current) (r c : Nat) :
    logicalScaled cfg p current r c=logicalScaled cfg p initial r c := by
  unfold logicalScaled
  rw [logical_preserved p initial current ha hb hout,hout p.scale _ hs]

theorem result_preserved (cfg : FP.Config) (p : Parameters) (initial current : State Word)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (hout : Outside p.output initial current) (r c : Nat) :
    logicalResult cfg p current r c=logicalResult cfg p initial r c := by
  unfold logicalResult
  rw [scaled_preserved cfg p initial current ha hb hs hout]

def one (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (pid : Nat) (s : State Word) :=
  FP.Structural.exec (backend rest cfg u) (source p) (atProgram s pid)
def grid (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) :=
  GridExecution.sequence (one rest cfg u p) (List.range (Geometry.count (geometry p))) s

theorem single_logical (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) (hdefined : LogicalDefined cfg p initial) (current : State Word) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one rest cfg u p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  have hd : LogicalDefined cfg p s := by
    intro r c hr hc
    rw [scaled_preserved cfg p initial s ha hb hscale hout]
    exact hdefined r c hr hc
  obtain ⟨final,he,hread,hframe⟩ := correctness rest cfg u p s
    (inputDefined_of_logical cfg p s heven hcover hd) hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,expected_logical cfg p s heven hcover i hi]
    change Cell.mk .int (logicalResult cfg p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,result_preserved cfg p initial s ha hb hscale hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : outputOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid i
      rw [ho]
      exact h x hx

theorem grid_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) (hdefined : LogicalDefined cfg p initial) :
    ∃ final, grid rest cfg u p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one rest cfg u p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single_logical rest cfg u p hgeo heven hcover ha hb hscale initial hdefined current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean active_at
#axiomsClean offset_at
#axiomsClean logical_preserved
#axiomsClean scaled_preserved
#axiomsClean result_preserved
#axiomsClean single_logical
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord

import bench.tritonbench_g.rotary_emb.Proofs.HostExecution
import Mathlib.Data.List.Nodup
import Mathlib.Data.List.ProdSigma

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile Triton FullMath
noncomputable section

/-- A concrete finite schedule containing every source-grid program once.
The composition theorem also supports every permutation of this schedule. -/
def Plan.schedule (c : Plan) : List c.PID :=
  (List.finRange c.headBlocks).product (List.finRange c.seqBlocks)

theorem schedule_nodup (c : Plan) : c.schedule.Nodup :=
  (List.nodup_finRange _).product (List.nodup_finRange _)

@[simp] theorem schedule_mem (c : Plan) (b : c.PID) : b ∈ c.schedule := by
  rcases b with ⟨bh,bs⟩
  simp [Plan.schedule, List.mem_finRange]

/-- Logical output cells visited by the HEAD_Q-derived source grid.
In particular, this does not promise to visit every K head. -/
def Plan.covered (c : Plan) (x : Bool) (n h d : Nat) : Prop :=
  n < c.request.q.length ∧ h < (c.tensor x).heads ∧
  h < c.headBlocks*4 ∧ d < c.evenWidth

theorem q_heads_covered (c : Plan) (h : Nat) (hh : h < c.request.q.heads) :
    h < c.headBlocks*4 := by
  unfold Plan.headBlocks
  omega

theorem coordinates_cover (c : Plan) (x : Bool) (n h d : Nat)
    (hc : c.covered x n h d) :
    ∃ b : c.PID, ∃ i : c.Lane, ∃ odd : Bool,
      c.active x b i ∧ c.seq b i = n ∧ c.head b i = h ∧
      FullMath.dimension odd i = d := by
  have hbh : h/4 < c.headBlocks := by have hh := hc.2.2.1; omega
  have hbs : n/16 < c.seqBlocks := by
    have hn := hc.1
    unfold Plan.seqBlocks
    omega
  have hpair : d/2 < c.parameters.dimension/2 := by
    have hd := hc.2.2.2
    unfold Plan.evenWidth at hd
    omega
  let b : c.PID := (⟨h/4,hbh⟩,⟨n/16,hbs⟩)
  let i : c.Lane := (⟨n%16,by change n%16<16; omega⟩,
    ⟨h%4,by change h%4<4; omega⟩,⟨d/2,hpair⟩,PUnit.unit)
  have hn : c.seq b i = n := by dsimp [Plan.seq,b,i]; omega
  have hh : c.head b i = h := by dsimp [Plan.head,b,i]; omega
  refine ⟨b,i,decide (d%2=1),?_,hn,hh,?_⟩
  · exact ⟨hn ▸ hc.1,hh ▸ hc.2.1⟩
  · dsimp [FullMath.dimension,i]
    split_ifs with he <;> simp_all <;> omega

/-- Independent logical result, indexed by tensor coordinates and initial
memory alone, using the source's different even/odd trigonometric cells. -/
def Plan.logicalResult (c : Plan) (Q K Cos Sin : RegionName) (s : BlockState)
    (x : Bool) (n h d : Nat) : ℝ :=
  let v0 := s.readMem (region Q K x) ((c.tensor x).address n h (d/2*2))
  let v1 := s.readMem (region Q K x) ((c.tensor x).address n h (d/2*2+1))
  let cv := s.readMem Cos (n*c.request.cos.batchStride+d*c.request.cos.dimStride)
  let sv := s.readMem Sin (n*c.request.cos.batchStride+d*c.request.cos.dimStride)
  if d%2=1 then v0*sv+v1*cv else v0*cv-v1*sv

theorem result_logical (c : Plan) (Q K Cos Sin : RegionName) (s : BlockState)
    (x : Bool) (b : c.PID) (odd : Bool) (i : c.Lane) :
    c.result Q K Cos Sin s x b odd i =
      c.logicalResult Q K Cos Sin s x (c.seq b i) (c.head b i) (FullMath.dimension odd i) := by
  cases odd <;>
    simp [Plan.result,Plan.logicalResult,Plan.offset,Plan.trig,FullMath.dimension,
      Nat.add_div, Nat.add_mod]

/-- Exact characterization of the complete launch's write footprint. -/
theorem footprint_covered (c : Plan) (Q K : RegionName) (a : RegionName × Nat) :
    c.foot Q K c.schedule a ↔
      ∃ x n h d, c.covered x n h d ∧ a=(region Q K x,(c.tensor x).address n h d) := by
  constructor
  · rintro ⟨b,_,x,i,o,hi,ha⟩
    exact ⟨x,c.seq b i,c.head b i,FullMath.dimension o i,
      ⟨hi.1,hi.2,head_bound c b i,dim_bound c o i⟩,ha⟩
  · rintro ⟨x,n,h,d,hc,ha⟩
    obtain ⟨b,i,o,hi,hn,hh,hd⟩ := coordinates_cover c x n h d hc
    exact ⟨b,schedule_mem c b,x,i,o,hi,by simpa [Plan.offset,hn,hh,hd] using ha⟩

/-- An unvisited logical cell cannot alias a visited cell under the explicit
tensor-layout injectivity and region-separation premises. -/
theorem untouched_outside (c : Plan) {bounds Q K Cos Sin}
    (D : Domain c bounds Q K Cos Sin) (x : Bool) (n h d : Nat)
    (hn : n < (c.tensor x).length) (hh : h < (c.tensor x).heads)
    (hd : d < (c.tensor x).width)
    (hout : c.headBlocks*4 ≤ h ∨ c.evenWidth ≤ d) :
    ¬c.foot Q K c.schedule (region Q K x,(c.tensor x).address n h d) := by
  rw [footprint_covered]
  rintro ⟨y,n',h',d',hc,he⟩
  have hr := congrArg Prod.fst he
  have heq : x=y := by
    cases x <;> cases y
    · rfl
    · exact False.elim (D.separate.1 hr)
    · exact False.elim (D.separate.1 hr.symm)
    · rfl
  subst y
  have ha := congrArg Prod.snd he
  have hh' := hc.2.2.1
  have hd' := hc.2.2.2
  have hcoord := D.injective x n h d n' h' d' hn hh hd
    (by simpa [tensor_length c D x] using hc.1) hc.2.1
    (lt_of_lt_of_le hc.2.2.2 (D.width x)) ha
  rcases hcoord with ⟨_,rfl,rfl⟩
  omega

/-- Complete source execution in one shared final state, with values referring
to the memory before the entire launch and a raw-cell frame outside its writes. -/
theorem grid_correct (c : Plan) (q k cos sin : FullSource.Storage) (Q K Cos Sin : RegionName)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (D : Domain c A.extent Q K Cos Sin) (hQr : Q∈A.regions) (hKr : K∈A.regions)
    (s : BlockState) :
    c.SafeRun (c.kernel q k cos sin Q K Cos Sin) A.extent c.schedule s ∧
    ∃ t, c.run (c.kernel q k cos sin Q K Cos Sin) c.schedule s = some t ∧
      (∀ x n h d, c.covered x n h d →
        t.readMem (region Q K x) ((c.tensor x).address n h d) =
          c.logicalResult Q K Cos Sin s x n h d) ∧
      BlockState.WriteWithin (c.foot Q K c.schedule) s t ∧
      (∀ x n h d, n < (c.tensor x).length → h < (c.tensor x).heads →
        d < (c.tensor x).width → c.headBlocks*4 ≤ h ∨ c.evenWidth ≤ d →
        t.mem (region Q K x) ((c.tensor x).address n h d) =
          s.mem (region Q K x) ((c.tensor x).address n h d)) := by
  obtain ⟨hsafe,t,ht,hr,hf⟩ := run_correct c q k cos sin Q K Cos Sin
    A hd hcov D hQr hKr c.schedule (schedule_nodup c) s
  refine ⟨hsafe,t,ht,?_,hf,?_⟩
  · intro x n h d hc
    obtain ⟨b,i,o,hi,hn,hh,hd⟩ := coordinates_cover c x n h d hc
    have hout := (hr b (schedule_mem c b) x o i hi).trans
      (result_logical c Q K Cos Sin s x b o i)
    simpa [Plan.offset,hn,hh,hd] using hout
  · intro x n h d hn hh hd hout
    exact (hf _ _ (untouched_outside c D x n h d hn hh hd hout)).symm

#specNonCircular Plan.logicalResult avoiding [Plan.run,Plan.kernel,exec]
#axiomsClean schedule_nodup
#axiomsClean schedule_mem
#axiomsClean q_heads_covered
#axiomsClean coordinates_cover
#axiomsClean result_logical
#axiomsClean footprint_covered
#axiomsClean untouched_outside
#axiomsClean grid_correct

end
end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid

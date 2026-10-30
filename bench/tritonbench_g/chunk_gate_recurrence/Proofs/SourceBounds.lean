import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Interval

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine

def BExpr.fixed (x : Int) : BExpr := .lit (Interval.singleton x)

/-- Constant-size interval evaluation mirrors _source_ranges. It checks all
intermediate signed32 nodes without iterating over tensor elements. -/
def sourceRanges (c : Shape) (s : SeedStrides) : List BExpr :=
  let n := BExpr.fixed c.N
  let k := BExpr.fixed c.K
  let v := BExpr.fixed c.V
  let head := BExpr.lit ⟨0,(c.heads : Int)-1⟩
  let tk := BExpr.lit ⟨0,(c.KT : Int)-1⟩
  let tv := BExpr.lit ⟨0,(c.VT : Int)-1⟩
  let e := BExpr.lit ⟨0,63⟩
  let j := BExpr.lit ⟨0,15⟩
  let plane := BExpr.mul k v
  let feature := BExpr.add (.add (.add (.mul (.mul tk v) (.fixed 64)) (.mul e v))
    (.mul tv (.fixed 16))) j
  let main := BExpr.add (.mul (.mul (.mul head n) k) v) feature
  let step := BExpr.lit ⟨0,(c.N : Int)-1⟩
  let tiles := BExpr.mul (.div k (.fixed 64)) (.div v (.fixed 16))
  let dg := BExpr.add (.mul (.mul head n) tiles) (.add (.mul tk (.div v (.fixed 16))) tv)
  let seed := BExpr.add (.add (.add (.mul (.div head (.fixed c.H)) (.fixed s.batch))
    (.mul (.mod head (.fixed c.H)) (.fixed s.head)))
    (.mul (.add (.mul tk (.fixed 64)) e) (.fixed s.key)))
    (.mul (.add (.mul tv (.fixed 16)) j) (.fixed s.value))
  [n,k,v,.fixed c.H,.fixed c.KT,.fixed c.VT,plane,main,seed,
   .add main (.mul step plane),
   .add main (.mul (.add step (.fixed 1)) plane),
   .add main (.mul (.sub (.sub n (.fixed 2)) step) plane),
   .add main (.mul (.sub (.sub n (.fixed 1)) step) plane),
   .add dg (.mul (.sub (.sub n (.fixed 2)) step) tiles),
   .add (.mul head n) step,
   .add (.mul head n) (.sub (.sub n (.fixed 1)) step),
   .add (.mul (.mul head k) v) feature]

set_option maxHeartbeats 1200000 in
theorem source_ranges_bind (c : Shape) (s : SeedStrides) (h tk tv e j count : Nat)
    (hh : h < c.heads) (hk : tk < c.KT) (hv : tv < c.VT) (he : e < 64) (hj : j < 16) (hc : count < c.N) :
    List.Forall₂ BExpr.Binds (sourceRanges c s) (sourceTrees c s h tk tv e j count) := by
  dsimp only [sourceRanges,sourceTrees,BExpr.fixed]
  repeat' first
    | apply List.Forall₂.cons
    | exact List.Forall₂.nil
    | apply BExpr.Binds.add
    | apply BExpr.Binds.sub
    | apply BExpr.Binds.mul
    | apply BExpr.Binds.div
    | apply BExpr.Binds.mod
    | apply BExpr.Binds.lit
  all_goals simp only [Interval.Contains,Interval.singleton]; omega

theorem bound_member {as : List BExpr} {es : List Expr} (related : List.Forall₂ BExpr.Binds as es)
    (e : Expr) (member : e∈es) : ∃ a,a∈as ∧ a.Binds e := by
  induction related with
  | nil => simp at member
  | @cons a e' as es relation rest ih =>
    rcases List.mem_cons.mp member with same|tail
    · subst e
      exact ⟨a,by simp,relation⟩
    · obtain ⟨b,hb,bind⟩ := ih tail
      exact ⟨b,by simp [hb],bind⟩

def rangeCheck (c : Shape) (s : SeedStrides) : Bool := (sourceRanges c s).all BExpr.valid

theorem source_valid (c : Shape) (s : SeedStrides) (accepted : rangeCheck c s=Bool.true)
    (h tk tv e j count : Nat) (hh : h < c.heads) (hk : tk < c.KT) (hv : tv < c.VT)
    (he : e < 64) (hj : j < 16) (hc : count < c.N)
    (x : Expr) (member : x∈sourceTrees c s h tk tv e j count) : x.Valid := by
  obtain ⟨a,ha,bind⟩ := bound_member (source_ranges_bind c s h tk tv e j count hh hk hv he hj hc) x member
  exact (BExpr.certified a x bind (List.all_eq_true.mp accepted a ha)).1


def pointerIndices (seed : Bool) (r : Region) : List Nat := match r with
  | .x => [7,9]
  | .output => [7,10,11]
  | .incoming => [12]
  | .dx => [11]
  | .tileGate => [13]
  | .gate => [14,15]
  | .dseed => [16]
  | .seed => if seed then [8] else []
  | .dgate | .caller => []

def allRegions : List Region := [.x,.gate,.seed,.incoming,.output,.dx,.tileGate,.dseed,.dgate,.caller]

def pointerCheck (c : Shape) (s : SeedStrides) (seed : Bool) (allocation : Region → Allocation) : Bool :=
  allRegions.all fun r => (pointerIndices seed r).all fun i =>
    let bound := ((sourceRanges c s).getD i (.fixed 0)).bound
    decide (0 ≤ (allocation r).dataPointer+4*bound.lo ∧ (allocation r).dataPointer+4*bound.hi < 2^64)

theorem binds_getD {as : List BExpr} {es : List Expr} (related : List.Forall₂ BExpr.Binds as es)
    (i : Nat) : ((as.getD i (.fixed 0))).Binds (es.getD i (.lit 0)) := by
  induction related generalizing i with
  | nil => exact BExpr.Binds.lit ⟨le_rfl,le_rfl⟩
  | cons relation rest ih =>
    cases i
    · exact relation
    · exact ih _

theorem getD_valid (as : List BExpr) (checked : as.all BExpr.valid=Bool.true) (i : Nat) :
    (as.getD i (.fixed 0)).valid=Bool.true := by
  induction as generalizing i with
  | nil =>
    simp only [List.getD_nil]
    decide
  | cons a rest ih =>
    have check : a.valid=Bool.true ∧ rest.all BExpr.valid=Bool.true := by
      simpa only [List.all_cons,Bool.and_eq_true] using checked
    cases i
    · exact check.1
    · exact ih check.2 _

theorem formed_safe (c : Shape) (s : SeedStrides) (seed : Bool) (allocation : Region → Allocation)
    (checked : rangeCheck c s=Bool.true) (pointers : pointerCheck c s seed allocation=Bool.true)
    (h tk tv e j count : Nat) (hh : h < c.heads) (hk : tk < c.KT) (hv : tv < c.VT)
    (he : e < 64) (hj : j < 16) (hc : count < c.N)
    (r : Region) (x : Expr) (member : x∈pointerExprs c s seed h tk tv e j count r) :
    x.Valid ∧ 0 ≤ (allocation r).dataPointer+4*x.eval ∧ (allocation r).dataPointer+4*x.eval < 2^64 := by
  have inRegions : r∈allRegions := by cases r <;> simp [allRegions]
  have limits := List.all_eq_true.mp pointers r inRegions
  change x∈(pointerIndices seed r).map (fun i => (sourceTrees c s h tk tv e j count).getD i (.lit 0)) at member
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp member
  have binds := binds_getD (source_ranges_bind c s h tk tv e j count hh hk hv he hj hc) i
  have certificate := BExpr.certified _ _ binds (getD_valid (sourceRanges c s) checked i)
  have limit := List.all_eq_true.mp limits i hi
  simp only [decide_eq_true_eq] at limit
  have inside := certificate.2
  simp only [Interval.Contains] at inside
  exact ⟨certificate.1,by omega,by omega⟩

#axiomsClean formed_safe
#axiomsClean source_ranges_bind
#axiomsClean source_valid
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked

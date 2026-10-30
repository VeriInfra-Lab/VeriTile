import bench.tritonbench_g.lora_expand_gemv.Proofs.Single
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete
namespace VeriTile.Bench.Repairs.LoraExpandGemv
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def offset (g : Geometry) (q : Nat) := yoff g (q/g.N) (q%g.N)
def cell (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (q : Nat) :=
  resultCell plans g raw (q/g.N) (q%g.N)

def Snapshot (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (seen : List Nat) (mem : Memory) : Prop :=
  (∀ r o,r≠("Y" : RegionName) → mem r o=entry g raw r o) ∧
  (∀ q,q<g.count → mem "Y" (offset g q)=
    if q∈seen then cell plans g raw q else entry g raw "Y" (offset g q)) ∧
  (∀ r o,(r≠("Y" : RegionName) ∨ ∀ q,q<g.count → o≠offset g q) → mem r o=entry g raw r o)

theorem offset_injective (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (q p : Nat) (hq : q<g.count) (hp : p<g.count) (eq : offset g q=offset g p) : q=p := by
  have hq' := coordinates g q hq
  have hp' := coordinates g p hp
  have h := y_injective g raw v _ _ _ _ hq'.2 hq'.1 hp'.2 hp'.1 eq
  have h1 := Nat.div_add_mod q g.N
  have h2 := Nat.div_add_mod p g.N
  rw [h.1,h.2] at h1
  omega

theorem snapshot_start (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) :
    Snapshot plans g raw [] (entry g raw) := by simp [Snapshot]

theorem snapshot_put (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (seen : List Nat) (mem : Memory) (h : Snapshot plans g raw seen mem)
    (q : Nat) (hq : q<g.count) :
    Snapshot plans g raw (q::seen) (put mem (offset g q) (cell plans g raw q)) := by
  refine ⟨?_,?_,?_⟩
  · intro r o hr
    simpa [put,hr] using h.1 r o hr
  · intro p hp
    by_cases he : p=q
    · subst p; simp [put]
    · have ho : offset g p≠offset g q := fun eq => he (offset_injective g raw v p q hp hq eq)
      simp [put,ho,he,h.2.1 p hp]
  · intro r o ho
    have hn : ¬(r="Y" ∧ o=offset g q) := by
      intro he
      rcases ho with hr|ha
      · exact hr he.1
      · exact ha q hq he.2
    simpa [put,hn] using h.2.2 r o ho

def execute (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry) :
    List Nat → Memory → Option Memory
  | [],mem => some mem
  | q::qs,mem => (FP.Structural.exec (Branches.engine rest .ieeeValue plans)
      (Branches.source .singleHalf .single g.parameters)
      (program g (q/g.N) (q%g.N) mem)).bind fun next => execute rest plans g qs next.mem

theorem one_program (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (seen : List Nat) (mem : Memory)
    (h : Snapshot plans g raw seen mem) (q : Nat) (hq : q<g.count) (new : q∉seen) :
    ∃ next,FP.Structural.exec (Branches.engine rest .ieeeValue plans)
      (Branches.source .singleHalf .single g.parameters) (program g (q/g.N) (q%g.N) mem)=some next ∧
      next.mem=put mem (offset g q) (cell plans g raw q) := by
  have hc := coordinates g q hq
  have hf : Fresh g raw mem (q/g.N) (q%g.N) := ⟨h.1,by simpa [offset,new] using h.2.1 q hq⟩
  obtain ⟨next,he,hm,_⟩ := Branches.correctness rest .ieeeValue plans
    g.parameters (program g (q/g.N) (q%g.N) mem) (by change 0<256; decide)
  refine ⟨next,he,?_⟩
  rw [hm,write_single plans g _ _ mem hc.1]
  have hi : (mem "I" (q/g.N)).read .int=logicalIndex g raw (q/g.N) := by
    rw [h.1 "I" _ (by decide)]
    simp [entry,denseIndices,Cell.read_same]
  rw [hi]
  by_cases ha : logicalIndex g raw (q/g.N)=-1
  · simp only [ha,ite_true,cell,resultCell]
    funext r a
    by_cases ho : r="Y" ∧ a=offset g q
    · rcases ho with ⟨rfl,rfl⟩
      simpa [put,offset] using hf.2
    · simp [put,ho]
  · simp only [ha,ite_false]
    rw [source_value plans g raw v _ _ hc.2 hc.1 mem hf ha]
    simp [offset,cell,resultCell,ha]

theorem execute_prefix (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (todo seen : List Nat) (mem : Memory)
    (h : Snapshot plans g raw seen mem)
    (bounded : ∀ q∈todo,q<g.count) (distinct : todo.Nodup)
    (disjoint : ∀ q∈todo,q∉seen) :
    ∃ final,execute rest plans g todo mem=some final ∧
      Snapshot plans g raw (todo.reverse++seen) final := by
  induction todo generalizing seen mem with
  | nil => exact ⟨mem,rfl,by simpa using h⟩
  | cons q qs ih =>
    have hq := bounded q (by simp)
    obtain ⟨next,he,hm⟩ := one_program rest plans g raw v seen mem h q hq (disjoint q (by simp))
    have hn : Snapshot plans g raw (q::seen) next.mem := by rw [hm]; exact snapshot_put plans g raw v seen mem h q hq
    have hdistinct := List.nodup_cons.mp distinct
    obtain ⟨final,hr,hf⟩ := ih (q::seen) next.mem hn
      (fun p hp => bounded p (by simp [hp])) hdistinct.2 (by
        intro p hp
        have hpq : p≠q := by intro eq; exact hdistinct.1 (eq ▸ hp)
        simp [hpq,disjoint p (by simp [hp])])
    refine ⟨final,?_,?_⟩
    · simp [execute,he,hr]
    · simpa [List.reverse_cons,List.append_assoc] using hf

theorem whole_source (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (raw : Nat → Int) (v : Valid g raw) (schedule : List Nat)
    (all : schedule.Perm (List.range g.count)) :
    ∃ final,execute rest plans g schedule (entry g raw)=some final ∧
      (∀ b n,b<g.B → n<g.N → final "Y" (yoff g b n)=resultCell plans g raw b n) ∧
      (∀ r o,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → o≠yoff g b n) →
        final r o=entry g raw r o) := by
  obtain ⟨final,he,hf⟩ := execute_prefix rest plans g raw v schedule [] (entry g raw)
    (snapshot_start plans g raw) (fun q hq => List.mem_range.mp (all.mem_iff.mp hq))
    (all.nodup_iff.mpr List.nodup_range) (by simp)
  refine ⟨final,he,?_,?_⟩
  · intro b n hb hn
    have hc := coverage g b n hb hn
    have hm : b*g.N+n∈schedule := all.mem_iff.mpr (List.mem_range.mpr hc.1)
    simpa [hm,offset,cell,hc.2.1,hc.2.2] using hf.2.1 (b*g.N+n) hc.1
  · intro r o ho
    apply hf.2.2 r o
    rcases ho with hr|ha
    · exact .inl hr
    · right
      intro q hq
      have hc := coordinates g q hq
      exact ha _ _ hc.2 hc.1

#axiomsClean offset_injective
#axiomsClean one_program
#axiomsClean execute_prefix
#axiomsClean whole_source
#specNonCircular resultCell avoiding [execute,Branches.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LoraExpandGemv

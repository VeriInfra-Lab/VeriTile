import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.AdmissionCore
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Tensor
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.CacheTransform.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 100000

def cosine (r : Request) : Parameters :=
  ⟨"cos_cache", "cos_rows", "cos_output", r.count, r.width*4, 0, 256⟩
def sine (r : Request) : Parameters :=
  ⟨"sin_cache", "sin_rows", "sin_output", r.count, r.width*4, 0, 256⟩

def positioned (s : State α) (pid : Nat) : State α :=
  { s with pids := fun ax => if ax=0 then pid else 0 }

/-- Each step executes the complete canonical DSL, with START=0 and the
actual unmodified program ID. This is kernel execution, not host orchestration. -/
def launch [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : Nat → Option (State α)
  | 0 => some s
  | k+1 => do
      let t ← launch M p s k
      FP.Structural.exec M (source p) (positioned t k)

def pair [Inhabited α] (M : Algebra α) (r : Request) (s : State α) : Option (State α) := do
  let t ← launch M (cosine r) s (blocks r.count)
  launch M (sine r) t (blocks r.count)

theorem launch_success [Inhabited α] (M : Algebra α) (r : Request) (h : InputContract r)
    (p : Parameters) (count : p.count=r.count) (start : p.start=0) (block : p.block=256)
    (s : State α) (n : Nat) (hn : n≤blocks r.count) :
    launch M p s n=some (runTiles M p s n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have bounds : n<blocks r.count := by omega
    have selection := single_launch r h n bounds
    have parameters : tileParameters p n=p := by
      cases p <;> simp_all only [tileParameters]
    have position : ∀ s : State α,tileState s n=positioned s n := by
      intro s
      simp only [tileState,positioned,selection.2]
    rw [launch,ih (by omega)]
    simpa only [parameters,position,runTiles] using
      tile_success M p n (runTiles M p s n)

theorem pair_success [Inhabited α] (M : Algebra α) (r : Request) (h : InputContract r)
    (s : State α) : pair M r s=some (pairGather M (cosine r) (sine r) s) := by
  unfold pair
  rw [launch_success M r h (cosine r) rfl rfl rfl s _ (Nat.le_refl _)]
  change launch M (sine r) (runTiles M (cosine r) s (blocks r.count)) (blocks r.count)=_
  rw [launch_success M r h (sine r) rfl rfl rfl _ _ (Nat.le_refl _)]
  rfl

/-- Initial source state parameterized only by the two payload byte windows
and the actually read row-table values. No output value is supplied. -/
def initial (r : Request) (cosBytes sinBytes : Nat → Nat) (cosRows sinRows : Nat → Int)
    (s : State α) : State α :=
  { s with mem := fun region offset =>
      if region="cos_cache" then .mk .nat (cosBytes offset)
      else if region="sin_cache" then .mk .nat (sinBytes offset)
      else if region="cos_rows" then .mk .int (cosRows offset)
      else if region="sin_rows" then .mk .int (sinRows offset)
      else s.mem region offset }

theorem gather_input [Inhabited α] (r : Request) (h : InputContract r)
    (p : Parameters) (capacity : Nat) (cache : Nat → Nat) (rows : Nat → Int) (s : State α)
    (count : p.count=r.count) (width : p.rowBytes=r.width*4)
    (freshCache : p.cache≠p.output) (freshRows : p.rows≠p.output)
    (table : ∀ q,q<r.outputRows → s.mem p.rows q=.mk .int (rows q))
    (tableValues : 0<r.count → ∀ q,q<r.outputRows → rows q=(tableRows r.mode r.naturalLengths q : Int))
    (payload : ∀ o,o<capacity*(r.width*4) → s.mem p.cache o=.mk .nat (cache o))
    (valid : ValidLengths r.mode r.naturalLengths capacity)
    (extent : capacity*r.width*4<2^63) (positive : 0<p.count) :
    GatherInput p s r.outputRows capacity (tableRows r.mode r.naturalLengths) cache := by
  have nonempty : 0<r.count := by simpa only [count] using positive
  have shape : p.count=r.outputRows*p.rowBytes := by
    simp only [count,width,Request.count,Nat.mul_assoc]
  have rowBytes : 0<r.width*4 := by
    rw [count,Request.count,Nat.mul_assoc] at positive
    exact Nat.pos_of_mul_pos_left positive
  refine ⟨shape,by simpa only [width] using rowBytes,freshCache,freshRows,?_,?_,?_,?_⟩
  · intro q hq
    rw [table q hq,tableValues nonempty q hq]
  · intro q hq
    apply tableRows_bound r.mode r.naturalLengths capacity valid q
    simpa only [←outputRows_eq] using hq
  · have bigger := Nat.le_mul_of_pos_right capacity rowBytes
    rw [←Nat.mul_assoc] at bigger
    omega
  · simpa only [width] using payload

/-- Complete actual-source byte outputs for both kernels, successful execution
and every tagged cell outside the two destination windows. -/
theorem complete [Inhabited α] (M : Algebra α) (r : Request) (h : InputContract r)
    (cosBytes sinBytes : Nat → Nat) (cosRows sinRows : Nat → Int) (s : State α)
    (cosTableValues : 0<r.count → ∀ q,q<r.outputRows → cosRows q=(tableRows r.mode r.naturalLengths q : Int))
    (sinTableValues : 0<r.count → ∀ q,q<r.outputRows → sinRows q=(tableRows r.mode r.naturalLengths q : Int)) :
    let first := initial r cosBytes sinBytes cosRows sinRows s
    ∃ t,pair M r first=some t ∧
      (∀ o,o<r.count → t.mem "cos_output" o=.mk .nat
        (byteTarget (cosine r) (tableRows r.mode r.naturalLengths) cosBytes o)) ∧
      (∀ o,o<r.count → t.mem "sin_output" o=.mk .nat
        (byteTarget (sine r) (tableRows r.mode r.naturalLengths) sinBytes o)) ∧
      (∀ (region : RegionName) o,(region≠"cos_output" ∨ r.count≤o) →
        (region≠"sin_output" ∨ r.count≤o) → t.mem region o=first.mem region o) := by
  let first := initial r cosBytes sinBytes cosRows sinRows s
  have valid := admitted_lengths r h
  have inputsCos : 0<(cosine r).count → GatherInput (cosine r) first r.outputRows r.cosRows
      (tableRows r.mode r.naturalLengths) cosBytes := by
    apply gather_input r h (cosine r) r.cosRows cosBytes cosRows first rfl rfl
      (by simp [cosine,sine]) (by simp [cosine,sine]) _ cosTableValues _ valid.1 h.2.2.1
    · intro q hq; simp [first,initial,cosine]
    · intro o ho; simp [first,initial,cosine]
  have inputsSin : 0<(sine r).count → GatherInput (sine r) first r.outputRows r.sinRows
      (tableRows r.mode r.naturalLengths) sinBytes := by
    apply gather_input r h (sine r) r.sinRows sinBytes sinRows first rfl rfl
      (by simp [cosine,sine]) (by simp [cosine,sine]) _ sinTableValues _ valid.2 h.2.2.2.1
    · intro q hq; simp [first,initial,sine]
    · intro o ho; simp [first,initial,sine]
  obtain ⟨hc,hs,hf⟩ := pair_correct M (cosine r) (sine r) first r.outputRows r.cosRows r.sinRows
    (tableRows r.mode r.naturalLengths) cosBytes sinBytes inputsCos inputsSin
    (by simp [cosine,sine]) (by simp [cosine,sine]) (by simp [cosine,sine])
  exact ⟨pairGather M (cosine r) (sine r) first,pair_success M r h first,hc,hs,hf⟩

#axiomsClean launch_success
#axiomsClean pair_success
#axiomsClean complete
#specNonCircular mathematical avoiding [source,pair,initial,gather,pairGather]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.KernelSource

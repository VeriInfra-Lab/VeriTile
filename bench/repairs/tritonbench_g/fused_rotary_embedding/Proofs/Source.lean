import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Vector

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding
variable {α : Type}

theorem masked_range (n limit : Nat) (h : limit≤n) :
    (List.range n).filter (fun i => decide (i<limit))=List.range limit := by
  induction n generalizing limit with
  | zero =>
    have : limit=0 := by omega
    subst limit
    simp
  | succ n ih =>
    by_cases hl : limit≤n
    · rw [List.range_succ,List.filter_append,ih limit hl]
      simp [show ¬n<limit by omega]
    · have : limit=n+1 := by omega
      subst limit
      apply List.filter_eq_self.mpr
      intro a ha
      simpa using (List.mem_range.mp ha)

def programHead (s : Shape) (pid : Program) (kind : Kind) := if kind=.q then pid.1 else pid.1/s.groups

theorem program_head_bound (s : Shape) (hs : s.Valid) (pid : Program) (hp : pid∈programs s) (kind : Kind) :
    programHead s pid kind<s.heads kind := by
  have ph := (List.mem_product.mp hp).1
  have bh : pid.1<s.headsQ := List.mem_range.mp ph
  by_cases kq : kind=.q
  · simpa [programHead,Shape.heads,kq] using bh
  · have h := (Nat.div_lt_iff_lt_mul (groups_positive s hs)).mpr
      (show pid.1<s.headsK*s.groups by simpa [Nat.mul_comm s.headsK s.groups,groups_mul s hs] using bh)
    simpa [programHead,Shape.heads,kq] using h

theorem branchWork_explicit (s : Shape) (hs : s.Valid) (pid : Program) (hp : pid∈programs s) (kind : Kind) :
    branchWork s pid kind = if kind=.q ∨ pid.1%s.groups=0 then
      (List.range s.half).map (fun i => (⟨kind,pid.2,programHead s pid kind,i⟩ : Work)) else [] := by
  have ht : pid.2<s.tokens := List.mem_range.mp (List.mem_product.mp hp).2
  have hh := program_head_bound s hs pid hp kind
  dsimp only [programHead] at hh
  unfold branchWork
  by_cases guard : kind=.q ∨ pid.1%s.groups=0
  · simp only [leader,guard,decide_true,Function.comp_def,List.filter_map,List.filter_true,
      List.map_map,workOf,List.filter_map]
    simp only [Shape.Active,ht,hh,true_and]
    rw [masked_range s.halfBlock s.half (half_covered s)]
    simp [guard,programHead,Function.comp_def]
  · simp [leader,guard,List.filter_map,Function.comp_def]

/-- Actual single full-width V load/scatter, with its original lane order. -/
def fullVWrites (c : Config) (pid : Program) (m : Memory α) : List (Stores.Write α) :=
  if pid.1%c.shape.groups=0 then
    ((List.range c.shape.fullBlock).filter (fun d => decide (d<c.shape.width))).map (fun d =>
      ((Region.vcache,c.shape.cache false (c.block pid.2) (pid.1/c.shape.groups) (c.slot pid.2) d),
       m .v (c.shape.data .v pid.2 (pid.1/c.shape.groups) d)))
  else []

theorem fullVWrites_eq (c : Config) (hc : c.Valid) (p : Kind → Ops α)
    (pid : Program) (hp : pid∈programs c.shape) (m : Memory α) :
    fullVWrites c pid m=vectorWrites c p (branchWork c.shape pid .v) m := by
  rw [branchWork_explicit c.shape hc.shape pid hp .v]
  by_cases guard : pid.1%c.shape.groups=0
  · simp only [fullVWrites,guard,ite_true,Kind.noConfusion,false_or]
    rw [masked_range _ _ (full_covered c.shape)]
    have wh : c.shape.width=c.shape.half+c.shape.half := by have := width_halves c.shape hc.shape;omega
    rw [wh,List.range_add,List.map_append,List.map_map]
    simp [vectorWrites,List.map_map,Function.comp_def,pairValues,component,Config.output,
      Shape.input,Shape.dim,Kind.input,Kind.output,programHead,Nat.add_comm]
  · simp [fullVWrites,guard,vectorWrites]

/-- This source includes the full-width V vector operation, not an assumed
pair implementation. Its connection to the proved pair schedule is below. -/
def actualProgram (c : Config) (p : Kind → Ops α) (pid : Program) (m : Memory α) : Memory α :=
  let afterQ := vectorBranch c p (branchWork c.shape pid .q) m
  let afterK := vectorBranch c p (branchWork c.shape pid .k) afterQ
  Stores.commit (fullVWrites c pid afterK) afterK
def actualPrograms (c : Config) (p : Kind → Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::ps,m => actualPrograms c p ps (actualProgram c p pid m)
def actualSource (c : Config) (p : Kind → Ops α) (m : Memory α) := actualPrograms c p (programs c.shape) m

theorem actualProgram_eq (c : Config) (hc : c.Valid) (p : Kind → Ops α)
    (pid : Program) (hp : pid∈programs c.shape) (m : Memory α) :
    actualProgram c p pid m=sourceProgram c p pid m := by
  unfold actualProgram sourceProgram
  dsimp only
  rw [fullVWrites_eq c hc p pid hp]
  rfl
theorem actualPrograms_eq (c : Config) (hc : c.Valid) (p : Kind → Ops α) (ps : List Program)
    (inside : ∀ pid∈ps,pid∈programs c.shape) (m : Memory α) :
    actualPrograms c p ps m=sourcePrograms c p ps m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    simp only [actualPrograms,sourcePrograms]
    rw [actualProgram_eq c hc p pid (inside pid (by simp))]
    exact ih (fun pid hp => inside pid (by simp [hp])) _
theorem actualSource_eq_run (c : Config) (hc : c.Valid) (p : Kind → Ops α) (m : Memory α) :
    actualSource c p m=run c p m := by
  rw [actualSource,actualPrograms_eq c hc p _ (fun _ h => h)]
  exact source_eq_run c hc p m

#axiomsClean fullVWrites_eq
#axiomsClean actualSource_eq_run
end VeriTile.Bench.Repairs.FusedRotaryEmbedding

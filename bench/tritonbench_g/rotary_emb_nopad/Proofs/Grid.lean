import bench.tritonbench_g.rotary_emb_nopad.Proofs.Execution

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
open Checked
abbrev Program := Nat×Nat
abbrev Lane := Bool×(Nat×Nat)
abbrev Launch := Program×Lane

def keys (r : Request) : List Bool := if r.cached then [true,false] else [false,true]
def programs (r : Request) : List Program := (List.range r.headPrograms) ×ˢ (List.range r.tokenPrograms)
def lanes (r : Request) : List Lane := (keys r) ×ˢ ((List.range r.blockTokens) ×ˢ (List.range r.blockPairs))
def launches (r : Request) : List Launch := (programs r) ×ˢ (lanes r)
def leader (r : Request) (key : Bool) (head : Nat) : Bool :=
  if !r.cached && key then decide (head%r.groups=0) else true

def headOf (r : Request) (key : Bool) (head : Nat) := if !r.cached && key then head/r.groups else head

def workOf (r : Request) (launch : Launch) : Work :=
  ⟨launch.2.1,launch.1.2*r.blockTokens+launch.2.2.1,headOf r launch.2.1 launch.1.1,launch.2.2.2⟩

def eligible (r : Request) (launch : Launch) : Bool := leader r launch.2.1 launch.1.1

def schedule (r : Request) : List Work :=
  (((launches r).filter (eligible r)).map (workOf r)).filter (fun w => decide (r.Active w))

theorem block_tokens_pos (r : Request) : 0<r.blockTokens := by cases h : r.cached <;> simp [Request.blockTokens,h]
theorem keys_mem (r : Request) (key : Bool) : key∈keys r := by cases h : r.cached <;> cases key <;> simp [keys,h]
theorem keys_nodup (r : Request) : (keys r).Nodup := by cases h : r.cached <;> simp [keys,h]

theorem launches_nodup (r : Request) : (launches r).Nodup :=
  (List.nodup_range.product List.nodup_range).product
    ((keys_nodup r).product (List.nodup_range.product List.nodup_range))

theorem head_injective (r : Request) (key : Bool) (a b : Nat)
    (ha : leader r key a=true) (hb : leader r key b=true)
    (same : headOf r key a=headOf r key b) : a=b := by
  by_cases grouped : (!r.cached && key)=true
  · have amod : a%r.groups=0 := by simpa [leader,grouped] using ha
    have bmod : b%r.groups=0 := by simpa [leader,grouped] using hb
    have eq : a/r.groups=b/r.groups := by simpa [headOf,grouped] using same
    have ax := Nat.mod_add_div a r.groups
    have bx := Nat.mod_add_div b r.groups
    rw [amod,zero_add] at ax
    rw [bmod,zero_add,←eq] at bx
    omega
  · simpa [headOf,grouped] using same

theorem work_injective (r : Request) (a b : Launch)
    (ha : a∈(launches r).filter (eligible r)) (hb : b∈(launches r).filter (eligible r))
    (same : workOf r a=workOf r b) : a=b := by
  rcases a with ⟨⟨ph,ps⟩,key,it,ip⟩
  rcases b with ⟨⟨ph',ps'⟩,key',it',ip'⟩
  have ea := (List.mem_filter.mp ha).2
  have eb := (List.mem_filter.mp hb).2
  have ma := (List.mem_filter.mp ha).1
  have mb := (List.mem_filter.mp hb).1
  simp only [launches,programs,lanes,List.mem_product,List.mem_range] at ma mb
  simp only [workOf,Work.mk.injEq] at same
  obtain ⟨seq,token⟩ := mixed_injective r.blockTokens _ _ _ _ ma.2.2.1 mb.2.2.1 same.2.1
  have head : ph=ph' := by
    apply head_injective r key ph ph' ea
    · simpa only [eligible,same.1] using eb
    · simpa only [same.1] using same.2.2.1
  simp_all

theorem schedule_nodup (r : Request) : (schedule r).Nodup :=
  (((launches_nodup r).filter (eligible r)).map_on (fun a ha b hb eq => work_injective r a b ha hb eq)).filter _

theorem schedule_active (r : Request) (w : Work) (member : w∈schedule r) : r.Active w :=
  of_decide_eq_true (List.mem_filter.mp member).2

def programHead (r : Request) (w : Work) := if !r.cached && w.key then w.head*r.groups else w.head

theorem head_roundtrip (c : Call) (indices : Indices) (accepted : check c indices=true) (w : Work) :
    leader c.request w.key (programHead c.request w)=true ∧
    headOf c.request w.key (programHead c.request w)=w.head := by
  have positive := (groups_positive c indices accepted).1
  by_cases grouped : (!c.request.cached && w.key)=true
  · simp [programHead,leader,headOf,grouped,Nat.mul_div_cancel _ positive]
  · simp [programHead,leader,headOf,grouped]

theorem program_head_bound (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : programHead c.request w<c.request.headPrograms := by
  have groups := groups_positive c indices accepted
  have hq := (shapes c indices accepted).1
  have hk := (shapes c indices accepted).2.1
  have keyBound := active.2.1
  cases cached : c.request.cached
  · cases key : w.key
    · simpa [programHead,Request.headPrograms,Request.heads,cached,key] using keyBound
    · have headBound : w.head<c.request.kHeads := by simpa [Request.heads,key] using keyBound
      have bound := Nat.mul_lt_mul_of_pos_right headBound groups.1
      simpa [programHead,Request.headPrograms,cached,key,groups.2] using bound
  · have cover : c.request.qHeads≤2^Nat.clog 2 c.request.qHeads := Nat.le_pow_clog (by decide) _
    have kh : c.request.kHeads≤c.request.qHeads := by nlinarith [groups.1,groups.2]
    have head : w.head<c.request.qHeads := by cases hk : w.key <;> simp [Request.heads,hk] at keyBound <;> omega
    simpa [programHead,Request.headPrograms,cached] using lt_of_lt_of_le head cover

theorem schedule_covers (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) : w∈schedule c.request := by
  apply List.mem_filter.mpr
  refine ⟨?_,by simpa using active⟩
  apply List.mem_map.mpr
  let launch : Launch := ((programHead c.request w,w.token/c.request.blockTokens),
    (w.key,w.token%c.request.blockTokens,w.pair))
  refine ⟨launch,?_,?_⟩
  · apply List.mem_filter.mpr
    refine ⟨?_,(head_roundtrip c indices accepted w).1⟩
    have positive := block_tokens_pos c.request
    have token : w.token/c.request.blockTokens<c.request.tokenPrograms := by
      have bound := active.1
      cases hc : c.request.cached <;> simp [Request.blockTokens,Request.tokenPrograms,hc] <;> omega
    have pair := lt_of_lt_of_le active.2.2 (Nat.le_pow_clog (by decide : 1<2) c.request.half)
    simp only [launch,launches,programs,lanes,List.mem_product,List.mem_range]
    exact ⟨⟨program_head_bound c indices accepted w active,token⟩,
      keys_mem c.request w.key,Nat.mod_lt _ positive,pair⟩
  · have eq := (head_roundtrip c indices accepted w).2
    have token := Nat.mod_add_div w.token c.request.blockTokens
    have tokenEq : w.token/c.request.blockTokens*c.request.blockTokens+w.token%c.request.blockTokens=w.token := by
      rw [Nat.mul_comm]
      omega
    simp only [launch,workOf,eq,tokenEq]

#axiomsClean schedule_nodup
#axiomsClean schedule_covers
end VeriTile.Bench.Repairs.RotaryEmbNopad

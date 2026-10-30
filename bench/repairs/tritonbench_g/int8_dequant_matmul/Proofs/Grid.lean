import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Plan

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

abbrev Commit (s : Shape) (c : Config) := MatrixCell s × Fin c.S
def outputAddress (s : Shape) (b : Buffers) (x : MatrixCell s) : Address :=
  (b.out,b.outOffset+x.1.val*s.N+x.2.val)

theorem outputAddress_injective (s : Shape) (b : Buffers) : Function.Injective (outputAddress s b) := by
  intro x y same
  have h := congrArg Prod.snd same
  simp only [outputAddress,Prod.snd] at h
  have xn := x.2.isLt
  have yn := y.2.isLt
  have npos : 0<s.N := by omega
  have eq : x.1.val*s.N+x.2.val=y.1.val*s.N+y.2.val := by omega
  have divs := congrArg (fun k => k/s.N) eq
  have rems := congrArg (fun k => k%s.N) eq
  simp only [Nat.mul_comm _ s.N,Nat.mul_add_div npos,Nat.div_eq_of_lt xn,Nat.div_eq_of_lt yn,Nat.add_zero] at divs
  apply Prod.ext
  · exact Fin.ext divs
  · apply Fin.ext
    rw [divs] at eq
    omega

def projection (x : MatrixCell s) (events : List (Commit s c)) : List (Fin c.S) :=
  events.filterMap (fun e => if e.1=x then some e.2 else none)

/-- Per-cell atomic linearizations may differ. This certificate only says
each dispatched source split commits once; it constrains no values or memory.
Grouped-grid ownership below proves the source-side unique event identity. -/
structure Schedule (s : Shape) (c : Config) where
  events : List (Commit s c)
  complete : ∀ x,(projection x events).Perm (List.finRange c.S)

def commit (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (value : H) (e : Commit s c) (m : Memory F H) : Memory F H :=
  write16 m (outputAddress s b e.1)
    (if c.S=1 then value else a.hadd (read16 a m (outputAddress s b e.1)) value)

def gridRun (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) :
    List (Commit s c) → Memory F H → Memory F H
  | [],m => m
  | e::es,m => gridRun a s c b selected divisor hasBias es
      (commit a s c b (partialValue a s c b selected divisor hasBias m e.1.1 e.1.2 e.2) e m)

def reset (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) : Memory F H :=
  fun p => if p.1=b.out ∧ b.outOffset≤p.2 ∧ p.2<b.outOffset+s.M*s.N then .inr (.inr a.hzero) else m p
def launch (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c) (m : Memory F H) :=
  gridRun a s c b selected divisor hasBias q.events (if c.S=1 then m else reset a s b m)

theorem commit_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (v : H) (e : Commit s c) (m : Memory F H) (p : Address) (h : p.1≠b.out) :
    commit a s c b v e m p=m p := by
  have different : p≠outputAddress s b e.1 := by intro he; exact h (congrArg Prod.fst he)
  simp only [commit,write16,if_neg different]
theorem gridRun_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (events : List (Commit s c))
    (m : Memory F H) : InputsAgree b (gridRun a s c b selected divisor hasBias events m) m := by
  intro p hp
  induction events generalizing m with
  | nil => rfl
  | cons e es ih => rw [gridRun,ih,commit_frame _ _ _ _ _ _ _ _ hp]
theorem reset_frame (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) :
    InputsAgree b (reset a s b m) m := by
  intro p hp
  simp [reset,hp]
theorem launch_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c) (m : Memory F H) :
    InputsAgree b (launch a s c b selected divisor hasBias q m) m := by
  intro p hp
  unfold launch
  rw [gridRun_frame _ _ _ _ _ _ _ _ _ p hp]
  split_ifs
  · rfl
  · exact reset_frame a s b m p hp

def cellFold (a : Ops I F H) (c : Config) (values : Fin c.S → H) (zs : List (Fin c.S)) (seed : H) :=
  zs.foldl (fun old z => if c.S=1 then values z else a.hadd old (values z)) seed

theorem gridRun_cell (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Fresh)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (events : List (Commit s c))
    (initial m : Memory F H) (same : InputsAgree b m initial) (x : MatrixCell s) :
    read16 a (gridRun a s c b selected divisor hasBias events m) (outputAddress s b x)=
      cellFold a c (fun z => partialValue a s c b selected divisor hasBias initial x.1 x.2 z)
        (projection x events) (read16 a m (outputAddress s b x)) := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    have frame : InputsAgree b
        (commit a s c b (partialValue a s c b selected divisor hasBias m e.1.1 e.1.2 e.2) e m) initial := by
      intro p hp
      rw [commit_frame _ _ _ _ _ _ _ _ hp]
      exact same p hp
    rw [gridRun,ih _ frame]
    by_cases eq : e.1=x
    · have addr : outputAddress s b x=outputAddress s b e.1 := by rw [eq]
      simp only [projection,List.filterMap_cons,if_pos eq,Option.toList_some,List.cons_append,
        List.nil_append,cellFold,List.foldl_cons,commit,write16,if_pos addr,read16]
      rw [partial_current a s c b fresh selected divisor hasBias m initial same,eq]
    · have addr : outputAddress s b x≠outputAddress s b e.1 := by
        intro h
        exact eq ((outputAddress_injective s b h).symm)
      simp only [projection,List.filterMap_cons,if_neg eq,Option.toList_none,List.nil_append,
        commit,write16,read16,if_neg addr]

theorem reset_output (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) (x : MatrixCell s) :
    read16 a (reset a s b m) (outputAddress s b x)=a.hzero := by
  have hi : x.1.val*s.N+x.2.val<s.M*s.N := by
    have h := Nat.mul_le_mul_right s.N x.1.isLt
    simp only [Nat.succ_mul] at h
    have hn := x.2.isLt
    omega
  simp [read16,reset,outputAddress,show b.outOffset≤b.outOffset+x.1.val*s.N+x.2.val by omega,
    show b.outOffset+x.1.val*s.N+x.2.val<b.outOffset+s.M*s.N by omega]

theorem cellFold_seed (a : Ops I F H) (c : Config) (values : Fin c.S → H)
    (zs : List (Fin c.S)) (complete : zs.Perm (List.finRange c.S)) (single : c.S=1) (x y : H) :
    cellFold a c values zs x=cellFold a c values zs y := by
  have len : zs.length=1 := by simpa [single] using complete.length_eq
  cases zs with
  | nil => simp at len
  | cons z zs => simp only [cellFold,List.foldl_cons,if_pos single]

def wordReference (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) : H :=
  cellFold a c (fun z => partialValue a s c b selected divisor hasBias initial x.1 x.2 z)
    (projection x q.events) a.hzero

theorem launch_output (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Fresh)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) :
    read16 a (launch a s c b selected divisor hasBias q initial) (outputAddress s b x)=
      wordReference a s c b selected divisor hasBias q initial x := by
  unfold launch wordReference
  by_cases h : c.S=1
  · rw [if_pos h,gridRun_cell a s c b fresh selected divisor hasBias q.events initial initial (fun _ _ => rfl)]
    exact cellFold_seed _ _ _ _ (q.complete x) h _ _
  · rw [if_neg h,gridRun_cell a s c b fresh selected divisor hasBias q.events initial (reset a s b initial)
      (reset_frame a s b initial),reset_output]

theorem gridRun_typed (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (events : List (Commit s c))
    (m : Memory F H) (x : MatrixCell s)
    (h : (∃ value,m (outputAddress s b x)=.inr (.inr value)) ∨ projection x events≠[]) :
    ∃ value,gridRun a s c b selected divisor hasBias events m (outputAddress s b x)=.inr (.inr value) := by
  induction events generalizing m with
  | nil =>
    rcases h with h|h
    · exact h
    · exact False.elim (h rfl)
  | cons e es ih =>
    rw [gridRun]
    apply ih
    by_cases eq : e.1=x
    · left
      unfold commit write16
      refine ⟨if c.S=1 then partialValue a s c b selected divisor hasBias m e.1.1 e.1.2 e.2
        else a.hadd (read16 a m (outputAddress s b e.1)) (partialValue a s c b selected divisor hasBias m e.1.1 e.1.2 e.2),?_⟩
      simp only [eq,ite_true]
    · have different : outputAddress s b x≠outputAddress s b e.1 := by
        intro he
        exact eq ((outputAddress_injective s b he).symm)
      rcases h with h|h
      · left
        obtain ⟨v,hv⟩ := h
        exact ⟨v,by simpa [commit,write16,different] using hv⟩
      · right
        simpa [projection,eq] using h

theorem launch_word (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Fresh)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) :
    launch a s c b selected divisor hasBias q initial (outputAddress s b x)=
      .inr (.inr (wordReference a s c b selected divisor hasBias q initial x)) := by
  have nonempty : projection x q.events≠[] := by
    intro h
    have len := (q.complete x).length_eq
    rw [h] at len
    simp only [List.length_nil,List.length_finRange] at len
    have positive := c.positiveS
    omega
  obtain ⟨value,hv⟩ := gridRun_typed a s c b selected divisor hasBias q.events
    (if c.S=1 then initial else reset a s b initial) x (Or.inr nonempty)
  have out := launch_output a s c b fresh selected divisor hasBias q initial x
  change read16 a (gridRun a s c b selected divisor hasBias q.events _) _=_ at out
  simp only [read16,hv] at out
  change gridRun a s c b selected divisor hasBias q.events _ _=_
  rw [hv,out]

namespace Grouped
open IntScaledMatmul.Revision001.Geometry
def valid (s : Shape) (c : Config) : Valid (geometry s c) where
  blockRows := c.positiveM
  blockColumns := c.positiveN
  groupRows := by change 0<8; decide
  writable := by
    intro x y same
    have hi := outputAddress_injective s (Buffers.mk 0 0 0 0 0 0 0 0 0 0 0 0)
    apply hi
    apply Prod.ext
    · rfl
    · simpa [outputAddress,address,geometry] using same

theorem every_cell (s : Shape) (c : Config) (x : MatrixCell s) :
    ∃ pid,pid<count (geometry s c) ∧ Owns (geometry s c) pid x :=
  covers _ (valid s c) x
theorem only_one_owner (s : Shape) (c : Config) (p q : Nat)
    (hp : p<count (geometry s c)) (hq : q<count (geometry s c)) (x : MatrixCell s)
    (hpx : Owns (geometry s c) p x) (hqx : Owns (geometry s c) q x) : p=q :=
  owner_unique _ (valid s c) p q hp hq x hpx hqx
theorem only_one_lane (s : Shape) (c : Config) (pid : Nat) (i j : Lane (geometry s c))
    (hr : outRow (geometry s c) pid i=outRow (geometry s c) pid j)
    (hc : outColumn (geometry s c) pid i=outColumn (geometry s c) pid j) : i=j :=
  lane_injective _ pid i j hr hc
end Grouped

#axiomsClean gridRun_frame
#axiomsClean gridRun_cell
#axiomsClean launch_frame
#axiomsClean launch_output
#axiomsClean launch_word
#axiomsClean Grouped.every_cell
#axiomsClean Grouped.only_one_owner
#axiomsClean Grouped.only_one_lane
end
end VeriTile.Bench.Repairs.Int8DequantMatmul

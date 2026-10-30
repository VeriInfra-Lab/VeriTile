import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Core

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev Commit (s : Shape) (c : Config) := MatrixCell s × Fin c.S
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
  writeStored m (outputAddress s b e.1)
    (if c.S=1 then contribution a b.outPrecision value else outputAdd a b.outPrecision (readStored a b.outPrecision m (outputAddress s b e.1)) value)

def gridRun (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) :
    List (Commit s c) → Memory F H → Memory F H
  | [],m => m
  | e::es,m => gridRun a s c b selected es
      (commit a s c b (partialValue a s c b selected m e.1.1 e.1.2 e.2) e m)

def reset (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) : Memory F H :=
  fun p => if p.1=b.out ∧ b.outOffset≤p.2 ∧ p.2<b.outOffset+s.M*s.N then .inr (outputZero a b.outPrecision) else m p
def launch (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (q : Schedule s c) (m : Memory F H) :=
  gridRun a s c b selected q.events (reset a s b m)

theorem commit_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (v : H) (e : Commit s c) (m : Memory F H) (p : Address) (h : p.1≠b.out) :
    commit a s c b v e m p=m p := by
  have different : p≠outputAddress s b e.1 := by intro he; exact h (congrArg Prod.fst he)
  simp only [commit,writeStored,if_neg different]
theorem gridRun_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (events : List (Commit s c))
    (m : Memory F H) : InputsAgree b (gridRun a s c b selected events m) m := by
  intro p hp
  induction events generalizing m with
  | nil => rfl
  | cons e es ih => rw [gridRun,ih,commit_frame _ _ _ _ _ _ _ _ hp]
theorem reset_frame (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) :
    InputsAgree b (reset a s b m) m := by
  intro p hp
  simp [reset,hp]
theorem launch_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (q : Schedule s c) (m : Memory F H) :
    InputsAgree b (launch a s c b selected q m) m := by
  intro p hp
  unfold launch
  rw [gridRun_frame a s c b selected q.events (reset a s b m) p hp]
  exact reset_frame a s b m p hp

def cellFold (a : Ops I F H) (c : Config) (precision : Precision) (values : Fin c.S → H)
    (zs : List (Fin c.S)) (seed : Stored F H) :=
  zs.foldl (fun old z => if c.S=1 then contribution a precision (values z) else outputAdd a precision old (values z)) seed

theorem gridRun_cell (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Separate)
    (selected : Selection s c) (events : List (Commit s c))
    (initial m : Memory F H) (same : InputsAgree b m initial) (x : MatrixCell s) :
    readStored a b.outPrecision (gridRun a s c b selected events m) (outputAddress s b x)=
      cellFold a c b.outPrecision (fun z => partialValue a s c b selected initial x.1 x.2 z)
        (projection x events) (readStored a b.outPrecision m (outputAddress s b x)) := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    have frame : InputsAgree b
        (commit a s c b (partialValue a s c b selected m e.1.1 e.1.2 e.2) e m) initial := by
      intro p hp
      rw [commit_frame _ _ _ _ _ _ _ _ hp]
      exact same p hp
    rw [gridRun,ih _ frame]
    by_cases eq : e.1=x
    · have addr : outputAddress s b x=outputAddress s b e.1 := by rw [eq]
      simp only [projection,List.filterMap_cons,if_pos eq,Option.toList_some,List.cons_append,
        List.nil_append,cellFold,List.foldl_cons,commit,writeStored,if_pos addr,readStored]
      rw [partial_current a s c b fresh selected m initial same,eq]
    · have addr : outputAddress s b x≠outputAddress s b e.1 := by
        intro h
        exact eq ((outputAddress_injective s b h).symm)
      simp only [projection,List.filterMap_cons,if_neg eq,Option.toList_none,List.nil_append,
        commit,writeStored,readStored,if_neg addr]

theorem reset_output (a : Ops I F H) (s : Shape) (b : Buffers) (m : Memory F H) (x : MatrixCell s) :
    readStored a b.outPrecision (reset a s b m) (outputAddress s b x)=outputZero a b.outPrecision := by
  have hi : x.1.val*s.N+x.2.val<s.M*s.N := by
    have h := Nat.mul_le_mul_right s.N x.1.isLt
    simp only [Nat.succ_mul] at h
    have hn := x.2.isLt
    omega
  simp [readStored,reset,outputAddress,show b.outOffset≤b.outOffset+x.1.val*s.N+x.2.val by omega,
    show b.outOffset+x.1.val*s.N+x.2.val<b.outOffset+s.M*s.N by omega]

def wordReference (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) : Stored F H :=
  cellFold a c b.outPrecision (fun z => partialValue a s c b selected initial x.1 x.2 z)
    (projection x q.events) (outputZero a b.outPrecision)

theorem launch_output (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Separate)
    (selected : Selection s c) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) :
    readStored a b.outPrecision (launch a s c b selected q initial) (outputAddress s b x)=
      wordReference a s c b selected q initial x := by
  unfold launch wordReference
  rw [gridRun_cell a s c b fresh selected q.events initial (reset a s b initial)
    (reset_frame a s b initial),reset_output]

theorem gridRun_typed (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (events : List (Commit s c))
    (m : Memory F H) (x : MatrixCell s)
    (h : (∃ value,m (outputAddress s b x)=.inr value) ∨ projection x events≠[]) :
    ∃ value,gridRun a s c b selected events m (outputAddress s b x)=.inr value := by
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
      unfold commit writeStored
      refine ⟨if c.S=1 then contribution a b.outPrecision (partialValue a s c b selected m e.1.1 e.1.2 e.2)
        else outputAdd a b.outPrecision (readStored a b.outPrecision m (outputAddress s b e.1)) (partialValue a s c b selected m e.1.1 e.1.2 e.2),?_⟩
      simp only [eq,ite_true]
    · have different : outputAddress s b x≠outputAddress s b e.1 := by
        intro he
        exact eq ((outputAddress_injective s b he).symm)
      rcases h with h|h
      · left
        obtain ⟨v,hv⟩ := h
        exact ⟨v,by simpa [commit,writeStored,different] using hv⟩
      · right
        simpa [projection,eq] using h

theorem launch_word (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Separate)
    (selected : Selection s c) (q : Schedule s c)
    (initial : Memory F H) (x : MatrixCell s) :
    launch a s c b selected q initial (outputAddress s b x)=
      .inr (wordReference a s c b selected q initial x) := by
  have nonempty : projection x q.events≠[] := by
    intro h
    have len := (q.complete x).length_eq
    rw [h] at len
    simp only [List.length_nil,List.length_finRange] at len
    have positive := c.positiveS
    omega
  obtain ⟨value,hv⟩ := gridRun_typed a s c b selected q.events
    (reset a s b initial) x (Or.inr nonempty)
  have out := launch_output a s c b fresh selected q initial x
  change readStored a b.outPrecision (gridRun a s c b selected q.events _) _=_ at out
  simp only [readStored,hv] at out
  change gridRun a s c b selected q.events _ _=_
  rw [hv,out]

#axiomsClean gridRun_frame
#axiomsClean gridRun_cell
#axiomsClean launch_frame
#axiomsClean launch_output
#axiomsClean launch_word
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization

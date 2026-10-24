import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Contract

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
open Direct
noncomputable section
set_option maxHeartbeats 3000000

/-- Source-derived local event program for one full grid. Dot sites contain
only fixed K32 instruction/scalar operations, with the actual previous carry.
The schedule is the complete source commit set; it may represent any legal
per-cell atomic order. No allocator, zero hook, trial or host return is run. -/
def program (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) : Code I F H Unit :=
  gridCode (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) events

def execute (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) (initial : Memory F H) :
    Unit × Memory F H := (program r out c selected divisor events).eval a initial

theorem execution_eq (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) (initial : Memory F H) :
    execute a r out c selected divisor events initial=
      ((),gridRun a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) events initial) :=
  eval_gridCode a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) events initial

theorem output (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (p : Prepared r out c) (selected : Selection (requestShape r) c) (divisor : F)
    (q : Schedule (requestShape r) c) (initial : Memory F H)
    (zero : Zeroed a (requestShape r) c (requestBuffers r out) initial)
    (cell : MatrixCell (requestShape r)) :
    read16 a (execute a r out c selected divisor q.events initial).2
      (outputAddress (requestShape r) (requestBuffers r out) cell)=
      wordReference a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) q initial cell := by
  rw [execution_eq]
  simp only [Prod.snd]
  rw [gridRun_cell a (requestShape r) c (requestBuffers r out) p.fresh selected divisor
    (hasBias r) q.events initial initial (fun _ _=>rfl)]
  by_cases single : c.S=1
  · exact cellFold_seed a c _ _ (q.complete cell) single _ _
  · rw [zeroed_read zero single cell]
    rfl

theorem output_word (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (p : Prepared r out c) (selected : Selection (requestShape r) c) (divisor : F)
    (q : Schedule (requestShape r) c) (initial : Memory F H)
    (zero : Zeroed a (requestShape r) c (requestBuffers r out) initial)
    (cell : MatrixCell (requestShape r)) :
    (execute a r out c selected divisor q.events initial).2
      (outputAddress (requestShape r) (requestBuffers r out) cell)=
      .inr (.inr (wordReference a (requestShape r) c (requestBuffers r out)
        selected divisor (hasBias r) q initial cell)) := by
  have nonempty : projection cell q.events≠[] := by
    intro h
    have len:=(q.complete cell).length_eq
    rw [h] at len
    simp only [List.length_nil,List.length_finRange] at len
    have positive:=c.positiveS
    omega
  obtain ⟨value,hv⟩:=gridRun_typed a (requestShape r) c (requestBuffers r out)
    selected divisor (hasBias r) q.events initial cell (Or.inr nonempty)
  have result:=output a r out c p selected divisor q initial zero cell
  rw [execution_eq] at result ⊢
  simp only [Prod.snd,read16,hv] at result ⊢
  rw [result]

/-- Every element outside the exact set of output cells is preserved,
including unused elements in the output storage itself. -/
theorem grid_frame (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool)
    (events : List (Commit s c)) (initial : Memory F H) (address : Address)
    (outside : ∀ cell : MatrixCell s,address≠outputAddress s b cell) :
    gridRun a s c b selected divisor hasBias events initial address=initial address := by
  induction events generalizing initial with
  | nil=>rfl
  | cons e es ih=>
    rw [gridRun,ih]
    simp only [commit,write16,if_neg (outside e.1)]

theorem frame (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) (initial : Memory F H) (address : Address)
    (outside : ∀ cell : MatrixCell (requestShape r),
      address≠outputAddress (requestShape r) (requestBuffers r out) cell) :
    (execute a r out c selected divisor events initial).2 address=initial address := by
  rw [execution_eq]
  exact grid_frame a (requestShape r) c (requestBuffers r out) selected divisor
    (hasBias r) events initial address outside

theorem native_execution (abi : LocalEventABI I F H σ a) (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) (initial : σ) :
    let result:=(program r out c selected divisor events).native abi initial
    (result.1,abi.observe result.2)=execute a r out c selected divisor events (abi.observe initial) :=
  Code.native_refines abi (program r out c selected divisor events) initial

theorem empty_execution (a : Ops I F H) (r : Request) (out : Nat) (c : Config)
    (selected : Selection (requestShape r) c) (divisor : F)
    (events : List (Commit (requestShape r) c)) (initial : Memory F H)
    (empty : (requestShape r).M=0 ∨ (requestShape r).N=0) :
    execute a r out c selected divisor events initial=((),initial) := by
  have noCell : ∀ cell : MatrixCell (requestShape r),False := by
    intro cell
    have hm:=cell.1.isLt
    have hn:=cell.2.isLt
    rcases empty with hm'|hn' <;> omega
  have nil : events=[] := by
    cases events with
    | nil=>rfl
    | cons e es=>exact False.elim (noCell e.1)
  rw [nil]
  rfl

#axiomsClean execution_eq
#axiomsClean output
#axiomsClean output_word
#axiomsClean frame
#axiomsClean native_execution
#axiomsClean empty_execution
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast

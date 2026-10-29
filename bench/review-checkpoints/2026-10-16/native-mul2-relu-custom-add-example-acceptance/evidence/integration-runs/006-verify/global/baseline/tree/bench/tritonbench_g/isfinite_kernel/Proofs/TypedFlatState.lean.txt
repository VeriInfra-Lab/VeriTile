import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Typed state relocation for finite allocation. Data channels and cell tags
are preserved; pointer payloads move to the flat region. Background cells and
the flat unspecified-value oracle remain explicit. This leaf proves state
and memory laws only, not execution of a translated kernel. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural

variable {α : Type} (A : FlatAlloc)

def trValue : (d : TileDType) → Value α d → Value α d
  | .real, v => v
  | .fp32, v => v
  | .fp16, v => v
  | .bf16, v => v
  | .f8e4, v => v
  | .f8e5, v => v
  | .nat, v => v
  | .int, v => v
  | .bool, v => v
  | .ptr, p => (A.flat, A.addr p.1 p.2)
  | .blockPtr, p =>
      { p with region := A.flat, baseOffset := A.addr p.region p.baseOffset }

def trCell : Cell α → Cell α
  | .mk d v => .mk d (trValue A d v)

def trValues {d : TileDType} {sh : TileShape} (v : Values α d sh) : Values α d sh :=
  fun i => trValue A d (v i)

def readFlat (background : RegionName → Nat → Cell α) (s : State α) (o : Nat) : Cell α :=
  match A.decode o with
  | some (r, i) => trCell A (s.mem r i)
  | none => background A.flat o

def flattenState (background : RegionName → Nat → Cell α)
    (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)
    (s : State α) : State α where
  mem := fun r o => if r = A.flat then readFlat A background s o else background r o
  regs := fun d sh n => (s.regs d sh n).map (trValues A)
  pids := s.pids
  numPids := s.numPids
  undef := flatUndef

theorem state_ext (s t : State α) (hm : s.mem = t.mem) (hr : s.regs = t.regs)
    (hp : s.pids = t.pids) (hn : s.numPids = t.numPids) (hu : s.undef = t.undef) : s = t := by
  cases s
  cases t
  simp_all

theorem trCell_read_data [Inhabited α] (c : Cell α) (d : TileDType)
    (hp : d ≠ .ptr) (hb : d ≠ .blockPtr) :
    (trCell A c).read d = c.read d := by
  cases c with
  | mk cd cv =>
      by_cases he : cd = d
      · subst cd
        cases d <;> simp_all [trCell, trValue, Cell.read, defaultValue]
      · simp [trCell, Cell.read, he]

theorem mem_addr (hd : A.Disjoint) (background flatUndef) (s : State α)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r) :
    (flattenState A background flatUndef s).mem A.flat (A.addr r o) = trCell A (s.mem r o) := by
  simp [flattenState, readFlat, A.decode_addr hd hr ho]

theorem readFlat_write (hd : A.Disjoint) (background) (s : State α)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (cell : Cell α) (o' : Nat) :
    readFlat A background (s.write r o cell) o' =
      if o' = A.addr r o then trCell A cell else readFlat A background s o' := by
  unfold readFlat
  cases hdec : A.decode o' with
  | none =>
      have hne : o' ≠ A.addr r o := fun he => by
        rw [he, A.decode_addr hd hr ho] at hdec
        simp at hdec
      rw [if_neg hne]
  | some ro =>
      obtain ⟨r2, o2⟩ := ro
      obtain ⟨_, henc, _⟩ := A.decode_sound hdec
      by_cases he : o' = A.addr r o
      · rw [if_pos he]
        rw [he, A.decode_addr hd hr ho] at hdec
        have h2 := Option.some_inj.mp hdec
        rw [Prod.mk.injEq] at h2
        obtain ⟨rfl, rfl⟩ := h2
        simp [State.write]
      · rw [if_neg he]
        have hcell : ¬(r2 = r ∧ o2 = o) := by
          rintro ⟨rfl, rfl⟩
          exact he henc
        simp [State.write, hcell]

theorem flatten_setReg (background flatUndef) (s : State α) (n : RegName)
    (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    flattenState A background flatUndef (s.setReg n d sh v) =
      (flattenState A background flatUndef s).setReg n d sh (trValues A v) := by
  apply state_ext <;> try rfl
  funext d' sh' n'
  simp only [flattenState, State.setReg]
  split
  · split
    · split
      · subst_vars; rfl
      · rfl
    · rfl
  · rfl

theorem flatten_write (hd : A.Disjoint) (background flatUndef) (s : State α)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (cell : Cell α) :
    flattenState A background flatUndef (s.write r o cell) =
      (flattenState A background flatUndef s).write A.flat (A.addr r o) (trCell A cell) := by
  apply state_ext <;> try rfl
  funext r' o'
  simp only [flattenState, State.write]
  by_cases hrf : r' = A.flat
  · subst r'
    rw [if_pos rfl]
    change readFlat A background (s.write r o cell) o' = _
    rw [readFlat_write A hd background s hr ho]
    simp only [true_and, ite_true]
  · simp only [hrf, if_false, false_and]

def FlatWritten (written : RegionName → Nat → Prop) (r : RegionName) (o : Nat) : Prop :=
  r = A.flat ∧ ∃ rr oo, rr ∈ A.regions ∧ oo < A.extent rr ∧
    written rr oo ∧ o = A.addr rr oo

theorem flatten_frame (background flatUndef) (initial final : State α)
    (written : RegionName → Nat → Prop)
    (hframe : ∀ r o, ¬written r o → final.mem r o = initial.mem r o)
    (r : RegionName) (o : Nat) (hmiss : ¬FlatWritten A written r o) :
    (flattenState A background flatUndef final).mem r o =
      (flattenState A background flatUndef initial).mem r o := by
  simp only [flattenState]
  by_cases hr : r = A.flat
  · subst r
    rw [if_pos rfl, if_pos rfl]
    unfold readFlat
    cases hdec : A.decode o with
    | none => rfl
    | some pair =>
        obtain ⟨rr, oo⟩ := pair
        obtain ⟨hrr, henc, hoo⟩ := A.decode_sound hdec
        change trCell A (final.mem rr oo) = trCell A (initial.mem rr oo)
        rw [hframe rr oo (fun hw => hmiss ⟨rfl, rr, oo, hrr, hoo, hw, henc⟩)]
  · rw [if_neg hr, if_neg hr]

#axiomsClean state_ext
#axiomsClean trCell_read_data
#axiomsClean mem_addr
#axiomsClean readFlat_write
#axiomsClean flatten_setReg
#axiomsClean flatten_write
#axiomsClean flatten_frame

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat

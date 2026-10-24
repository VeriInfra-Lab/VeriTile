import bench.repairs.tritonbench_g.int4_matmul.Proofs.Grid
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Packing
namespace VeriTile.Bench.Repairs.Int4Matmul.Source
set_option maxHeartbeats 4000000

theorem masked_nibble_zero (slot : Nat) : nibble (0 : Word) slot=0 := by simp [nibble]

/-- Literal four-load source dequantization. Each packed zero is an int32 word,
while the masked coefficient zero is a floating +0. -/
def loadedB (s : Shape) (p : Ops α) (m : State α) (col k : Nat) :=
  let word := if k<s.k then m.packed .weight (qAddress s k (col%s.n)) else (0 : Word)
  let zero := if k<s.k then m.packed .zero (zAddress s k (col%s.n)) else (0 : Word)
  let scale := if k<s.k then m.values .scale (sAddress s k (col%s.n)) else p.zero
  p.mul (p.fromInt (nibble word k-nibble zero (col%s.n))) scale

theorem loadedB_eq (s : Shape) (p : Ops α) (m : State α) (col k : Nat) :
    loadedB s p m col k=operandB s p m col k := by
  by_cases h : k<s.k <;> simp [loadedB,operandB,difference,h,nibble]

theorem shift_position (t : Tile) (ht : t.Valid) (sp block lane : Nat) :
    kIndex t sp block lane%8=(sp*t.bk+lane)%8 := by
  obtain ⟨q,hq⟩ := ht.bk8
  unfold kIndex
  rw [hq]
  have eq : sp*(8*q)+block*(8*q)*t.splits+lane=(sp*(8*q)+lane)+(block*q*t.splits)*8 := by ring
  rw [eq,Nat.add_mul_mod_self_right]

theorem mask_expression (t : Tile) (sp block lane : Nat) :
    sp*t.bk+lane+block*t.bk*t.splits=kIndex t sp block lane := by unfold kIndex;omega

def commit (s : Shape) (t : Tile) (p : Ops α) (w : Work) (value : α) (current : State α) :=
  store current .output (cAddress s w.row w.col)
    (if t.splits=1 then value else p.atomicAdd (current.values .output (cAddress s w.row w.col)) value)

/-- A vector program evaluates every dot tile from its load snapshot, then
scatters; atomics read their output cell at the actual commit point. -/
def prepared (s : Shape) (t : Tile) (p : Ops α) (snapshot : State α) : List Work → State α → State α
  | [],current => current
  | w::ws,current => prepared s t p snapshot ws (commit s t p w (partialValue s t p snapshot w) current)

theorem commit_eq (s : Shape) (t : Tile) (p : Ops α) (snapshot current : State α)
    (w : Work) (hw : s.Active t w) (h : Readonly snapshot current) :
    commit s t p w (partialValue s t p snapshot w) current=step s t p w current := by
  have hs := partial_reference s t p snapshot snapshot w hw ⟨fun _ _ _ => rfl,rfl⟩
  have hc := partial_reference s t p snapshot current w hw h
  simp only [commit,step,hs,hc]

theorem prepared_execute (s : Shape) (t : Tile) (p : Ops α) (snapshot current : State α)
    (ws : List Work) (active : ∀ w∈ws,s.Active t w) (h : Readonly snapshot current) :
    prepared s t p snapshot ws current=execute s t p ws current := by
  induction ws generalizing current with
  | nil => rfl
  | cons w ws ih =>
    have hw := active w (by simp)
    change prepared s t p snapshot ws (commit s t p w (partialValue s t p snapshot w) current)=_
    rw [commit_eq s t p snapshot current w hw h]
    exact ih (step s t p w current) (fun v hv => active v (by simp [hv]))
      (by unfold step;exact store_readonly snapshot current h _ _)

#axiomsClean loadedB_eq
#axiomsClean shift_position
#axiomsClean mask_expression
#axiomsClean prepared_execute
end VeriTile.Bench.Repairs.Int4Matmul.Source

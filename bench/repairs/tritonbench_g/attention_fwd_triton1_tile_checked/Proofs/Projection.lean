import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Loop

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Projection
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def inputs (ops : Ops α) (literal : ℝ → α) (c : Shape) (q k v : Nat → Term α) : Inputs α :=
  ⟨fun h r d => (q (c.mainOffset h r d)).evaluate ops literal,
   fun h r d => (k (c.mainOffset h r d)).evaluate ops literal,
   fun h r d => (v (c.mainOffset h r d)).evaluate ops literal⟩

variable (ops : Ops α) (literal : ℝ → α) (zero : literal 0=ops.zero)
  (c : Shape) (q k v : Nat → Term α) (scale : ℝ) (head chunk : Nat)
include zero

theorem query_eval (i : Fin 32) (j : Fin c.D) :
    (TermBody.query c q scale head chunk (i,j,PUnit.unit)).evaluate ops literal=
      Body.query ops c (inputs ops literal c q k v) (literal scale) head chunk i.val j.val := by
  by_cases active : chunk*32+i.val<c.T
  all_goals simp [TermBody.query,Term.evaluate,Body.query,masked,inputs,
    TermBody.mainAddress,Shape.mainOffset,active,j.isLt,zero]

theorem key_eval (i : Fin c.D) (j : Fin 32) :
    (TermBody.key c k head chunk (i,j,PUnit.unit)).evaluate ops literal=
      Body.key ops c (inputs ops literal c q k v) head chunk i.val j.val := by
  by_cases active : chunk*32+j.val<c.T
  all_goals simp [TermBody.key,Term.evaluate,Body.key,masked,inputs,
    TermBody.mainAddress,Shape.mainOffset,active,i.isLt,zero]

theorem value_eval (i : Fin 32) (j : Fin c.D) :
    (TermBody.value c v head chunk (i,j,PUnit.unit)).evaluate ops literal=
      Body.value ops c (inputs ops literal c q k v) head chunk i.val j.val := by
  by_cases active : chunk*32+i.val<c.T
  all_goals simp [TermBody.value,Term.evaluate,Body.value,masked,inputs,
    TermBody.mainAddress,Shape.mainOffset,active,j.isLt,zero]

theorem score_eval (i j : Fin 32) :
    (TermBody.score c q k scale head chunk (i,j,PUnit.unit)).evaluate ops literal=
      Body.score ops c (Body.query ops c (inputs ops literal c q k v) (literal scale) head chunk)
        (Body.key ops c (inputs ops literal c q k v) head chunk) i.val j.val := by
  simp only [TermBody.score,TermBody.dot,Term.evaluate]
  simp_rw [query_eval ops literal zero c q k v scale head chunk,
    key_eval ops literal zero c q k v head chunk]
  exact finChain_eq ops _ _ _ _

theorem within_eval (i : Fin 32) (j : Fin c.D) :
    (TermBody.within c q k v scale head chunk (i,j,PUnit.unit)).evaluate ops literal=
      Body.localOutput ops
        (Body.score ops c (Body.query ops c (inputs ops literal c q k v) (literal scale) head chunk)
          (Body.key ops c (inputs ops literal c q k v) head chunk))
        (Body.value ops c (inputs ops literal c q k v) head chunk) i.val j.val := by
  change finChain ops 32
    (fun d => ops.cast ((TermBody.score c q k scale head chunk (i,d,PUnit.unit)).evaluate ops literal))
    (fun d => (TermBody.value c v head chunk (d,j,PUnit.unit)).evaluate ops literal) ops.zero=_
  simp_rw [score_eval ops literal zero c q k v scale head chunk,
    value_eval ops literal zero c q k v head chunk]
  unfold finChain Body.localOutput
  apply chain_congr
  all_goals intro d hd; simp only [dif_pos hd]

theorem output_eval (carry : TermBody.Matrix α c.D c.D) (numericCarry : Tile α)
    (hc : ∀ i j : Fin c.D, (carry (i,j,PUnit.unit)).evaluate ops literal=numericCarry i.val j.val)
    (ifcond : Bool) (i : Fin 32) (j : Fin c.D) :
    (TermBody.output c q k v scale head chunk carry ifcond (i,j,PUnit.unit)).evaluate ops literal=
      (Body.evaluate ops c (inputs ops literal c q k v) (literal scale)
        head chunk numericCarry ifcond).output i.val j.val := by
  unfold TermBody.output Body.evaluate Body.output
  by_cases first : (ifcond && decide (chunk=0))=Bool.true
  · simp only [first,↓reduceIte]
    exact within_eval ops literal zero c q k v scale head chunk i j
  · simp only [first,Bool.false_eq_true,↓reduceIte,TermBody.add,TermBody.dot,Term.evaluate]
    simp only [query_eval ops literal zero c q k v scale head chunk,hc,
      within_eval ops literal zero c q k v scale head chunk]
    unfold finChain
    apply chain_congr
    all_goals intro d hd; simp only [dif_pos hd]

theorem carry_eval (carry : TermBody.Matrix α c.D c.D) (numericCarry : Tile α)
    (hc : ∀ i j : Fin c.D, (carry (i,j,PUnit.unit)).evaluate ops literal=numericCarry i.val j.val)
    (ifcond : Bool) (i j : Fin c.D) :
    (TermBody.nextCarry c k v head chunk carry ifcond (i,j,PUnit.unit)).evaluate ops literal=
      Body.nextCarry ops (Body.key ops c (inputs ops literal c q k v) head chunk)
        (Body.value ops c (inputs ops literal c q k v) head chunk) numericCarry
        (ifcond && decide (chunk=0)) i.val j.val := by
  unfold TermBody.nextCarry Body.nextCarry
  by_cases first : (ifcond && decide (chunk=0))=Bool.true
  all_goals simp only [first,Bool.false_eq_true,↓reduceIte,TermBody.add,TermBody.dot,Term.evaluate]
  all_goals simp only [key_eval ops literal zero c q k v head chunk,
    value_eval ops literal zero c q k v head chunk,hc]
  all_goals unfold finChain
  all_goals apply chain_congr
  all_goals intro d hd; simp only [dif_pos hd]

theorem graph_carry (ifcond : Bool) (n : Nat) (i j : Fin c.D) :
    (Graph.carry c k v head ifcond n (i,j,PUnit.unit)).evaluate ops literal=
      Reference.carry ops c (inputs ops literal c q k v) (literal scale) head ifcond n i.val j.val := by
  induction n generalizing i j with
  | zero => simp [Graph.carry,Reference.carry,Term.evaluate,zero]
  | succ n ih =>
    rw [Graph.carry]
    apply carry_eval ops literal zero c q k v head n
    intro a b
    exact ih a b

theorem graph_output (ifcond : Bool) (i : Fin 32) (j : Fin c.D) :
    (Graph.output c q k v scale head chunk ifcond (i,j,PUnit.unit)).evaluate ops literal=
      Reference.output ops c (inputs ops literal c q k v) (literal scale) head chunk ifcond i.val j.val := by
  apply output_eval ops literal zero c q k v scale head chunk
  intro a b
  exact graph_carry ops literal zero c q k v scale head ifcond chunk a b

#axiomsClean graph_carry
#axiomsClean graph_output
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Projection

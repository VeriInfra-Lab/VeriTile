import bench.tritonbench_g.rotary_emb_nopad.Proofs.Guards

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

def wordExtent (r : Request) (i : Fin 7) := (if i.val<5 then 1 else 2)*(expectedShape r i).prod

def destinationRole (r : Request) (key : Bool) : Fin 7 := if r.cached && key then 4 else role key

theorem tensor_facts (c : Call) (indices : Indices) (accepted : check c indices=true)
    (i : Fin 7) (presentI : present c.request i=true) :
    (c.tensor i).shape=expectedShape c.request i ∧
    (c.tensor i).elementBytes=(if i.val<5 then 4 else 8) ∧
    ((c.tensor i).shape.prod=0 ∨
      ((c.tensor i).base=(c.tensor i).allocationBase ∧ (c.tensor i).base=4*base c.request i ∧
       (c.tensor i).base%(c.tensor i).elementBytes=0 ∧
       (c.tensor i).elementBytes*(c.tensor i).shape.prod≤(c.tensor i).allocationBytes ∧
       (c.tensor i).base+(c.tensor i).allocationBytes≤2^64)) := by
  have h := tensor_checked c indices accepted
  simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  have hi := h.1 i presentI
  exact ⟨hi.1,hi.2.2.2.2.2.2.2.2.2.1,hi.2.2.2.2.2.2.2.2.2.2⟩

theorem physical_separate (c : Call) (indices : Indices) (accepted : check c indices=true)
    (i j : Fin 7) (pi : present c.request i=true) (pj : present c.request j=true) (different : i≠j)
    (positiveI : 0<wordExtent c.request i) (positiveJ : 0<wordExtent c.request j) :
    base c.request i+wordExtent c.request i≤base c.request j ∨
      base c.request j+wordExtent c.request j≤base c.request i := by
  have h := tensor_checked c indices accepted
  simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  have hi := tensor_facts c indices accepted i pi
  have hj := tensor_facts c indices accepted j pj
  have ni : (c.tensor i).shape.prod≠0 := by
    rw [hi.1]
    unfold wordExtent at positiveI
    intro z; rw [z,Nat.mul_zero] at positiveI; omega
  have nj : (c.tensor j).shape.prod≠0 := by
    rw [hj.1]
    unfold wordExtent at positiveJ
    intro z; rw [z,Nat.mul_zero] at positiveJ; omega
  have fi := hi.2.2.resolve_left ni
  have fj := hj.2.2.resolve_left nj
  have bytesI : 4*wordExtent c.request i≤(c.tensor i).allocationBytes := by
    have fit := fi.2.2.2.1
    rw [hi.1,hi.2.1] at fit
    unfold wordExtent
    split_ifs at * <;> omega
  have bytesJ : 4*wordExtent c.request j≤(c.tensor j).allocationBytes := by
    have fit := fj.2.2.2.1
    rw [hj.1,hj.2.1] at fit
    unfold wordExtent
    split_ifs at * <;> omega
  have separation := h.2 i j pi pj different
  simp only [separate,ni,nj,false_or] at separation
  rcases separation with before|after
  · left; omega
  · right; omega

theorem pair_bound (row col rows cols : Nat) (hr : row<rows) (hc : col<cols) : row*cols+col<rows*cols := by
  have mul := Nat.mul_le_mul_right cols (by omega : row+1≤rows)
  nlinarith

theorem mixed_injective (cols i j i' j' : Nat) (hj : j<cols) (hj' : j'<cols)
    (equal : i*cols+j=i'*cols+j') : i=i' ∧ j=j' := by
  have positive : 0<cols := by omega
  have division (a b : Nat) (hb : b<cols) : (a*cols+b)/cols=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hb]
    simp
  have eq := congrArg (fun x => x/cols) equal
  dsimp only at eq
  rw [division i j hj,division i' j' hj'] at eq
  exact ⟨eq,by simpa [eq] using equal⟩

theorem dim_bound (r : Request) (w : Work) (active : r.Active w) (odd : Bool) : r.dim w.pair odd<r.width := by
  have := active.2.2
  cases odd <;> simp [Request.dim,Request.width] <;> omega

theorem dim_injective (r : Request) (a b : Work) (ha : r.Active a) (hb : r.Active b) (oa ob : Bool)
    (same : r.dim a.pair oa=r.dim b.pair ob) : a.pair=b.pair ∧ oa=ob := by
  have := ha.2.2
  have := hb.2.2
  cases oa <;> cases ob <;> simp [Request.dim] at * <;> omega

theorem input_range (r : Request) (w : Work) (active : r.Active w) (odd : Bool) :
    base r (role w.key)≤r.data w.key w.token w.head (r.dim w.pair odd) ∧
    r.data w.key w.token w.head (r.dim w.pair odd)<base r (role w.key)+wordExtent r (role w.key) := by
  have bound := pair_bound _ _ _ _
    (pair_bound _ _ _ _ active.1 active.2.1) (dim_bound r w active odd)
  cases hk : w.key <;> simp [hk,Request.data,Request.heads,role,base,wordExtent,expectedShape,Nat.mul_assoc] at * <;> omega

theorem coefficient_range (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) (sine : Bool) :
    let i : Fin 7 := if sine then 3 else 2
    base c.request i≤c.request.coefficient sine w.token w.pair ∧
    c.request.coefficient sine w.token w.pair<base c.request i+wordExtent c.request i := by
  have h := shapes c indices accepted
  have t := active.1
  have j := active.2.2
  have cosBound := pair_bound w.token w.pair c.request.cosRows c.request.cosWidth (by omega) (by omega)
  have sinBound := pair_bound w.token w.pair c.request.sinRows c.request.sinWidth (by omega) (by omega)
  cases sine <;> simp [Request.coefficient,base,wordExtent,expectedShape] <;> omega

theorem destination_present (r : Request) (key : Bool) : present r (destinationRole r key)=true := by
  cases h : r.cached <;> cases key <;> simp [present,destinationRole,role,h]

theorem destination_range (c : Call) (indices : Indices) (accepted : check c indices=true)
    (w : Work) (active : c.request.Active w) (odd : Bool) :
    base c.request (destinationRole c.request w.key)≤
      c.request.destination indices w.key w.token w.head (c.request.dim w.pair odd) ∧
    c.request.destination indices w.key w.token w.head (c.request.dim w.pair odd)<
      base c.request (destinationRole c.request w.key)+wordExtent c.request (destinationRole c.request w.key) := by
  by_cases caching : (c.request.cached && w.key)=true
  · have flags : c.request.cached=true ∧ w.key=true := by simpa only [Bool.and_eq_true] using caching
    have hc := flags.1
    have hk := flags.2
    have slots := slots c indices accepted w active hc w.token active.1
    have head : w.head<c.request.kHeads := by simpa [Request.heads,hk] using active.2.1
    have row := pair_bound _ _ _ _ (pair_bound _ _ _ _ slots.1 head) slots.2
    have bound := pair_bound _ _ _ _ row (dim_bound c.request w active odd)
    simp [Request.destination,hc,hk,destinationRole,base,wordExtent,expectedShape,Nat.mul_assoc] at *
    omega
  · have same : destinationRole c.request w.key=role w.key := by simp [destinationRole,caching]
    simpa only [Request.destination,caching,↓reduceIte,same] using input_range c.request w active odd

end VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

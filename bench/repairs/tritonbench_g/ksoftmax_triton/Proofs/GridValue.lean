import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.RowValue

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def rightAddress (op : Operation) (Q K batch row lane : Nat) : Nat :=
  match op with
  | .forward _ _ mode => Forward.maskAddress mode batch row K lane
  | .backward _ _ => Native.flat Q K batch row lane

def value {α : Type} (M : Algebra α) (op : Operation) (K row : Nat)
    {D : Nat} (left right : Fin D → α) (i : Fin D) : Cell α :=
  match op with
  | .forward log causal mode => forwardValue M K row log causal mode left right i
  | .backward log causal => backwardValue M K row log causal left right i

/-- Reference over immutable logical input coordinates, not an execution. -/
def gridValue {α : Type} (M : Algebra α) (op : Operation) (Q K batch row : Nat)
    {D : Nat} (left right : Nat → α) (i : Fin D) : Cell α :=
  value M op K row (fun j => left (Native.flat Q K batch row j.val))
    (fun j => right (rightAddress op Q K batch row j.val)) i

theorem row_value {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (Q K D : Nat) (hD : 0<D)
    (batch row : Nat) (s : State α) (i : Fin D) (hi : i.val<K) :
    (runRow M op Dst Left Right Q K D hD batch row s).mem Dst (Native.flat Q K batch row i.val)=
      gridValue M op Q K batch row (fun j => (s.mem Left j).read .real)
        (fun j => (s.mem Right j).read .real) i := by
  cases op with
  | forward log causal mode =>
    simpa only [runRow,gridValue,value,rightAddress,atProgram,Native.flat,ite_true,
      show ¬(1:Nat)=0 by decide,ite_false] using
      forward_value M Dst Left Right (Q*K) K (Q*K) K K K D hD log mode causal
        (atProgram s batch row) i hi
  | backward log causal =>
    simpa only [runRow,gridValue,value,rightAddress,atProgram,Native.flat,ite_true,
      show ¬(1:Nat)=0 by decide,ite_false] using
      backward_value M Dst Left Right (Q*K) K (Q*K) K (Q*K) K K D log causal
        (atProgram s batch row) i hi

theorem row_other {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (batch otherBatch : Fin B) (row otherRow : Fin Q) (s : State α)
    (i : Fin D) (hi : i.val<K) (different : (batch,row)≠(otherBatch,otherRow)) :
    (runRow M op Dst Left Right Q K D hD batch row s).mem Dst
        (Native.flat Q K otherBatch otherRow i.val)=s.mem Dst (Native.flat Q K otherBatch otherRow i.val) := by
  have miss : ∀ j : Fin D, j.val<K → Native.flat Q K otherBatch otherRow i.val≠
      (atProgram s batch row).pids 0*(Q*K)+(atProgram s batch row).pids 1*K+j.val := by
    intro j hj eq
    have eq' : Native.flat Q K otherBatch otherRow i.val=Native.flat Q K batch row j.val := by
      simpa only [atProgram,ite_true,show ¬(1:Nat)=0 by decide,ite_false,Native.flat] using eq
    obtain ⟨hb,hr,_⟩:=Native.flat_injective Q K _ _ _ _ _ _ otherRow.isLt row.isLt hi hj eq'
    apply different
    exact Prod.ext (Fin.ext hb.symm) (Fin.ext hr.symm)
  cases op with
  | forward log causal mode =>
    exact Floating.forward_frame M Dst Left Right (Q*K) K (Q*K) K K K D hD log mode causal
      (atProgram s batch row) Dst _ (Or.inr miss)
  | backward log causal =>
    exact Floating.backward_frame M Dst Left Right (Q*K) K (Q*K) K (Q*K) K K D log causal
      (atProgram s batch row) Dst _ (Or.inr miss)

theorem gridValue_after_row {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (batch : Fin B) (row : Fin Q) (targetBatch targetRow : Nat) (s : State α)
    (i : Fin D) (leftSeparate : Left≠Dst) (rightSeparate : Right≠Dst) :
    let t:=runRow M op Dst Left Right Q K D hD batch row s
    gridValue M op Q K targetBatch targetRow (fun j => (t.mem Left j).read .real)
      (fun j => (t.mem Right j).read .real) i=
    gridValue M op Q K targetBatch targetRow (fun j => (s.mem Left j).read .real)
      (fun j => (s.mem Right j).read .real) i := by
  dsimp only
  have hl : (fun j => ((runRow M op Dst Left Right Q K D hD batch row s).mem Left j).read .real)=
      (fun j => (s.mem Left j).read .real) := by
    funext j
    rw [row_frame M op Dst Left Right B Q K D hD batch row s Left j (Or.inl leftSeparate)]
  have hr : (fun j => ((runRow M op Dst Left Right Q K D hD batch row s).mem Right j).read .real)=
      (fun j => (s.mem Right j).read .real) := by
    funext j
    rw [row_frame M op Dst Left Right B Q K D hD batch row s Right j (Or.inl rightSeparate)]
  rw [hl,hr]

/-- Arbitrary schedules, including repeats, preserve the unique row result.
Each step is the proved actual source execution, not a row-output oracle. -/
theorem rows_observe {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (rows : List (Fin B × Fin Q)) (s : State α) (batch : Fin B) (row : Fin Q)
    (i : Fin D) (hi : i.val<K) (leftSeparate : Left≠Dst) (rightSeparate : Right≠Dst) :
    (runRows M op Dst Left Right B Q K D hD rows s).mem Dst (Native.flat Q K batch row i.val)=
      if (batch,row)∈rows then gridValue M op Q K batch row
        (fun j => (s.mem Left j).read .real) (fun j => (s.mem Right j).read .real) i
      else s.mem Dst (Native.flat Q K batch row i.val) := by
  induction rows generalizing s with
  | nil => simp [runRows]
  | cons p rest ih =>
    change (runRows M op Dst Left Right B Q K D hD rest
      (runRow M op Dst Left Right Q K D hD p.1 p.2 s)).mem Dst _ = _
    rw [ih,gridValue_after_row M op Dst Left Right B Q K D hD p.1 p.2 _ _ s i leftSeparate rightSeparate]
    by_cases member : (batch,row)∈rest
    · simp [member]
    · by_cases same : p=(batch,row)
      · subst p
        simp only [member,if_false,List.mem_cons,eq_self,true_or,if_true]
        exact row_value M op Dst Left Right Q K D hD batch row s i hi
      · simp only [member,if_false,List.mem_cons,Prod.mk.eta,Ne.symm same,false_or]
        exact row_other M op Dst Left Right B Q K D hD p.1 batch p.2 row s i hi same

theorem programs_mem (B Q : Nat) (batch : Fin B) (row : Fin Q) : (batch,row)∈programs B Q := by
  simp [programs,List.mem_flatMap]

theorem grid_observe {α : Type} [Inhabited α] (M : Algebra α) (op : Operation)
    (Dst Left Right : RegionName) (B Q K D : Nat) (hD : 0<D)
    (s : State α) (batch : Fin B) (row : Fin Q) (i : Fin D) (hi : i.val<K)
    (leftSeparate : Left≠Dst) (rightSeparate : Right≠Dst) :
    (runRows M op Dst Left Right B Q K D hD (programs B Q) s).mem Dst (Native.flat Q K batch row i.val)=
      gridValue M op Q K batch row (fun j => (s.mem Left j).read .real)
        (fun j => (s.mem Right j).read .real) i := by
  simpa only [programs_mem,if_true] using rows_observe M op Dst Left Right B Q K D hD
    (programs B Q) s batch row i hi leftSeparate rightSeparate

#specNonCircular gridValue avoiding [runRows,runRow,Floating.forward_run,Floating.backward_run]
#axiomsClean rows_observe
#axiomsClean grid_observe
end
end VeriTile.Bench.Repairs.Ksoftmax

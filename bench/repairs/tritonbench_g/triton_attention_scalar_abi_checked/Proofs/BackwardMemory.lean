import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.Grid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

theorem scatter_mem {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (l : List ι) (s : State α) :
    (l.foldl (fun s i=>s.write r (off i) (v i)) s).mem=
      l.foldl (fun m i=>writeMemory m r (off i) (v i)) s.mem := by
  induction l generalizing s with
  | nil=>rfl
  | cons i l ih=>rw [List.foldl_cons,List.foldl_cons,ih];rfl

theorem memoryFold_readback (r : RegionName) (off : TileIndex shape → Nat) (v : TileIndex shape → Cell α)
    (mem : Memory α) (i : TileIndex shape) (hinj : Function.Injective off) :
    ((TileShape.allIndices shape).foldl (fun m i=>writeMemory m r (off i) (v i)) mem) r (off i)=v i := by
  have h:=State.scatter_readback r off v (Grid.entry (1,1) (0,0) mem) i hinj
  rw [scatter_mem] at h
  exact h

theorem backwardOffset_inj (p : BackwardParameters) (s : State α) (b b' : Nat)
    (i j : BLane p) (h : backwardOffset p s b i=backwardOffset p s b' j) :
    b=b' ∧ i=j := by
  have hi:=i.2.1.isLt
  have hj:=j.2.1.isLt
  have hid:=i.1.isLt
  have hjd:=j.1.isLt
  have hcol : i.2.1=j.2.1 := by
    apply Fin.ext
    have hh:=congrArg (· % p.head) h
    simpa only [backwardOffset,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hi,Nat.mod_eq_of_lt hj] using hh
  have hdim : 0<p.head := by omega
  have hrow : b*p.blockM+i.1.val=b'*p.blockM+j.1.val := by
    have hc:=congrArg Fin.val hcol
    simp only [backwardOffset] at h
    nlinarith
  have hBM : 0<p.blockM := by omega
  have rowDiv (x : Nat) (i : Fin p.blockM) : (x*p.blockM+i.val)/p.blockM=x := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hBM,Nat.div_eq_of_lt i.isLt,Nat.zero_add]
  have hblock : b=b' := by
    simpa only [rowDiv] using congrArg (· / p.blockM) hrow
  have hlane : i.1=j.1 := by
    apply Fin.ext
    simpa only [hblock,Nat.add_left_cancel_iff] using hrow
  exact ⟨hblock,Prod.ext hlane (Prod.ext hcol (Subsingleton.elim _ _))⟩

theorem backwardOffset_injective (p : BackwardParameters) (s : State α) (b : Nat) :
    Function.Injective (backwardOffset p s b) := by
  intro i j h
  exact (backwardOffset_inj p s b b i j h).2

theorem backwardInner_readonly (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (keyBlock n : Nat) (r : RegionName) (off : Nat) (hr : r≠p.dq) :
    (backwardInnerRounds P p s mem keyBlock n).mem r off=mem r off := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    change ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m i=>writeMemory m p.dq
      (backwardOffset p s (keyBlock+n) i) _) _) r off=_
    rw [memoryFold_frame _ _ _ _ _ _ _ (by intro i hi;exact Or.inl hr),ih]

theorem backwardInner_otherBlock (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (keyBlock n b : Nat) (i : BLane p)
    (hb : b<keyBlock ∨ keyBlock+n≤b) :
    (backwardInnerRounds P p s mem keyBlock n).mem p.dq (backwardOffset p s b i)=
      mem p.dq (backwardOffset p s b i) := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    change ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m j=>writeMemory m p.dq
      (backwardOffset p s (keyBlock+n) j) _) _) p.dq (backwardOffset p s b i)=_
    rw [memoryFold_frame]
    · exact ih (by omega)
    · intro j _
      right
      intro he
      have:= (backwardOffset_inj p s b (keyBlock+n) i j he).1
      omega

#axiomsClean memoryFold_readback
#axiomsClean backwardInner_otherBlock
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source

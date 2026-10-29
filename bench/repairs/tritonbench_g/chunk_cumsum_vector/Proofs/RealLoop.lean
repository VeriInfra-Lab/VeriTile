import bench.tritonbench_g.reversed_cumsum.Proofs.SignedRange
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealStep
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealData

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealLoop
open VeriTile.Triton IndexProgram RealProgram RealValues RealStep RealData
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def active (h : Call) (p : Geometry.Program h) (t : Nat) (j : Fin BS) : Prop :=
  t<h.T ∧ p.feature*BS+j.val<h.S
def address (h : Call) (p : Geometry.Program h) (t : Nat) (j : Fin BS) : Nat :=
  h.outputStrides.address h.H p.row t (p.feature*BS+j.val)
def NoCollision (h : Call) (p : Geometry.Program h) : Prop :=
  ∀ (t u : Nat) (i j : Fin BS), active h p t i → active h p u j →
    address h p t i=address h p u j → t=u ∧ i=j
def footprint (h : Call) (p : Geometry.Program h) : WriteFootprint :=
  fun cell => cell.1=h.output ∧ ∃ (t : Nat) (j : Fin BS), active h p t j ∧ cell.2=address h p t j

structure Holds (initial : BlockState) (h : Call) (c : Config) (p : Geometry.Program h)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) : Prop where
  undef : u.undef=initial.undef
  count : q≤Geometry.chunks h c
  batch : u.regs .int [] "i_b"=some (Tile.scalar (p.row/h.H))
  head : u.regs .int [] "i_h"=some (Tile.scalar (p.row%h.H))
  mask : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (p.feature*BS))
  columns : u.regs .int [BS] "columns"=some (coordinate h.S BS (p.feature*BS))
  triangular : u.regs .real [c.BT,c.BT] "m_s"=some (lowerTriTile c.BT)
  accumulated : u.regs .real [BS] "b_z"=some (Tile.vec fun j =>
    some (carry c (value h c p.feature p.row a (initial.undef h.input)) q j))
  input : ∀ off, u.readMem h.input off=a off
  output : ∀ (t : Nat) (j : Fin BS), active h p t j → t<q*c.BT →
    u.readMem h.output (address h p t j)=expected h p.feature p.row a t j
  frame : BlockState.WriteWithin (footprint h p) initial u

theorem scatter_frame (h : Call) (c : Config) (hc : 0<c.BT) (p : Geometry.Program h)
    (q : Nat) (hq : q<Geometry.chunks h c) (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint h p) u (scatter h c p.feature p.row q f acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  have coords := Geometry.active_coordinates h c p q idx.1.val idx.2.1.val ha
  have bounds := (Geometry.active_iff h c p q idx.1.val idx.2.1.val (Geometry.chunk_start h c hc q hq)).mp ha
  refine ⟨rfl,q*c.BT+idx.1.val,idx.2.1,bounds,?_⟩
  simp only [coords.1,coords.2,address]

theorem scatter_undef (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState) :
    (scatter h c feature row q f acc u).undef=u.undef := by
  unfold scatter
  generalize TileShape.allIndices [c.BT,BS]=xs
  induction xs generalizing u with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    split <;> rw [ih] <;> rfl

theorem iteration (initial : BlockState) (h : Call) (c : Config) (hc : 0<c.BT)
    (p : Geometry.Program h) (a : Nat → ℝ) (distinct : h.input≠h.output)
    (unique : NoCollision h p) (q : Nat) (u : BlockState)
    (hq : q<Geometry.chunks h c) (inv : Holds initial h c p a q u) :
    ∃ v, stepStmts (loopBody h c) (u.setReg "i_t" .int [] (Tile.scalar q))=some v ∧
      Holds initial h c p a (q+1) v := by
  let f := value h c p.feature p.row a (initial.undef h.input)
  let acc := carry c f q
  let entry := u.setReg "i_t" .int [] (Tile.scalar q)
  let pre := beforeStore h c p.feature p.row q f acc entry
  let v := result h c p.feature p.row q f acc entry
  have he := RealStep.executes h c p.feature p.row q f acc entry
    (by simp [entry]) (by simp [entry,inv.batch]) (by simp [entry,inv.head])
    (by simp [entry,inv.mask]) (by simp [entry,inv.columns])
    (by simp [entry,inv.triangular]) (by simpa [entry,acc,f] using inv.accumulated)
    (by intro i j; simpa [entry,inv.input,inv.undef,f] using value_chunk h c hc p a (initial.undef h.input) q hq i j)
  refine ⟨v,he,?_,Nat.succ_le_of_lt hq,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact (scatter_undef h c p.feature p.row q f acc pre).trans inv.undef
  · simp [v,result,scatter,beforeStore,entry,inv.batch]
  · simp [v,result,scatter,beforeStore,entry,inv.head]
  · simp [v,result,scatter,beforeStore,entry,inv.mask]
  · simp [v,result,scatter,beforeStore,entry,inv.columns]
  · simp [v,result,scatter,beforeStore,entry,inv.triangular]
  · simp [v,result,acc,f,next_carry]
  · intro off
    change (scatter h c p.feature p.row q f acc pre).readMem h.input off=a off
    rw [scatter,BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ distinct]
    simpa [pre,beforeStore,entry] using inv.input off
  · intro t j ha ht
    change (scatter h c p.feature p.row q f acc pre).readMem h.output (address h p t j)=_
    have start := Geometry.chunk_start h c hc q hq
    by_cases prior : t<q*c.BT
    · rw [scatter,BlockState.scatter_prop_masked_preserves_other_offset]
      · simpa [pre,beforeStore,entry] using inv.output t j ha prior
      · intro idx hi heq
        have coords := Geometry.active_coordinates h c p q idx.1.val idx.2.1.val hi
        have bounds := (Geometry.active_iff h c p q idx.1.val idx.2.1.val start).mp hi
        rw [coords.1,coords.2] at heq
        have eq := (unique _ _ idx.2.1 j bounds ha heq).1
        omega
    · have hi : t-q*c.BT<c.BT := by rw [Nat.add_mul,Nat.one_mul] at ht; omega
      let i : Fin c.BT := ⟨t-q*c.BT,hi⟩
      have ti : q*c.BT+i.val=t := by dsimp [i]; omega
      have hm : Geometry.active h c p.feature q i.val j.val :=
        (Geometry.active_iff h c p q i.val j.val start).mpr (by simpa [ti] using ha)
      have coords := Geometry.active_coordinates h c p q i.val j.val hm
      have read := BlockState.scatter_readback_prop_masked_nd_of_true
        (region:=h.output) pre
        (fun idx : TileIndex [c.BT,BS] => h.outputStrides.address h.H p.row
          (Geometry.time h c q idx.1.val) (Geometry.column h p.feature idx.2.1.val))
        (fun idx => acc idx.2.1+chunkPrefix c f q idx.1 idx.2.1)
        (fun idx => Geometry.active h c p.feature q idx.1.val idx.2.1.val)
        (i,j,PUnit.unit) hm (by
          rintro ⟨k,e,⟨⟩⟩ hk heq
          have kc := Geometry.active_coordinates h c p q k.val e.val hk
          have kb := (Geometry.active_iff h c p q k.val e.val start).mp hk
          dsimp only at heq
          rw [kc.1,kc.2,coords.1,coords.2] at heq
          obtain ⟨ht,hj⟩ := unique _ _ e j kb ((Geometry.active_iff h c p q i.val j.val start).mp hm) heq
          have ki : k=i := Fin.ext (by omega)
          subst k; subst e; rfl)
      change (scatter h c p.feature p.row q f acc pre).readMem h.output
        (h.outputStrides.address h.H p.row (Geometry.time h c q i.val) (Geometry.column h p.feature j.val))=_ at read
      rw [coords.1,coords.2,ti] at read
      rw [address,read]
      exact (prefix_at_chunk h c p a (initial.undef h.input) q i j (by simpa [ti] using ha.1) ha.2).trans (by rw [ti])
  · have frame := scatter_frame h c hc p q hq f acc pre
    intro r off hn
    exact (inv.frame r off hn).trans (frame r off hn)

def statement (h : Call) (c : Config) : Stmt :=
  .forRangeInt "i_t" (.constInt 0) (.castNatToInt (countOp h c)) (.constNat 1) (loopBody h c)
theorem count_eval (h : Call) (c : Config) (u : BlockState) :
    evalOp (countOp h c) u=some (Tile.scalar (Geometry.chunks h c)) := by
  simp only [countOp,evalOp_constNat]


theorem executes (initial : BlockState) (h : Call) (c : Config) (hc : 0<c.BT)
    (p : Geometry.Program h) (a : Nat → ℝ) (distinct : h.input≠h.output)
    (unique : NoCollision h p) (u : BlockState) (inv : Holds initial h c p a 0 u) :
    ∃ v, stepStmt (statement h c) u=some v ∧ Holds initial h c p a (Geometry.chunks h c) v := by
  let P := fun j : Int => fun v : BlockState =>
    0 ≤ j ∧ Holds initial h c p a j.toNat v
  have step (j : Int) (v : BlockState) (hj : j < (Geometry.chunks h c:Int)) (hv : P j v) :
      ∃ w, stepStmts (loopBody h c) (v.setReg "i_t" .int [] (Tile.scalar j))=some w ∧ P (j+1) w := by
    have inside : j.toNat < Geometry.chunks h c := by dsimp [P] at hv; omega
    obtain ⟨w,he,hw⟩ := iteration initial h c hc p a distinct unique j.toNat v inside hv.2
    refine ⟨w,?_,by dsimp [P] at *; exact ⟨by omega,by simpa only [show (j+1).toNat=j.toNat+1 by omega] using hw⟩⟩
    simpa only [Int.toNat_of_nonneg hv.1] using he
  obtain ⟨final,v,he,hf,hv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange.executes
    "i_t" (Geometry.chunks h c:Int) 1 (loopBody h c) P (by decide) step 0 u ⟨by omega,by simpa using inv⟩
  have eq : final.toNat=Geometry.chunks h c := by have := hv.2.count; dsimp [P] at hv; omega
  refine ⟨v,?_,by simpa only [eq] using hv.2⟩
  rw [statement,stepForRangeIntAux.forRangeInt_unfold]
  simpa only [evalOp,Option.bind_some,Option.bind_eq_bind,countOp] using he

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealLoop

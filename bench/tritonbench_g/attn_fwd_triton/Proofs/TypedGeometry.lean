import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedFrame
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

abbrev Coordinate (p : Config) := Fin p.batches × Fin p.heads × Fin p.context × Fin p.headDim

def logicalAddress (p : Config) (c : Coordinate p) :=
  c.1.val*p.outputBatch+c.2.1.val*p.outputHead+c.2.2.1.val*p.outputRow+c.2.2.2.val*p.outputDim

def NonOverlapping (p : Config) := Function.Injective (logicalAddress p)

def rowCount (p : Config) := if p.headDim=0 then 0 else (p.context+127)/128
abbrev ProgramIndex (p : Config) := Fin (rowCount p) × Fin (p.batches*p.heads)

def atProgram {α : Type} (p : Config) (s : State α) (idx : ProgramIndex p) : State α :=
  {s with pids := fun ax => if ax=0 then idx.1.val else if ax=1 then idx.2.val else 0
          numPids := fun ax => if ax=0 then rowCount p else if ax=1 then p.batches*p.heads else 1}

def coordinate (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (i : TileIndex [128,BD]) (hi : outputActive p BD idx.1.val i=Bool.true) : Coordinate p :=
  have hH : 0<p.heads := by have := idx.2.isLt; nlinarith
  have hs : idx.1.val*128+i.1.val<p.context ∧ i.2.1.val<p.headDim := by simpa [outputActive] using hi
  (⟨idx.2.val/p.heads,(Nat.div_lt_iff_lt_mul hH).mpr idx.2.isLt⟩,
    ⟨idx.2.val%p.heads,Nat.mod_lt _ hH⟩,⟨idx.1.val*128+i.1.val,hs.1⟩,⟨i.2.1.val,hs.2⟩)

theorem coordinate_address (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (i : TileIndex [128,BD]) (hi : outputActive p BD idx.1.val i=Bool.true) :
    logicalAddress p (coordinate p BD idx i hi)=outputAddress p BD idx.1.val idx.2.val i := rfl

theorem disjoint (p : Config) (BD : Nat) (hn : NonOverlapping p)
    (a b : ProgramIndex p) (region : RegionName) (offset : Nat)
    (ha : Written p BD a.1.val a.2.val region offset)
    (hb : Written p BD b.1.val b.2.val region offset) : a=b := by
  rcases ha with ⟨_,i,hi,ha⟩
  rcases hb with ⟨_,j,hj,hb⟩
  have he : logicalAddress p (coordinate p BD a i hi)=logicalAddress p (coordinate p BD b j hj) := ha.symm.trans hb
  have hc := hn he
  have hr := congrArg (fun c : Coordinate p => c.2.2.1.val) hc
  have hz := congrArg (fun c : Coordinate p => c.1.val) hc
  have hh := congrArg (fun c : Coordinate p => c.2.1.val) hc
  change a.1.val*128+i.1.val=b.1.val*128+j.1.val at hr
  change a.2.val/p.heads=b.2.val/p.heads at hz
  change a.2.val%p.heads=b.2.val%p.heads at hh
  have hrow : a.1.val=b.1.val := by have := i.1.isLt; have := j.1.isLt; omega
  have hhead : a.2.val=b.2.val := by
    have h₁ := Nat.mod_add_div a.2.val p.heads
    have h₂ := Nat.mod_add_div b.2.val p.heads
    rw [hz,hh] at h₁
    omega
  exact Prod.ext (Fin.ext hrow) (Fin.ext hhead)

def owner (p : Config) (c : Coordinate p) : ProgramIndex p :=
  (⟨c.2.2.1.val/128,by
      have hD : p.headDim≠0 := Nat.ne_zero_of_lt c.2.2.2.isLt
      simp only [rowCount,hD,↓reduceIte]
      have := c.2.2.1.isLt
      omega⟩,
    ⟨c.1.val*p.heads+c.2.1.val,by
      have := c.1.isLt
      have := c.2.1.isLt
      nlinarith⟩)

def lane (p : Config) (BD : Nat) (hD : p.headDim≤BD) (c : Coordinate p) : TileIndex [128,BD] :=
  (⟨c.2.2.1.val%128,Nat.mod_lt _ (by decide)⟩,⟨c.2.2.2.val,lt_of_lt_of_le c.2.2.2.isLt hD⟩,PUnit.unit)

theorem lane_row (p : Config) (BD : Nat) (hD : p.headDim≤BD) (c : Coordinate p) :
    (owner p c).1.val*128+(lane p BD hD c).1.val=c.2.2.1.val := by
  simpa [owner,lane,Nat.mul_comm] using Nat.div_add_mod c.2.2.1.val 128

theorem lane_active (p : Config) (BD : Nat) (hD : p.headDim≤BD) (c : Coordinate p) :
    outputActive p BD (owner p c).1.val (lane p BD hD c)=Bool.true := by
  simp only [outputActive,lane_row]
  simp [lane,c.2.2.1.isLt,c.2.2.2.isLt]

theorem lane_address (p : Config) (BD : Nat) (hD : p.headDim≤BD) (c : Coordinate p) :
    outputAddress p BD (owner p c).1.val (owner p c).2.val (lane p BD hD c)=logicalAddress p c := by
  have hz : (owner p c).2.val/p.heads=c.1.val := by
    have hH : 0<p.heads := lt_of_le_of_lt (Nat.zero_le _) c.2.1.isLt
    simp [owner,Nat.add_div,Nat.div_eq_of_lt c.2.1.isLt,hH,Nat.mul_div_cancel,Nat.mod_eq_of_lt c.2.1.isLt,c.2.1.isLt]
  have hh : (owner p c).2.val%p.heads=c.2.1.val := by
    simp [owner,Nat.add_mod,Nat.mod_eq_of_lt c.2.1.isLt]
  simp only [outputAddress,hz,hh,lane_row]
  rfl

def Window (p : Config) (region : RegionName) (offset : Nat) : Prop :=
  region=p.Out ∧ ∃ c : Coordinate p, offset=logicalAddress p c

theorem written_within (p : Config) (BD : Nat) (idx : ProgramIndex p) (region : RegionName) (offset : Nat)
    (h : Written p BD idx.1.val idx.2.val region offset) : Window p region offset := by
  rcases h with ⟨hr,i,hi,he⟩
  exact ⟨hr,coordinate p BD idx i hi,he⟩

theorem covered (p : Config) (BD : Nat) (hD : p.headDim≤BD) (c : Coordinate p) :
    Written p BD (owner p c).1.val (owner p c).2.val p.Out (logicalAddress p c) :=
  ⟨rfl,lane p BD hD c,lane_active p BD hD c,(lane_address p BD hD c).symm⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

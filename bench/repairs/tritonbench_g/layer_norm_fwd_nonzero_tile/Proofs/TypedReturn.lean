import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedGrid
import bench.tritonbench_g.layer_norm_fwd.Proofs.HostReturn

/-! Eight returned tensor contents in the selected opaque FP source model.
Wrapper object identity, setup failures and native lowering are distinct. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 2000000

@[ext] structure Result (α : Type) (M N seedCount : Nat) where
  y : Fin M → Fin N → Cell α
  y1 : Option (Fin M → Fin N → Cell α)
  mean : Option (Fin M → Cell α)
  rstd : Fin M → Cell α
  residual : Fin M → Fin N → Cell α
  seeds : Option (Fin seedCount → Cell α)
  mask : Option (Fin M → Fin N → Cell α)
  mask1 : Option (Fin M → Fin N → Cell α)

def observe {α : Type} (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (s : State α) : Result α M N (HostReturn.seedCount f M) where
  y := fun i j => s.mem r.y (i.val*stride.y+j.val)
  y1 := if f.w1 then some (fun i j => s.mem r.y1 (i.val*stride.y1+j.val)) else none
  mean := if f.rms then none else some (fun i => s.mem r.mean i.val)
  rstd := fun i => s.mem r.rstd i.val
  residual := if f.residualOut then (fun i j => s.mem r.residualOut (i.val*stride.residualOut+j.val))
    else (fun i j => s.mem r.x (i.val*stride.x+j.val))
  seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
  mask := if f.dropout && f.dropoutMask then some (fun i j => s.mem r.dropoutMask (i.val*N+j.val)) else none
  mask1 := if f.dropout && f.dropoutMask && f.x1 then
    some (fun i j => s.mem r.dropoutMask ((M+i.val)*N+j.val)) else none

def lane {N block : Nat} (hN : N≤block) (j : Fin N) : TileIndex [block] :=
  (⟨j.val,lt_of_lt_of_le j.isLt hN⟩,PUnit.unit)

def oracle {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hN : N≤block) (s : State α) : Result α M N (HostReturn.seedCount f M) :=
  let v := fun row => combinedValues A cmp f s r stride M N p row block
  { y := fun i j => Cell.mk .real (affineValue A f.bias (normalizedValues A f.rms N eps (v i.val) (lane hN j))
      (raw A s r.weight 0 N j.val) (raw A s r.bias 0 N j.val))
    y1 := if f.w1 then some (fun i j => Cell.mk .real
      (affineValue A f.b1 (normalizedValues A f.rms N eps (v i.val) (lane hN j))
        (raw A s r.weight1 0 N j.val) (raw A s r.bias1 0 N j.val))) else none
    mean := if f.rms then none else some (fun i => Cell.mk .real (meanValue A N (v i.val)))
    rstd := fun i => Cell.mk .real (rstdValue A f.rms N eps (v i.val))
    residual := if f.residualOut then (fun i j => Cell.mk .real (v i.val (lane hN j)))
      else (fun i j => s.mem r.x (i.val*stride.x+j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then
      some (fun i j => Cell.mk .bool (keepValue A cmp ((s.mem r.seeds i.val).read .int) p j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then
      some (fun i j => Cell.mk .bool (keepValue A cmp ((s.mem r.seeds (M+i.val)).read .int) p j.val)) else none }

theorem grid_preserves_input {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (s t : State α) (hsep : Domain.ReadWriteSeparated f r)
    (hp : GridPost A cmp f r stride M N eps p block s t)
    (region : RegionName) (hr : Domain.readRegion f r region) (offset : Nat) :
    t.mem region offset=s.mem region offset := by
  apply hp.2 region offset
  intro i hw
  exact hsep region hr region (written_is_writeRegion hw) rfl

theorem observe_eq_oracle {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hN : N≤block) (s t : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hp : GridPost A cmp f r stride M N eps p block s t) :
    observe f r stride M N t=oracle A cmp f r stride M N eps p block hN s := by
  have hy := funext fun i => funext fun j => (hp.1 i).2.1 (lane hN j) j.isLt
  have hy1 := fun hf => funext fun i => funext fun j => (hp.1 i).2.2.1 hf (lane hN j) j.isLt
  have hs := funext fun i => (hp.1 i).2.2.2.1
  have hm := fun hf => funext fun i => (hp.1 i).2.2.2.2 hf
  have hr := fun hf => funext fun i => funext fun j => (hp.1 i).1.1 hf (lane hN j) j.isLt
  have hx := funext fun (i : Fin M) => funext fun (j : Fin N) =>
    grid_preserves_input A cmp f r stride M N eps p block s t hsep hp r.x
      (by simp [Domain.readRegion]) (i.val*stride.x+j.val)
  have hseed := fun (hd : f.dropout=Bool.true) => funext fun (i : Fin (HostReturn.seedCount f M)) =>
    grid_preserves_input A cmp f r stride M N eps p block s t hsep hp r.seeds
      (by simp [Domain.readRegion,hd]) i.val
  have hmask := fun hd hm => funext fun i => funext fun j => (hp.1 i).1.2.1 hd hm (lane hN j) j.isLt
  have hmask1 := fun hd hm hx => funext fun i => funext fun j => (hp.1 i).1.2.2 hd hm hx (lane hN j) j.isLt
  apply Result.ext
  all_goals dsimp only [observe,oracle]
  · exact hy
  · cases hf : f.w1 <;> simp [hf] <;> exact hy1 hf
  · cases hf : f.rms <;> simp [hf,hm]
  · exact hs
  · cases hf : f.residualOut <;> simp [hf,hx] <;> exact hr hf
  · cases hf : f.dropout <;> simp [hf,hseed]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> simp [hd,hm] <;> exact hmask hd hm
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;> simp [hd,hm,hx] <;> exact hmask1 hd hm hx

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource

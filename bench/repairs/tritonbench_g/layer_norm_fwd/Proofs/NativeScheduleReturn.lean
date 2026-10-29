import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeScheduleGrid
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

/-- Independent ordered FP target, using only the pre-launch input snapshot.
Owner-zero scalar stores and per-column vector reduction owners are distinct. -/
def oracle {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (s : State α) : TypedSource.Result α M N (HostReturn.seedCount f M) :=
  let x := snapshot r stride s
  { y := fun i j => .mk .real ((NativeParameters.y red plan f x M N eps p i.val j.val).eval a)
    y1 := if f.w1 then some (fun i j => .mk .real ((NativeParameters.y1 red plan1 f x M N eps p i.val j.val).eval a)) else none
    mean := if f.rms then none else some (fun i => .mk .real (a.cast32 ((NativeParameters.mean red f x M N p i.val 0).eval a)))
    rstd := fun i => .mk .real (a.cast32 ((NativeParameters.rstd red f x M N eps p i.val 0).eval a))
    residual := if f.residualOut then (fun i j => .mk .real
      (a.cast32 ((NativeParameters.combined f x M N p i.val j.val).eval a)))
      else (fun i j => s.mem r.x (i.val*stride.x+j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then some (fun i j => .mk .bool (keep a (x.seeds i.val) p j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then
      some (fun i j => .mk .bool (keep a (x.seeds (M+i.val)) p j.val)) else none }

def lane {N block : Nat} (hN : N≤block) (j : Fin N) : Fin block :=
  ⟨j.val,lt_of_lt_of_le j.isLt hN⟩

theorem grid_input_preserved {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (s t : State α) (hsep : Domain.ReadWriteSeparated f r)
    (hp : GridPost a red plan plan1 f r stride M N eps p s t)
    (region : RegionName) (hr : Domain.readRegion f r region) (offset : Nat) :
    t.mem region offset=s.mem region offset := by
  apply hp.2 region offset
  intro i hw
  exact hsep region hr region (TypedSource.written_is_writeRegion hw) rfl

theorem observe_eq_oracle {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (hN : N≤block) (s t : State α) (hsep : Domain.ReadWriteSeparated f r)
    (hp : GridPost a red plan plan1 f r stride M N eps p s t) :
    TypedSource.observe f r stride M N t=oracle a red plan plan1 f r stride M N eps p s := by
  have hy := funext fun i => funext fun j => (hp.1 i).2.2.2.1 (lane hN j) j.isLt
  have hy1 := fun hf => funext fun i => funext fun j => (hp.1 i).2.2.2.2.1 hf (lane hN j) j.isLt
  have hs := funext fun i => (hp.1 i).2.2.2.2.2.1
  have hm := fun hf => funext fun i => (hp.1 i).2.2.2.2.2.2 hf
  have hr := fun hf => funext fun i => funext fun j => (hp.1 i).1 hf (lane hN j) j.isLt
  have hx := funext fun (i : Fin M) => funext fun (j : Fin N) =>
    grid_input_preserved a red plan plan1 f r stride M N eps p s t hsep hp r.x
      (by simp [Domain.readRegion]) (i.val*stride.x+j.val)
  have hseed := fun (hd : f.dropout=Bool.true) => funext fun (i : Fin (HostReturn.seedCount f M)) =>
    grid_input_preserved a red plan plan1 f r stride M N eps p s t hsep hp r.seeds
      (by simp [Domain.readRegion,hd]) i.val
  have hmask := fun hd hm => funext fun i => funext fun j => (hp.1 i).2.1 hd hm (lane hN j) j.isLt
  have hmask1 := fun hd hm hx => funext fun i => funext fun j => (hp.1 i).2.2.1 hd hm hx (lane hN j) j.isLt
  apply TypedSource.Result.ext
  all_goals dsimp only [TypedSource.observe,oracle]
  · exact hy
  · cases hf : f.w1 <;> simp [hf] <;> exact hy1 hf
  · cases hf : f.rms <;> simp [hf,hm]
  · exact hs
  · cases hf : f.residualOut <;> simp [hf,hx] <;> exact hr hf
  · cases hf : f.dropout <;> simp [hf,hseed]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> simp [hd,hm] <;> exact hmask hd hm
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;> simp [hd,hm,hx] <;> exact hmask1 hd hm hx

#axiomsClean grid_input_preserved
#axiomsClean observe_eq_oracle
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule

import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Compilation
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.NativeWords
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Words
open VeriTile.Triton
open Native
noncomputable section
open Classical

abbrev finite := VeriTile.Bench.Repairs.SwigluTriton.Words.finite
abbrev sub := VeriTile.Bench.Repairs.SwigluTriton.Words.sub
abbrev mul := VeriTile.Bench.Repairs.SwigluTriton.Words.mul
abbrev fma := VeriTile.Bench.Repairs.SwigluTriton.Words.fma

abbrev zero := NativeWords.zero
abbrev one := NativeWords.one

/-- Only the six actual native primitive ports. Literal +0 and 1 have fixed
encodings; arbitrary Real literals cannot enter the native word theorem. -/
abbrev Instructions := NativeWords.Instructions

abbrev ports := NativeWords.ports

def software : Instructions where
  neg := fun word => (FP.Value.neg (⟨word⟩ : FP.Value .fp32)).bits
  sub := sub
  mul := mul
  fma := fma
  neu := fun a b => !(FP.eq FP.Config.ieeeValue (⟨a⟩ : FP.Value .fp32) ⟨b⟩)
  gt := fun a b => FP.lt FP.Config.ieeeValue (⟨b⟩ : FP.Value .fp32) ⟨a⟩

def reference (params : Scalars Word) (output : Output) (p g m : Word) : Word :=
  Native.reference (ports software) params output p g m

/-- Optional primitive software-conformance interface for a resolved profile:
FP32 RN-even basic operations with no FTZ; numeric ordered comparisons and
unordered !=. Finite OPERANDS are required at every actual site. No all-NaN
payload equivalence, split/FMA law, or whole-kernel result is assumed. These
primitive meanings are model boundaries, not per-task numerical deferrals;
the public kernel native theorem does not depend on this interface. -/
structure LocalExact (profile : Compilation.Profile) (native : Instructions) : Prop where
  neg : ∀ a,finite a → native.neg a=software.neg a
  sub : ∀ a b,finite a → finite b → native.sub a b=software.sub a b
  mul : ∀ a b,finite a → finite b → native.mul a b=software.mul a b
  fma : ∀ a b c,finite a → finite b → finite c → native.fma a b c=software.fma a b c
  neu : ∀ a b,finite a → finite b → native.neu a b=software.neu a b
  gt : ∀ a b,finite a → finite b → native.gt a b=software.gt a b

def difference (native : Instructions) (g m : Word) := native.sub m g
def update (native : Instructions) (params : Scalars Word) (g m : Word) :=
  native.fma params.beta1 (difference native g m) g
def decay (native : Instructions) (params : Scalars Word) := native.fma (native.neg params.lr) params.wd one
def chosen (native : Instructions) (params : Scalars Word) (g m : Word) :=
  if native.gt (update native params g m) zero then native.sub zero params.lr else params.lr
def gate (native : Instructions) (params : Scalars Word) (g m : Word) :=
  if native.neu (update native params g m) zero then one else zero

structure NumericDomain (native : Instructions) (params : Scalars Word) (p g m : Word) : Prop where
  lr : finite params.lr
  wd : finite params.wd
  beta1 : finite params.beta1
  beta2 : finite params.beta2
  parameter : finite p
  gradient : finite g
  momentum : finite m
  negativeLR : finite (native.neg params.lr)
  difference : finite (difference native g m)
  update : finite (update native params g m)
  decay : finite (decay native params)
  chosen : finite (chosen native params g m)
  step : finite (native.mul (Words.chosen native params g m) (gate native params g m))

theorem reference_exact (profile : Compilation.Profile) (native : Instructions)
    (laws : LocalExact profile native) (params : Scalars Word) (p g m : Word)
    (domain : NumericDomain native params p g m) (output : Output) :
    Native.reference (ports native) params output p g m=reference params output p g m := by
  have hz : finite zero := by unfold finite VeriTile.Bench.Repairs.SwigluTriton.Words.finite zero; decide
  have ho : finite one := by unfold finite VeriTile.Bench.Repairs.SwigluTriton.Words.finite one; decide
  have hn := laws.neg params.lr domain.lr
  have hd := laws.sub m g domain.momentum domain.gradient
  have hu := laws.fma params.beta1 (difference native g m) g domain.beta1 domain.difference domain.gradient
  have hdcy := laws.fma (native.neg params.lr) params.wd one domain.negativeLR domain.wd ho
  have hc := laws.gt (update native params g m) zero domain.update hz
  have hg := laws.neu (update native params g m) zero domain.update hz
  have hnegative := laws.sub zero params.lr hz domain.lr
  have dSame : difference native g m=difference software g m := hd
  have uSame : update native params g m=update software params g m := by
    unfold update
    rw [hu,dSame]
  have cSame : chosen native params g m=chosen software params g m := by
    unfold chosen
    rw [hc,uSame,hnegative]
  have gSame : gate native params g m=gate software params g m := by
    unfold gate
    rw [hg,uSame]
  have gateFinite : finite (gate native params g m) := by
    unfold gate
    split <;> assumption
  have hs := laws.mul (chosen native params g m) (gate native params g m) domain.chosen gateFinite
  have decaySame : decay native params=decay software params := by
    unfold decay
    rw [hdcy,hn]
  have stepSame : native.mul (chosen native params g m) (gate native params g m)=
      software.mul (chosen software params g m) (gate software params g m) := by rw [hs,cSame,gSame]
  cases output
  · simp only [reference,Native.reference,ports,NativeWords.ports,if_true,zero_ne_one,if_false]
    change native.fma (decay native params) p
      (native.mul (chosen native params g m) (gate native params g m))=_
    rw [laws.fma _ _ _ domain.decay domain.parameter domain.step,decaySame,stepSame]
    rfl
  · simp only [reference,Native.reference,ports,NativeWords.ports,if_true,zero_ne_one,if_false]
    change native.fma params.beta2 (difference native g m) g=_
    rw [laws.fma _ _ _ domain.beta2 domain.difference domain.gradient,dSame]
    rfl

#specNonCircular reference avoiding [Native.interpret,Native.sourceScalar,Tile.scalar,Tile.sourceRun,Machine.scalar,Machine.run]
#axiomsClean reference_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Words

import bench.repairs.tritonbench_g.rbe_triton_transform.Correct
import bench.repairs.tritonbench_g.rbe_triton_transform.FPEquiv

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Rotary
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
open Numerics
noncomputable section
set_option maxHeartbeats 2000000

/-- Independent logical pair rotation. The restored sibling helper fixes DIM128;
it does not use the unrelated main-wrapper head_dim parameter. -/
def expected (r : Request) (x : Nat → ℝ) (b m c : Nat) : ℝ :=
  let n := c / 2 * 2
  let angle := ((r.position : ℝ) + m) / Real.rpow 10000 ((n % 128 : Nat) / (128 : ℝ))
  let u := x ((b*r.rows+m)*r.columns+n)
  let v := if n+1 < r.columns then x ((b*r.rows+m)*r.columns+(n+1)) else 0
  if c % 2 = 1 then u*Real.sin angle+v*Real.cos angle
  else u*Real.cos angle-v*Real.sin angle

theorem independent_target (r : Request) (x : Nat → ℝ) (b m c : Nat) :
    Spec.result r x b m c = expected r x b m c := by
  simp [Spec.result,Spec.pair,Spec.angle,expected,Request.offset]

/-- Twelve scalar primitive family equalities. Selected offline PTX uses
`div.full.f32` at both source division sites. No transcendental exact-to-Real,
whole pair, row, reduction, output, source-run or memory equality is assumed. -/
structure Correspondence (a r : Primitives) : Prop where
  literal : ∀ x, a.literal x = r.literal x
  natural : ∀ x, a.natural x = r.natural x
  integer : ∀ x, a.integer x = r.integer x
  add : ∀ x y, a.add x y = r.add x y
  sub : ∀ x y, a.sub x y = r.sub x y
  mul : ∀ x y, a.mul x y = r.mul x y
  div : ∀ x y, a.div x y = r.div x y
  pow : ∀ x y, a.pow x y = r.pow x y
  neg : ∀ x, a.neg x = r.neg x
  sin : ∀ x, a.sin x = r.sin x
  cos : ∀ x, a.cos x = r.cos x
  fma : ∀ x y z, a.fma x y z = r.fma x y z

theorem correspondence_eq (a r : Primitives) (h : Correspondence a r) : a=r := by
  cases a; cases r
  simp only [Primitives.mk.injEq]
  exact ⟨funext (fun x => h.literal x),
    funext (fun x => h.natural x),
    funext (fun x => h.integer x),
    funext (fun x => funext (fun y => h.add x y)),
    funext (fun x => funext (fun y => h.sub x y)),
    funext (fun x => funext (fun y => h.mul x y)),
    funext (fun x => funext (fun y => h.div x y)),
    funext (fun x => funext (fun y => h.pow x y)),
    funext (fun x => h.neg x),
    funext (fun x => h.sin x),
    funext (fun x => h.cos x),
    funext (fun x => funext (fun y => funext (fun z => h.fma x y z)))⟩

theorem reference_consistent (r : Primitives) : Correspondence r r :=
  ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ _ => rfl, fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- The existing source-specific rotary execution/grid/memory proof is reused
through its unconditional native result. Its old exact-to-Real premises are not used. -/
theorem whole_native (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (input output : RegionName) (s : State Tagged)
    (call : Native.Preparation api leaf r input s)
    (abi : Native.IndexABI api leaf r input s call)
    (codec : ∀ w k, api.encode w k=wordByte w k) :
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor=call.output ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns → ∀ k : Fin 4,
      api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.logicalAddress call.output b m c+k.val) =
          wordByte (result p r (fun o => ((s.mem input o).read .fp32).word) b m c) k) ∧
    (∀ b, call.caller b → api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory b=api.byte call.before b) := by
  obtain ⟨hd,hs,hv,hb,hf⟩ := FPEquiv.whole_bits api p r input output s call abi codec
  exact ⟨hd,by simpa only [Native.entry] using hb,hf⟩

theorem whole_reference (api : Memory.API Word Store (BitVec 8)) (a ref : Primitives)
    (r : Request) (input output : RegionName) (s : State Tagged)
    (call : Native.Preparation api leaf r input s)
    (abi : Native.IndexABI api leaf r input s call)
    (codec : ∀ w k, api.encode w k=wordByte w k)
    (numeric : Correspondence a ref) :
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns → ∀ k : Fin 4,
      api.byte (Native.entry api leaf Tagged.word (operations a) r input output s call).memory
        (Native.logicalAddress call.output b m c+k.val) =
          wordByte (result ref r (fun o => ((s.mem input o).read .fp32).word) b m c) k) ∧
    (∀ b, call.caller b → api.byte (Native.entry api leaf Tagged.word (operations a) r input output s call).memory b=api.byte call.before b) := by
  rw [correspondence_eq a ref numeric]
  exact (whole_native api ref r input output s call abi codec).2

theorem whole_real (api : Memory.API ℝ Store Byte) (r : Request) (input output : RegionName)
    (s : State ℝ) (call : Native.Preparation api id r input s)
    (abi : Native.IndexABI api id r input s call) :
    (Native.entry api id id ideal r input output s call).tensor=call.output ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns →
      Native.logicalRead api (Native.entry api id id ideal r input output s call).memory
        call.output b m c = expected r (fun o => (s.mem input o).read .fp32) b m c) ∧
    (∀ b, call.caller b → api.byte (Native.entry api id id ideal r input output s call).memory b=api.byte call.before b) := by
  obtain ⟨hd,hs,hv,hf,he⟩ := Correct.whole_real api r input output s call abi
  refine ⟨hd,?_,hf⟩
  intro b m c hb hm hc
  simpa only [Native.entry,independent_target] using hv b m c hb hm hc

#axiomsClean independent_target
#axiomsClean correspondence_eq
#axiomsClean reference_consistent
#axiomsClean whole_native
#axiomsClean whole_reference
#axiomsClean whole_real
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.Rotary

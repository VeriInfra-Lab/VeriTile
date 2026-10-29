import bench.repairs.tritonbench_g.rope_transform.Proofs.Host
namespace VeriTile.Bench.Repairs.RopeTransform.Math
open VeriTile Triton FP.Structural
noncomputable section

/-- Mathematical interpretation of the typed source: storage/compute casts are
identity and the three reachable binary operators use real arithmetic. This
is explicitly a mathematical projection, not FP32 conformance. Unreachable
operations have fixed interpretations and no callback can supply an output. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y
    | .div => x/y | .max => max x y | .pow => 0
  unary := fun _ op x => match op with | .neg _ => -x | .abs _ => abs x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

/-- Conventional forward RoPE, with supplied coefficients. -/
def rotation (x y c sn : ℝ) (second : Bool) :=
  if second then y*c+x*sn else x*c-y*sn
theorem scalar_rotation (x y c sn : ℝ) (second : Bool) :
    scalar algebra x y c sn second=rotation x y c sn second := by cases second <;> rfl

def expected (r : Request) (before : State ℝ) (input : Host.Tensor)
    (key : Bool) (b h s c : Nat) : ℝ :=
  let j := if c<r.half then c else c-r.half
  rotation ((before.mem input.region (r.external key b h s j)).read .fp32)
    ((before.mem input.region (r.external key b h s (j+r.half))).read .fp32)
    ((before.mem "C" (s*r.half+j)).read .fp32)
    ((before.mem "S" (s*r.half+j)).read .fp32) (decide (r.half≤c))

theorem whole_real (r : Request) (call : Host.Preparation ℝ r) :
    ∃ result,Host.invoke algebra r call=some result ∧
      (∀ key,result.tensor key=call.output key) ∧ result.cosine=call.cosine ∧ result.sine=call.sine ∧
      (∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
        (result.state.mem (result.tensor key).region ((result.tensor key).offset b h s c)).read .fp32=
          expected r call.before (call.input key) key b h s c) ∧
      (∀ name o,Host.Outside r name o → result.state.mem name o=call.before.mem name o) := by
  obtain ⟨result,he,ht,hc,hs,hmeta,hv,hf,hcoeff⟩ := Host.whole_typed algebra r call
  refine ⟨result,he,ht,hc,hs,?_,hf⟩
  intro key b h s c hb hh hs hc
  rw [hv key b h s c hb hh hs hc]
  change scalar algebra _ _ _ _ _=rotation _ _ _ _ _
  exact scalar_rotation .. 

/-- Reuse the audited exact source-shape bridge. This statement is only about
address/control erasure and makes no native numerical-equivalence claim. -/
theorem erased_source_shape (r : Request) :
    (kernel r).eraseDType.surfaceBody=
      (VeriTile.Bench.TritonBenchG.RopeTransform.triton_rope_surface
        "Q" "K" "C" "S" (r.stride Bool.false) (r.stride Bool.true) r.half r.half r.sequence 0
        r.qHeads r.kHeads r.dimension (r.paddedHeads Bool.false) (r.paddedHeads Bool.true)
        r.paddedDimension 0 Bool.false).eraseDType.surfaceBody :=
  VeriTile.Bench.TritonBenchG.RopeTransform.Typed.source_shape r.parameters

#axiomsClean scalar_rotation
#axiomsClean whole_real
#axiomsClean erased_source_shape
#specNonCircular expected avoiding [Host.invoke,Grid.run,Grid.sourceRun,FP.Structural.exec,value,T.laneResult,Host.expected]
end
end VeriTile.Bench.Repairs.RopeTransform.Math

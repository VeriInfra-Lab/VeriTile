import bench.tritonbench_g.rotary_transform.Proofs.Physical
namespace VeriTile.Bench.Repairs.RotaryTransform.Physical
open Native

/-- Caller coefficient views and integer-offset view are materialized by the
original wrapper's contiguous calls. Caller allocations are other 1/2/3;
normalized regions are private temporary interpretations. -/
structure InputViews where
  cosineRow : Nat
  cosineColumn : Nat
  sineRow : Nat
  sineColumn : Nat
  offsetsStride : Nat
  deriving DecidableEq, Repr

def InputViews.cosineAt (v : InputViews) (s : Shape) (i : Nat) :=
  (i/s.half)*v.cosineRow+(i%s.half)*v.cosineColumn
def InputViews.sineAt (v : InputViews) (s : Shape) (i : Nat) :=
  (i/s.half)*v.sineRow+(i%s.half)*v.sineColumn

def stageInputs (s : Shape) (v : InputViews) (offsets64 : Bool) (m : Bytes) : Bytes := fun r b =>
  if r=.cosine then m (.other 2) (4*v.cosineAt s (b/4)+b%4)
  else if r=.sine then m (.other 3) (4*v.sineAt s (b/4)+b%4)
  else if r=.offsets then
    let n := if offsets64 then 8 else 4
    m (.other 1) ((b/n)*v.offsetsStride*n+b%n)
  else m r b

def CallerRegion (r : Region) : Prop :=
  r≠.cosine ∧ r≠.sine ∧ r≠.cumulative ∧ r≠.offsets

theorem stage_frame (s : Shape) (v : InputViews) (wide : Bool) (m : Bytes)
    (r : Region) (b : Nat) (h : CallerRegion r) : stageInputs s v wide m r b=m r b := by
  simp [stageInputs,CallerRegion] at *
  simp_all

theorem stage_input (s : Shape) (v : InputViews) (wide : Bool) (m : Bytes) :
    load (stageInputs s v wide m) .input=load m .input := by
  funext a
  simp [load,stageInputs]

theorem stage_cosine (s : Shape) (v : InputViews) (wide : Bool) (m : Bytes) (a : Nat) :
    load (stageInputs s v wide m) .cosine a=load m (.other 2) (v.cosineAt s a) := by
  unfold load
  congr 1
  funext i
  have hd : (4*a+i.val)/4=a := by omega
  have hm : (4*a+i.val)%4=i.val := by omega
  simp [stageInputs,hd,hm]

theorem stage_sine (s : Shape) (v : InputViews) (wide : Bool) (m : Bytes) (a : Nat) :
    load (stageInputs s v wide m) .sine a=load m (.other 3) (v.sineAt s a) := by
  unfold load
  congr 1
  funext i
  have hd : (4*a+i.val)/4=a := by omega
  have hm : (4*a+i.val)%4=i.val := by omega
  simp [stageInputs,hd,hm]

theorem stage_offset32 (s : Shape) (v : InputViews) (m : Bytes) (i : Nat) :
    load (stageInputs s v false m) .offsets i=load m (.other 1) (i*v.offsetsStride) := by
  unfold load
  congr 1
  funext j
  have hd : (4*i+j.val)/4=i := by omega
  have hm : (4*i+j.val)%4=j.val := by omega
  simp only [stageInputs,Bool.false_eq_true,↓reduceIte,hd,hm,reduceCtorEq]
  congr 1
  ring

theorem stage_offset64 (s : Shape) (v : InputViews) (m : Bytes) (i : Nat) (o : Bool) :
    load (stageInputs s v true m) .offsets (2*i+if o then 1 else 0)=
      load m (.other 1) (2*(i*v.offsetsStride)+if o then 1 else 0) := by
  unfold load
  congr 1
  funext j
  have hd : (4*(2*i+if o then 1 else 0)+j.val)/8=i := by cases o <;> simp <;> omega
  have hm : (4*(2*i+if o then 1 else 0)+j.val)%8=4*(if o then 1 else 0)+j.val := by
    cases o <;> simp <;> omega
  simp only [stageInputs,↓reduceIte,hd,hm,reduceCtorEq]
  congr 1
  ring

#axiomsClean stage_frame
#axiomsClean stage_input
#axiomsClean stage_cosine
#axiomsClean stage_sine
#axiomsClean stage_offset32
#axiomsClean stage_offset64
end VeriTile.Bench.Repairs.RotaryTransform.Physical

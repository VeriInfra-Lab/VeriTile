import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceModel

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Graph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- Independent, input-only, ordered three-pass graph. No source state,
register, memory, output or execution appears in these definitions. -/
def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def loadX (M : Algebra α) (N : Nat) (x : Nat → α) (j : Nat) :=
  M.cast (some .fp32) .fp32 .real (if j<N then x j else
    M.cast (some .fp32) .real .fp32 (zero M))
def add (M : Algebra α) := M.binary (some .fp32) .real .add
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def divide (M : Algebra α) (N : Nat) (x : α) :=
  M.binary (some .fp32) .real .div x (M.fromInt (some .fp32) (N : Int))
def sum (M : Algebra α) (a : Fin 128 → α) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
    (fun i : TileIndex [128] => a i.1) PUnit.unit

def mean (M : Algebra α) (N : Nat) (x : Nat → α) : α :=
  divide M N (sum M (foldRange
    (fun off a lane => add M (a lane) (loadX M N x (off+lane.val)))
    0 N 128 (fun _ => zero M)))
def centered (M : Algebra α) (N : Nat) (x : Nat → α) (mu : α) (j : Nat) : α :=
  if j<N then sub M (loadX M N x j) mu else zero M

def variance (M : Algebra α) (N : Nat) (x : Nat → α) (mu : α) : α :=
  divide M N (sum M (foldRange
    (fun off a lane => let z := centered M N x mu (off+lane.val)
      add M (a lane) (mul M z z)) 0 N 128 (fun _ => zero M)))
def inverse (M : Algebra α) (eps v : α) : α :=
  M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
    (M.unary (some .fp32) .sqrt (add M v eps))
def output (M : Algebra α) (N : Nat) (x w : Nat → α) (eps : α) (j : Nat) : α :=
  let mu := mean M N x
  let inv := inverse M eps (variance M N x mu)
  M.cast none .real .fp32 (mul M (mul M (sub M (loadX M N x j) mu) inv)
    (M.cast (some .fp32) .fp32 .real (w j)))

/-- Projection commutes with the ordered streaming fold. -/
theorem fold_lanes {α : Type} (step : Nat → (Fin 128 → α) → Fin 128 → α)
    (start stop : Nat) (a : Fin 128 → α) :
    foldRange (fun off values (i : TileIndex [128]) =>
      step off (fun j => values (j,PUnit.unit)) i.1)
      start stop 128 (fun i : TileIndex [128] => a i.1) =
    (fun i : TileIndex [128] => foldRange step start stop 128 a i.1) := by
  conv_lhs => rw [foldRange]
  conv_rhs => enter [i]; rw [foldRange]
  by_cases h : start<stop
  · simp only [show (128:Nat)≠0 by decide,h,dif_neg,dif_pos,↓reduceDIte]
    exact fold_lanes step (start+128) stop (step start a)
  · simp only [show (128:Nat)≠0 by decide,h,dif_neg,↓reduceDIte]
termination_by stop-start
decreasing_by omega

variable {α : Type} [Inhabited α]

theorem source_mean (M : Algebra α) (p : Source.Parameters) (s : State α)
    (hb : p.block=128) (x : Nat → α)
    (hx : ∀ j, j<p.columns → (s.mem p.x (Source.xBase p s+j*p.xDimStride)).read .fp32=x j) :
    Source.meanValue M p s=mean M p.columns x := by
  rcases p with ⟨rx,rw,ry,xs,xh,xd,ys,yh,yd,wh,wd,N,B,validBlock⟩
  dsimp only at hb
  subst B
  let p : Source.Parameters := ⟨rx,rw,ry,xs,xh,xd,ys,yh,yd,wh,wd,N,128,⟨by decide, by decide, 7, by decide⟩⟩
  have hv : Source.xValue M p s=loadX M p.columns x := by
    funext j
    by_cases h : j<p.columns
    · simpa only [Source.xValue,loadX,if_pos h] using
        congrArg (M.cast (some .fp32) .fp32 .real) (hx j h)
    · norm_num [Source.xValue,loadX,zero,h]
  have hf := fold_lanes (fun off a lane => add M (a lane) (loadX M p.columns x (off+lane.val))) 0 p.columns (fun _ => zero M)
  have hs : Source.meanStep M p s=(fun off values (i : TileIndex [128]) =>
      add M (values (i.1,PUnit.unit)) (loadX M p.columns x (off+i.1.val))) := by
    funext off values i
    rcases i with ⟨lane,⟨⟩⟩
    simp only [Source.meanStep,hv,add]
  change Source.meanValue M p s=mean M p.columns x
  unfold Source.meanValue mean divide sum
  rw [hs]
  exact congrArg (fun (values : TileIndex [128] → α) => M.binary (some .fp32) .real .div
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false values PUnit.unit)
      (M.fromInt (some .fp32) (p.columns : Int))) hf

theorem source_variance (M : Algebra α) (p : Source.Parameters) (s : State α)
    (hb : p.block=128) (x : Nat → α)
    (hx : ∀ j, j<p.columns → (s.mem p.x (Source.xBase p s+j*p.xDimStride)).read .fp32=x j) :
    Source.varianceValue M p s=variance M p.columns x (mean M p.columns x) := by
  rcases p with ⟨rx,rw,ry,xs,xh,xd,ys,yh,yd,wh,wd,N,B,validBlock⟩
  dsimp only at hb
  subst B
  let p : Source.Parameters := ⟨rx,rw,ry,xs,xh,xd,ys,yh,yd,wh,wd,N,128,⟨by decide, by decide, 7, by decide⟩⟩
  have hm := source_mean M p s rfl x hx
  have hv : Source.xValue M p s=loadX M p.columns x := by
    funext j
    by_cases h : j<p.columns
    · simpa only [Source.xValue,loadX,if_pos h] using
        congrArg (M.cast (some .fp32) .fp32 .real) (hx j h)
    · norm_num [Source.xValue,loadX,zero,h]
  have hc : Source.centered M p s=centered M p.columns x (mean M p.columns x) := by
    funext j
    norm_num only [Source.centered,centered,hv,hm,sub,zero]
  have hf := fold_lanes (fun off a lane => let z := centered M p.columns x (mean M p.columns x) (off+lane.val)
    add M (a lane) (mul M z z)) 0 p.columns (fun _ => zero M)
  have hs : Source.varStep M p s=(fun off values (i : TileIndex [128]) =>
      let z := centered M p.columns x (mean M p.columns x) (off+i.1.val)
      add M (values (i.1,PUnit.unit)) (mul M z z)) := by
    funext off values i
    rcases i with ⟨lane,⟨⟩⟩
    simp only [Source.varStep,hc,add,mul]
  change Source.varianceValue M p s=variance M p.columns x (mean M p.columns x)
  unfold Source.varianceValue variance divide sum
  rw [hs]
  exact congrArg (fun (values : TileIndex [128] → α) => M.binary (some .fp32) .real .div
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false values PUnit.unit)
      (M.fromInt (some .fp32) (p.columns : Int))) hf

theorem source_output (M : Algebra α) (p : Source.Parameters) (s : State α)
    (hb : p.block=128) (x w : Nat → α)
    (hx : ∀ j, j<p.columns → (s.mem p.x (Source.xBase p s+j*p.xDimStride)).read .fp32=x j)
    (hw : ∀ j, j<p.columns → (s.mem p.weight (Source.wBase p s+j*p.weightDimStride)).read .fp32=w j)
    (j : Nat) (hj : j<p.columns) :
    Source.yValue M p s j=output M p.columns x w (Source.epsilonValue s) j := by
  have hm := source_mean M p s hb x hx
  have hv := source_variance M p s hb x hx
  simp only [Source.yValue,Source.rstdValue,hm,hv,Source.xValue,Source.wValue,
    if_pos hj,hx j hj,hw j hj,output,inverse,loadX,if_pos hj,add,sub,mul]

/-- Only live logical inputs can influence an output. No obligation is made
about masked raw storage or weight loads at inactive lanes. -/
theorem output_congr (M : Algebra α) (N : Nat) (x x' w w' : Nat → α) (eps : α)
    (hx : ∀ j, j<N → x j=x' j) (hw : ∀ j, j<N → w j=w' j)
    (j : Nat) (hj : j<N) : output M N x w eps j=output M N x' w' eps j := by
  have hl : loadX M N x=loadX M N x' := by
    funext k
    by_cases h : k<N <;> simp [loadX,h,hx]
  simp only [output,mean,variance,centered,hl,hw j hj]

#specNonCircular output avoiding [FP.Structural.exec, Source.yValue, Source.referenceMem]
#axiomsClean source_output
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Graph

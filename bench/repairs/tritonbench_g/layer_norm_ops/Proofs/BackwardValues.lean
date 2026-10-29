import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.FiniteReference
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

structure BackwardBinding (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) : Prop where
  x : ∀ row,rowActive p pid row → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.x (row*p.xStride+i.1.val)).read .fp32=backwardX v row i
  weight : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.weight i.1.val).read .fp32=backwardWeight v i
  bias : p.recompute=Bool.true → p.hasBias=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.bias i.1.val).read .fp32=backwardBias v i
  dy : ∀ row,rowActive p pid row → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.dy (row*p.dyStride+i.1.val)).read .fp32=backwardDY v row i
  dres : p.hasDres=Bool.true → ∀ row,rowActive p pid row → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.dres (row*p.dresStride+i.1.val)).read .fp32=backwardDres v row i
  mean : p.rms=Bool.false → ∀ row,rowActive p pid row →
    (s.mem p.mean (row*p.meanStride)).read .fp32=backwardMean v row
  rstd : ∀ row,rowActive p pid row →
    (s.mem p.rstd (row*p.rstdStride)).read .fp32=backwardInverse v row

theorem row_lt (p : BackwardParameters) (pid row : Nat) (h : rowActive p pid row) : row<p.rows :=
  Nat.lt_of_lt_of_le h.2 (Nat.min_le_right _ _)

theorem backward_binding (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (h : (backwardIO A p pid).InputBinding v s) :
    BackwardBinding p pid v s := by
  constructor
  · intro row live i hi
    have hr := row_lt p pid row live
    simpa only [backwardX,extend,dif_pos hr] using
      h ⟨0,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,i) ⟨live,hi⟩
  · intro i hi
    exact h ⟨1,by simp [backwardIO,backwardInputs]⟩ i hi
  · intro hy hb i hi
    exact h ⟨2,by simp [backwardIO,backwardInputs]⟩ i ⟨hy,hb,hi⟩
  · intro row live i hi
    have hr := row_lt p pid row live
    simpa only [backwardDY,extend,dif_pos hr] using
      h ⟨3,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,i) ⟨live,hi⟩
  · intro hd row live i hi
    have hr := row_lt p pid row live
    simpa only [backwardDres,extend,dif_pos hr] using
      h ⟨4,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,i) ⟨hd,live,hi⟩
  · intro hm row live
    have hr := row_lt p pid row live
    simpa only [backwardMean,extend,dif_pos hr] using
      h ⟨5,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,()) ⟨hm,live⟩
  · intro row live
    have hr := row_lt p pid row live
    simpa only [backwardInverse,extend,dif_pos hr] using
      h ⟨6,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,()) live

theorem backward_load_eq (A : Algebra α) (p : BackwardParameters) (s : State α)
    (region : RegionName) (base : Nat) (values : TileIndex [p.block] → α)
    (bound : ∀ i : TileIndex [p.block],i.1.val<p.columns →
      (s.mem region (base+i.1.val)).read .fp32=values i) :
    Backward.loaded A p s region base=backwardLoad A p values := by
  funext i
  unfold Backward.loaded backwardLoad
  by_cases hi : i.1.val<p.columns <;> simp only [hi,if_true,if_false,bound i]

theorem backward_weight_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s) :
    (Backward.initial A p s).weight=backwardLoad A p (backwardWeight v) := by
  apply backward_load_eq
  intro i hi
  simpa only [Nat.zero_add] using bound.weight i hi

theorem backward_dy_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    Backward.dy A p (Backward.initial A p s) row=backwardLoad A p (backwardDY v row) :=
  backward_load_eq A p s p.dy (row*p.dyStride) _ (bound.dy row live)

theorem backward_hat_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    Backward.xhat A p (Backward.initial A p s) row=backwardHat A p v row := by
  have hx := backward_load_eq A p s p.x (row*p.xStride) _ (bound.x row live)
  funext i
  change (if i.1.val<p.columns then
    Forward.binary A .mul
      (if p.rms then Backward.loaded A p s p.x (row*p.xStride) i else
        Forward.binary A .sub (Backward.loaded A p s p.x (row*p.xStride) i)
          (Forward.cast A ((s.mem p.mean (row*p.meanStride)).read .fp32)))
      (Forward.cast A ((s.mem p.rstd (row*p.rstdStride)).read .fp32))
    else Forward.literal A 0)=_
  rw [hx,bound.rstd row live]
  cases hm : p.rms
  · simp only [backwardHat,hm,Bool.false_eq_true,if_false,bound.mean hm row live]
  · simp only [backwardHat,hm,if_true]

theorem backward_wdy_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    Backward.wdy A p (Backward.initial A p s) row=backwardWdy A p v row := by
  funext i
  simp only [Backward.wdy,backwardWdy,backward_weight_eq A p pid v s bound,
    backward_dy_eq A p pid v s bound row live]

theorem backward_dx_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    KernelReference.rowDX A p s row=backwardDX A p v row := by
  funext i
  unfold KernelReference.rowDX backwardDX Backward.dxBase Backward.c1 Backward.c2
  simp only [backward_hat_eq A p pid v s bound row live,backward_wdy_eq A p pid v s bound row live]
  have hr : Backward.rstdValue A p (Backward.initial A p s) row=
      Forward.cast A (backwardInverse v row) := congrArg (Forward.cast A) (bound.rstd row live)
  rw [hr]
  by_cases hd : p.hasDres=Bool.true
  · simp only [hd,if_true,backward_load_eq A p s p.dres (row*p.dresStride) _ (bound.dres hd row live)]
  · simp only [hd,Bool.false_eq_true,if_false]

theorem backward_y_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) (hy : p.recompute=Bool.true) :
    KernelReference.rowY A p s row=backwardY A p v row := by
  funext i
  unfold KernelReference.rowY Backward.yValue backwardY
  simp only [backward_hat_eq A p pid v s bound row live,backward_weight_eq A p pid v s bound]
  by_cases hb : p.hasBias=Bool.true
  · have hbias : (Backward.initial A p s).bias=backwardLoad A p (backwardBias v) := by
      apply backward_load_eq
      intro j hj
      simpa only [Nat.zero_add] using bound.bias hy hb j hj
    simp only [hb,if_true,hbias]
  · simp only [hb,Bool.false_eq_true,if_false]

theorem backward_dw_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    KernelReference.rowDW A p s row=backwardRowDW A p v row := by
  funext i
  simp only [KernelReference.rowDW,backwardRowDW,backward_dy_eq A p pid v s bound row live,
    backward_hat_eq A p pid v s bound row live]

theorem backward_db_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s)
    (row : Nat) (live : rowActive p pid row) :
    KernelReference.rowDB A p s row=backwardRowDB A p v row :=
  backward_dy_eq A p pid v s bound row live

theorem fold_congr {β : Type} (first second : Nat → β → β) (cur stop : Nat)
    (equal : ∀ row,cur≤row → row<stop → first row=second row) (initial : β) :
    foldRange first cur stop 1 initial=foldRange second cur stop 1 initial := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  by_cases active : cur<stop
  · simp only [Nat.one_ne_zero,active,dite_false,dite_true]
    rw [equal cur (by rfl) active]
    exact fold_congr first second (cur+1) stop
      (fun row low high => equal row (by omega) high) _
  · simp only [Nat.one_ne_zero,active,dite_false]
termination_by stop-cur
decreasing_by omega

theorem backward_partials_eq (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (s : State α) (bound : BackwardBinding p pid v s) (hp : s.pids 0=pid) :
    KernelReference.partials A p s=backwardPartials A p pid v := by
  unfold KernelReference.partials backwardPartials
  simp only [Backward.start,Backward.stop,hp]
  apply fold_congr
  intro row lower upper
  funext carry
  simp only [KernelReference.addRow,backwardAddRow,
    backward_dw_eq A p pid v s bound row ⟨lower,upper⟩,
    backward_db_eq A p pid v s bound row ⟨lower,upper⟩]

#axiomsClean backward_binding
#axiomsClean backward_dx_eq
#axiomsClean backward_partials_eq
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface

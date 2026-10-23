import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy2.Kernels
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.WholeReal
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AllFrames
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.RoleBindings
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ReturnBacking

namespace VeriTile.Bench.Repairs.CrossEntropy2.Correct
open VeriTile.Triton Source Host RealHost
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations Preserves FloatArray IntegerViews value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Public metadata domain: FP32 logits/upstream/LSE, int32 labels, positive
class width, explicit supported block size, smoothing and class interval.
The Real target normalizes all supplied columns, including proper class
partitions. No remote logits are inferred. Empty row batches are admitted. -/
abbrev Domain := Host.valid

/-- Independent mathematical row quantities. Ignored losses and gradients
are zero; LSE remains computed. Arbitrary int32 labels are in scope. -/
abbrev loss := Math.loss
abbrev logsumexp := Math.LSE
abbrev zLoss := Math.zLoss
abbrev gradient := Math.gradient
abbrev gradientWithSuppliedLSE := Math.gradientWithLSE

theorem forward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (v : Nat → Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (success : forward RealSource.algebra rt o x y initial=.ok returned final) :
    FloatArray obs final returned.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final returned.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final returned.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    returned.1.region≠returned.2.1.region ∧ returned.1.region≠returned.2.2.region ∧
    returned.2.1.region≠returned.2.2.region ∧ Preserves obs initial final :=
  RealHost.forward_correct obs rt lib ints o x y rows cols initial final returned shape xLive yLive v caller success

theorem backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (v : Nat → Nat → ℝ) (upstream L : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs initial g row=↑(upstream row))
    (lValues : ∀ row, row<rows → value obs initial l row=↑(L row))
    (success : backward RealSource.algebra rt o g x l y initial=.ok returned final) :
    FloatArray obs final returned [rows,cols]
      (fun i => ↑(Math.gradientWithLSE o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (L (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬initial.live returned.region ∧ Preserves obs initial final :=
  RealHost.backward_correct obs rt lib ints o g x l y rows cols initial final returned shape xLive yLive gLive lLive v upstream L caller gValues lValues success

theorem forward_backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World (WithBot ℝ) σ)
    (f : Return) (gradient : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region) (v : Nat → Nat → ℝ) (upstream : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs saved g row=↑(upstream row))
    (fSuccess : forward RealSource.algebra rt o x y initial=.ok f saved)
    (bSuccess : backward RealSource.algebra rt o g x f.2.1 y saved=.ok gradient final) :
    FloatArray obs final f.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final f.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final f.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final gradient [rows,cols]
      (fun i => ↑(Math.gradient cols o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬saved.live gradient.region ∧ Preserves obs initial final :=
  RealHost.forward_backward_correct obs rt lib ints o g x y rows cols initial saved final f gradient shape xLive yLive gLive v upstream caller gValues fSuccess bSuccess

theorem checked_forward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (mx my : Checked.TensorMetadata) (ms : Checked.ScalarMetadata) (splitBool : Bool)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Return)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (v : Nat → Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (success : Checked.forward RealSource.algebra rt mx my ms splitBool o x y rows cols (obs.integer initial y) initial=.ok returned final) :
    FloatArray obs final returned.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final returned.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final returned.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    returned.1.region≠returned.2.1.region ∧ returned.1.region≠returned.2.2.region ∧
    returned.2.1.region≠returned.2.2.region ∧ Preserves obs initial final :=
  RealHost.forward_correct obs rt lib ints o x y rows cols initial final returned shape xLive yLive v caller (Checked.forward_success.mp success).2

#axiomsClean checked_forward_correct

theorem checked_backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (mx my mg ml : Checked.TensorMetadata) (ms : Checked.ScalarMetadata)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial final : World (WithBot ℝ) σ) (returned : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : initial.live g.region) (lLive : initial.live l.region)
    (v : Nat → Nat → ℝ) (upstream L : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs initial g row=↑(upstream row))
    (lValues : ∀ row, row<rows → value obs initial l row=↑(L row))
    (success : Checked.backward RealSource.algebra rt mx my mg ml ms o g x l y rows cols (obs.integer initial y) initial=.ok returned final) :
    FloatArray obs final returned [rows,cols]
      (fun i => ↑(Math.gradientWithLSE o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (L (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬initial.live returned.region ∧ Preserves obs initial final :=
  RealHost.backward_correct obs rt lib ints o g x l y rows cols initial final returned shape xLive yLive gLive lLive v upstream L caller gValues lValues (Checked.backward_success.mp success).2

#axiomsClean checked_backward_correct

theorem checked_forward_backward_correct {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : Library obs arithmetic rt) (ints : IntegerViews obs)
    (mx my mg ml : Checked.TensorMetadata) (ms : Checked.ScalarMetadata) (splitBool : Bool)
    (o : Options) (g x y : Tensor) (rows cols : Nat) (initial saved final : World (WithBot ℝ) σ)
    (f : Return) (gradient : Tensor)
    (shape : x.shape=[rows,cols]) (xLive : initial.live x.region) (yLive : initial.live y.region)
    (gLive : saved.live g.region) (v : Nat → Nat → ℝ) (upstream : Nat → ℝ)
    (caller : ∀ row, row<rows → ∀ col, col<cols → value obs initial x (row*cols+col)=↑(v row col))
    (gValues : ∀ row, row<rows → value obs saved g row=↑(upstream row))
    (fSuccess : Checked.forward RealSource.algebra rt mx my ms splitBool o x y rows cols (obs.integer initial y) initial=.ok f saved)
    (bSuccess : Checked.backward RealSource.algebra rt mx my mg ml ms o g x f.2.1 y rows cols (obs.integer saved y) saved=.ok gradient final) :
    FloatArray obs final f.1 [rows]
      (fun row => ↑(Math.loss cols o.total o.start (obs.integer initial y row) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final f.2.1 [rows] (fun row => ↑(Math.LSE cols o.logitScale (v row))) ∧
    FloatArray obs final f.2.2 [rows]
      (fun row => ↑(Math.zLoss cols (obs.integer initial y row) o.ignored o.logitScale o.squareScale (v row))) ∧
    FloatArray obs final gradient [rows,cols]
      (fun i => ↑(Math.gradient cols o.total o.start (obs.integer initial y (i/cols)) o.ignored
        (Math.smoothing o.smoothing o.hasSmoothing) o.logitScale o.squareScale (upstream (i/cols)) (v (i/cols)) (i%cols))) ∧
    ¬saved.live gradient.region ∧ Preserves obs initial final :=
  RealHost.forward_backward_correct obs rt lib ints o g x y rows cols initial saved final f gradient shape xLive yLive gLive v upstream caller gValues (Checked.forward_success.mp fSuccess).2 (Checked.backward_success.mp bSuccess).2

#axiomsClean checked_forward_backward_correct

/- These public aliases include ordinary local failures, invalid metadata,
and the complete externally retained caller backing allocations. -/
export Frames (forward_frame backward_frame Frame.all_bytes)
export Physical (forward_roles forward_target_role backward_roles forward_return_backing backward_return_backing)

#axiomsClean forward_correct
#axiomsClean backward_correct
#axiomsClean forward_backward_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Correct

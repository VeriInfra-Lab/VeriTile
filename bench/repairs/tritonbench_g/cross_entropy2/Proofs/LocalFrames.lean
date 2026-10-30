import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ForwardValues
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardKernel
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Lifetimes

namespace VeriTile.Bench.Repairs.CrossEntropy2.Frames
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind Observations Preserves Mutation IntegerViews)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Retains bytes Word)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Assertions include the state returned with an exception. -/
def Post {α σ β : Type} (ok : β → World α σ → Prop) (error : World α σ → Prop) : Step (World α σ) β → Prop
  | .ok out state => ok out state
  | .error _ state => error state

theorem bind_post {α σ β γ : Type} (result : Step (World α σ) β)
    (next : β → World α σ → Step (World α σ) γ)
    (P : β → World α σ → Prop) (Q : γ → World α σ → Prop) (F : World α σ → Prop)
    (first : Post P F result) (rest : ∀ out state, P out state → Post Q F (next out state)) :
    Post Q F (bind result next) := by
  cases result with
  | ok out state => exact rest out state first
  | error reason state => exact first

structure Frame {α σ : Type} (obs : Observations α σ) (layout : Layout σ) (kept : RegionName → Prop)
    (before after : World α σ) : Prop where
  storage : Preserves obs before after
  lifetime : Retains layout kept before.native after.native

theorem Frame.refl {α σ : Type} (obs : Observations α σ) (layout : Layout σ) (kept : RegionName → Prop)
    (s : World α σ) : Frame obs layout kept s s := ⟨Preserves.refl _ _,Retains.refl _ _ _⟩
theorem Frame.trans {α σ : Type} {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
    {a b c : World α σ} (ab : Frame obs layout kept a b) (bc : Frame obs layout kept b c) : Frame obs layout kept a c :=
  ⟨ab.storage.trans bc.storage,ab.lifetime.trans bc.lifetime⟩

structure Writes {α σ : Type} (obs : Observations α σ) (layout : Layout σ) (kept : RegionName → Prop)
    (dst : Tensor) (before after : World α σ) : Prop where
  storage : Mutation obs before after dst
  lifetime : Retains layout kept before.native after.native

theorem Frame.freshWrite {α σ : Type} {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
    {a b c : World α σ} {dst : Tensor} (ab : Frame obs layout kept a b) (bc : Writes obs layout kept dst b c)
    (fresh : ¬a.live dst.region) : Frame obs layout kept a c :=
  ⟨ab.storage.freshMutation bc.storage fresh,ab.lifetime.trans bc.lifetime⟩

/-- Exactly one tensor-library operation. These constructors do not mention
host wrappers, kernels, output formulas, or whether the call succeeds. -/
inductive ReadCall {α σ : Type} (rt : Runtime α σ) (t : Tensor) (s : World α σ) : Step (World α σ) Tensor → Prop
  | resolve : ReadCall rt t s (rt.resolveNeg t s)
  | contiguous : ReadCall rt t s (rt.contiguous t s)
inductive FreshCall {α σ : Type} (rt : Runtime α σ) (s : World α σ) : Step (World α σ) Tensor → Prop
  | empty (shape : List Nat) (dtype : String) (device : Nat) : FreshCall rt s (rt.empty shape dtype device s)
  | emptyLike (t : Tensor) (packed : Bool) : FreshCall rt s (rt.emptyLike t packed s)
  | lse (t : Tensor) : FreshCall rt s (rt.logsumexp0 t s)
  | sum (t : Tensor) : FreshCall rt s (rt.sum0 t s)
  | scale (a : ℝ) (t : Tensor) : FreshCall rt s (rt.scalarMul a t s)
  | multiply (a b : Tensor) : FreshCall rt s (rt.multiply a b s)
  | widen (t : Tensor) (dtype : t.dtype="torch.int32") : FreshCall rt s (rt.toInt64 t s)
  | compare (t : Tensor) (target : Int) : FreshCall rt s (rt.equalInt t target s)
inductive WriteCall {α σ : Type} (rt : Runtime α σ) (dst : Tensor) (s : World α σ) : Step (World α σ) Tensor → Prop
  | add (src : Tensor) : WriteCall rt dst s (rt.addInplace dst src s)
  | mask (flags : Tensor) : WriteCall rt dst s (rt.maskedFillZero dst flags s)
inductive SetupCall {α σ : Type} (rt : Runtime α σ) (s : World α σ) : Step (World α σ) Unit → Prop
  | forward (p : Parameters) (count : Nat) : SetupCall rt s (rt.setupForward2 p count s)
  | backward (p : Parameters) (count : Nat) : SetupCall rt s (rt.setupBackward2 p count s)

/-- Ordinary local storage/exception contracts. A failed in-place operation
may partially change its own destination; that destination is proved fresh
by the wrapper. Python references in `kept` retain their allocation records. -/
structure LocalLibrary {α σ : Type} (obs : Observations α σ) (layout : Layout σ) (kept : RegionName → Prop)
    (rt : Runtime α σ) : Prop where
  read : ∀ input s result, ReadCall rt input s result →
    Post (fun t after => Frame obs layout kept s after ∧ after.live t.region ∧ t.dtype=input.dtype) (Frame obs layout kept s) result
  fresh : ∀ s result, FreshCall rt s result →
    Post (fun t after => Frame obs layout kept s after ∧ after.live t.region ∧ ¬s.live t.region) (Frame obs layout kept s) result
  write : ∀ dst s result, WriteCall rt dst s result →
    Post (fun t after => t=dst ∧ Writes obs layout kept dst s after) (Writes obs layout kept dst s) result
  setup : ∀ s result, SetupCall rt s result →
    Post (fun _ after => Frame obs layout kept s after) (Frame obs layout kept s) result

variable {α σ : Type} {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
  {rt : Runtime α σ}

theorem read_frame (lib : LocalLibrary obs layout kept rt) (old before : World α σ)
    (frame : Frame obs layout kept old before) (input : Tensor) (result : Step (World α σ) Tensor) (call : ReadCall rt input before result) :
    Post (fun t after => Frame obs layout kept old after ∧ after.live t.region ∧ t.dtype=input.dtype) (Frame obs layout kept old) result := by
  have facts := lib.read input before result call
  cases result with
  | ok t after => exact ⟨frame.trans facts.1,facts.2⟩
  | error reason after => exact frame.trans facts

theorem fresh_frame (lib : LocalLibrary obs layout kept rt) (old before : World α σ)
    (frame : Frame obs layout kept old before) (result : Step (World α σ) Tensor) (call : FreshCall rt before result) :
    Post (fun t after => Frame obs layout kept old after ∧ after.live t.region ∧ ¬old.live t.region) (Frame obs layout kept old) result := by
  have facts := lib.fresh before result call
  cases result with
  | ok t after => exact ⟨frame.trans facts.1,facts.2.1,fun live => facts.2.2 (frame.storage.live _ live)⟩
  | error reason after => exact frame.trans facts

theorem write_frame (lib : LocalLibrary obs layout kept rt) (old before : World α σ)
    (frame : Frame obs layout kept old before) (dst : Tensor) (fresh : ¬old.live dst.region)
    (result : Step (World α σ) Tensor) (call : WriteCall rt dst before result) :
    Post (fun t after => t=dst ∧ Frame obs layout kept old after) (Frame obs layout kept old) result := by
  have facts := lib.write dst before result call
  cases result with
  | ok t after => exact ⟨facts.1,frame.freshWrite facts.2 fresh⟩
  | error reason after => exact frame.freshWrite facts fresh

theorem setup_frame (lib : LocalLibrary obs layout kept rt) (old before : World α σ)
    (frame : Frame obs layout kept old before) (result : Step (World α σ) Unit) (call : SetupCall rt before result) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old) result := by
  have facts := lib.setup before result call
  cases result with
  | ok t after => exact frame.trans facts
  | error reason after => exact frame.trans facts

theorem normalize_frame (lib : LocalLibrary obs layout kept rt) (old before : World α σ)
    (frame : Frame obs layout kept old before) (input : Tensor) :
    Post (fun t after => Frame obs layout kept old after ∧ after.live t.region ∧ t.dtype=input.dtype) (Frame obs layout kept old) (normalize rt input before) := by
  apply bind_post (rt.resolveNeg input before) _ _ _ _ (read_frame lib old before frame input _ .resolve)
  intro resolved middle h
  have facts := read_frame lib old middle h.1 resolved _ .contiguous
  cases result : rt.contiguous resolved middle with
  | ok t after =>
    rw [result] at facts
    exact ⟨facts.1,facts.2.1,facts.2.2.trans h.2.2⟩
  | error reason after => simpa only [result] using facts

#axiomsClean bind_post
#axiomsClean normalize_frame
#axiomsClean write_frame
#specNonCircular LocalLibrary avoiding [Host.forward,Host.backward,Host.aggregate,Source.forward,Source.backward,Grid.forward,BackwardGrid.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Frames

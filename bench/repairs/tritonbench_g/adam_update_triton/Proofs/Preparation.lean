import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Runtime
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
noncomputable section

def Call.metadata (call : Call) : Role → TensorMetadata
  | .a => call.parameter | .b => call.momentum | .third => call.gradient

theorem Bindings.tensor {call : Call} (binding : Bindings call) (role : Role) :
    TensorBinding (call.metadata role) (call.request.view role) := by
  cases role
  · exact binding.parameter
  · exact binding.momentum
  · exact binding.gradient

theorem Bindings.count {call : Call} (binding : Bindings call) (role : Role) :
    (call.metadata role).numel=(call.request.view role).numel :=
  congrArg List.prod (binding.tensor role).shape

structure Prepared (q : Request) (b : Buffers) : Prop where
  window : ∀ role,Window (q.view role) ((b.layout q).allocation role)

theorem prepare_sound {M : Type} (rt : Runtime M) (lib : Library rt) (q : Request)
    (before after : World M) (b : Buffers) (success : prepare rt q before=.ok b after) :
    Prepared q b ∧ after=before := by
  unfold prepare at success
  cases hp : rt.flatten q.p before with
  | error reason failed => simp [hp,Result.bind] at success
  | ok p s1 =>
    rw [hp] at success
    simp only [Result.bind] at success
    obtain ⟨wp,hpstate⟩:=lib.flatten _ _ _ _ hp
    subst s1
    cases hg : rt.flatten q.grad before with
    | error reason failed => simp [hg,Result.bind] at success
    | ok g s2 =>
      rw [hg] at success
      simp only [Result.bind] at success
      obtain ⟨wg,hgstate⟩:=lib.flatten _ _ _ _ hg
      subst s2
      cases hm : rt.flatten q.momentum before with
      | error reason failed => simp [hm,Result.bind] at success
      | ok m s3 =>
        rw [hm] at success
        simp only [Result.bind] at success
        obtain ⟨wm,hmstate⟩:=lib.flatten _ _ _ _ hm
        subst s3
        obtain ⟨rfl,rfl⟩:=Result.ok.inj success
        refine ⟨⟨?_⟩,rfl⟩
        intro role
        cases role
        · exact wp
        · exact wm
        · exact wg

theorem Window.pointer (metadata : TensorMetadata) (input : View) (a : Allocation)
    (binding : TensorBinding metadata input) (window : Window input a) (positive : 0< input.numel) :
    a.baseByte=metadata.dataPtr.toNat := by
  have key:=congrArg Key.address (window.keys 0 positive)
  simp only [Allocation.key,Nat.mul_zero,Nat.add_zero] at key
  have ptr:=binding.pointer positive
  rw [←key] at ptr
  have bounds:=a.representable
  have bytes:=window.bytes
  have small : a.baseByte<2^64 := by omega
  rw [ptr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt small]

theorem Window.separate (x y : TensorMetadata) (vx vy : View) (a b : Allocation)
    (bx : TensorBinding x vx) (by' : TensorBinding y vy) (ax : Window vx a) (ay : Window vy b)
    (px : 0<vx.numel) (py : 0<vy.numel) (separate : DisjointRanges x y) : a.separate b := by
  have cx : x.numel=vx.numel := congrArg List.prod bx.shape
  have cy : y.numel=vy.numel := congrArg List.prod by'.shape
  unfold Allocation.separate
  right
  rw [ax.pointer x vx a bx px,ay.pointer y vy b by' py,ax.bytes,ay.bytes,←cx,←cy]
  exact separate

private theorem separate_symm {a b : Allocation} (h : a.separate b) : b.separate a := by
  rcases h with h|h|h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem Prepared.valid (call : Call) (admitted : Admitted call) (binding : Bindings call)
    (b : Buffers) (prepared : Prepared call.request b) (nonempty : 0<call.request.count) :
    Private.Valid (b.layout call.request) := by
  have cp:=binding.count .a
  have cg:=binding.count .third
  have cm:=binding.count .b
  change call.parameter.numel=call.request.count at cp
  change call.gradient.numel=call.request.grad.numel at cg
  change call.momentum.numel=call.request.momentum.numel at cm
  have positiveP : 0<call.parameter.numel := by change call.parameter.numel=call.request.count at cp;omega
  have positive (role : Role) : 0<(call.request.view role).numel := by
    cases role
    · exact nonempty
    · have := admitted.enough.2;change 0<call.request.momentum.numel;omega
    · have := admitted.enough.1;change 0<call.request.grad.numel;omega
  have sep:=admitted.separate positiveP
  have pm:=Window.separate _ _ _ _ _ _ binding.parameter binding.momentum
    (prepared.window .a) (prepared.window .b) (positive .a) (positive .b) sep.2.1
  have pg:=Window.separate _ _ _ _ _ _ binding.parameter binding.gradient
    (prepared.window .a) (prepared.window .third) (positive .a) (positive .third) sep.1
  have gm:=Window.separate _ _ _ _ _ _ binding.gradient binding.momentum
    (prepared.window .third) (prepared.window .b) (positive .third) (positive .b) sep.2.2
  refine ⟨by change (1:Nat)≤1;omega,?_,?_⟩
  · intro role
    simp only [Private.Layout.flat,Nat.mul_one,Allocation.words]
    rw [(prepared.window role).bytes]
    have eg:=admitted.enough.1
    have em:=admitted.enough.2
    simp only [Request.count] at cp
    cases role <;> simp [Request.view,Buffers.layout,Request.count] <;> omega
  · intro a c different
    cases a <;> cases c
    · exact False.elim (different rfl)
    · exact pm
    · exact pg
    · exact separate_symm pm
    · exact False.elim (different rfl)
    · exact separate_symm gm
    · exact separate_symm pg
    · exact gm
    · exact False.elim (different rfl)

theorem Prepared.keys (q : Request) (b : Buffers) (prepared : Prepared q b)
    (role : Role) (i : Nat) (hi : i<(q.view role).numel) :
    ((b.layout q).allocation role).key i=(q.view role).key i :=
  (prepared.window role).keys i hi

#axiomsClean prepare_sound
#axiomsClean Prepared.valid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Host

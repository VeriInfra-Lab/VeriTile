import bench.repairs.tritonbench_g.kldiv_ops.Proofs.HostForward

namespace VeriTile.Bench.Repairs.KldivOps.Host.Witness
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (coordinateOffset)
noncomputable section
set_option maxHeartbeats 6000000
attribute [local instance] Classical.propDecidable

/-- A consistency-model constructor, used only by the inhabitance theorem.
It checks ordinary primitive postconditions, never a KLDIV result. Production
host correctness above quantifies over the unmodified ordinary Services. -/
def validate (post : State→A→State→Prop) (action : Action A) : Action A := fun s=>
  match action s with
  | .error e t=>.error e t
  | .ok a t=>if post s a t then .ok a t else .error "local contract unavailable" s

theorem validate_success {post : State→A→State→Prop} {action : Action A} {s t : State} {a : A}
    (h : validate post action s=.ok a t) : post s a t := by
  cases hr : action s with
  | error e u=>simp [validate,hr] at h
  | ok b u=>
    by_cases valid : post s b u
    · simp only [validate,hr,if_pos valid,Result.ok.injEq] at h
      rcases h with ⟨rfl,rfl⟩
      exact valid
    · simp [validate,hr,valid] at h

theorem validate_ok (post : State→A→State→Prop) (action : Action A) (s t : State) (a : A)
    (h : action s=.ok a t) (valid : post s a t) : validate post action s=.ok a t := by
  simp [validate,h,valid]

def checked {Word : Type} (codec : Codec Word) (num : Numerics Word) (raw : Services Word) : Services Word where
  resolveConj a := validate (fun s b t=>a.backing∈s.live → Materialized codec num.neg a b s t) (raw.resolveConj a)
  resolveNeg a := validate (fun s b t=>a.backing∈s.live → Materialized codec num.neg a b s t ∧ b.negative=Bool.false) (raw.resolveNeg a)
  contiguous a := validate (fun s b t=>a.backing∈s.live → a.negative=Bool.false →
    Materialized codec num.neg a b s t ∧ Dense b) (raw.contiguous a)
  clone a := validate (fun s b t=>a.backing∈s.live → a.negative=Bool.false →
    Materialized codec num.neg a b s t ∧ b.negative=Bool.false ∧ (∀ r,r∈s.live → b.backing.disjoint r)) (raw.clone a)
  zeros sh device := validate (fun s b t=>Allocated sh device b s t ∧
    (∀ i,i<b.numel → value codec num.neg t.heap b i=num.zero)) (raw.zeros sh device)
  emptyContiguousLike a := validate (fun s b t=>a.backing∈s.live → Allocated a.shape a.device b s t) (raw.emptyContiguousLike a)
  copy dst src := validate (fun s b t=>dst.backing∈s.live → src.backing∈s.live → dst.shape=src.shape →
    Copied codec num.neg dst src b s t) (raw.copy dst src)
  sum a dim0 := validate (fun s b t=>a.backing∈s.live → a.shape.length=1 →
    Allocated [] a.device b s t ∧ value codec num.neg t.heap b 0=
      num.sum dim0 ((List.range a.numel).map (value codec num.neg s.heap a))) (raw.sum a dim0)
  divNat a n := validate (fun s b t=>a.backing∈s.live → a.shape=[] →
    Allocated [] a.device b s t ∧ value codec num.neg t.heap b 0=num.divide (value codec num.neg s.heap a 0) n) (raw.divNat a n)
  scalarOne device := validate (fun s b t=>Allocated [] device b s t ∧ value codec num.neg t.heap b 0=num.one) (raw.scalarOne device)
  equal a b := validate (fun s result t=>a.backing∈s.live → b.backing∈s.live →
    t=s ∧ result=tensorEqual codec num s.heap a b) (raw.equal a b)
  multiply a b := validate (fun s out t=>a.backing∈s.live → b.backing∈s.live →
    broadcastShape a.shape b.shape=some out.shape ∧ Allocated out.shape a.device out s t ∧
    (∀ i,i<out.numel → value codec num.neg t.heap out i=num.multiply
      (value codec num.neg s.heap a (broadcastIndex out.shape a.shape i))
      (value codec num.neg s.heap b (broadcastIndex out.shape b.shape i)))) (raw.multiply a b)
  configure rows cols := validate (fun s _ t=>t=s) (raw.configure rows cols)

theorem checked_library {Word : Type} (codec : Codec Word) (num : Numerics Word) (raw : Services Word) :
    Library codec num (checked codec num raw) where
  conj := by intro a s b t live h; dsimp only [checked] at h; exact (validate_success h) live
  neg := by intro a s b t live h; dsimp only [checked] at h; exact (validate_success h) live
  contiguous := by intro a s b t live normal h; dsimp only [checked] at h; exact (validate_success h) live normal
  clone := by intro a s b t live normal h; dsimp only [checked] at h; exact (validate_success h) live normal
  zeros := by intro sh device s b t h; dsimp only [checked] at h; exact validate_success h
  empty := by intro a s b t live h; dsimp only [checked] at h; exact (validate_success h) live
  copy := by intro dst src s b t dl sl shape h; dsimp only [checked] at h; exact (validate_success h) dl sl shape
  sum := by intro a dim0 s b t live shape h; dsimp only [checked] at h; exact (validate_success h) live shape
  divide := by intro a n s b t live shape h; dsimp only [checked] at h; exact (validate_success h) live shape
  one := by intro device s b t h; dsimp only [checked] at h; exact validate_success h
  equal := by intro a b s result t al bl h; dsimp only [checked] at h; exact (validate_success h) al bl
  multiply := by intro a b s out t al bl h; dsimp only [checked] at h; exact (validate_success h) al bl
  configure := by intro rows cols s t h; dsimp only [checked] at h; exact validate_success h

def input : Tensor where
  objectId := 1
  shape := [1,1]
  strides := [1,1]
  offset := 0
  backing := ⟨4,4,by norm_num⟩
  negative := Bool.false
  device := 0
  rank := rfl
  capacity := by
    intro i hi
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at hi
    have iz : i=0 := by omega
    subst i
    norm_num [coordinateOffset]

def output : Tensor := {input with objectId:=2,backing:=⟨16,4,by norm_num⟩}

theorem input_dense : Dense input := by
  refine ⟨rfl,?_,by norm_num [input]⟩
  intro i hi
  have iz : i=0 := by simpa [input,Tensor.numel] using hi
  subst i
  simp [input,Tensor.index,coordinateOffset]

theorem output_dense : Dense output := by
  refine ⟨rfl,input_dense.index,by norm_num [output]⟩

def entry (m : Heap) : State := ⟨m,[input.backing]⟩
def allocated {Word : Type} (codec : Codec Word) (num : Numerics Word) (s : State) : State :=
  ⟨store codec s.heap 16 num.zero,output.backing::s.live⟩

def unavailable : Action A := fun s=>.error "fixture primitive unused" s

def raw {Word : Type} (codec : Codec Word) (num : Numerics Word) : Services Word where
  resolveConj a := pure a
  resolveNeg a := pure a
  contiguous a := pure a
  clone _ := unavailable
  zeros _ _ := fun s=>.ok output (allocated codec num s)
  emptyContiguousLike _ := unavailable
  copy _ _ := unavailable
  sum _ _ := unavailable
  divNat _ _ := unavailable
  scalarOne _ := unavailable
  equal _ _ := unavailable
  multiply _ _ := unavailable
  configure _ _ := pure ()

theorem materialized_identity {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (a : Tensor) (s : State) (live : a.backing∈s.live) : Materialized codec num.neg a a s s :=
  ⟨rfl,rfl,Preserves.refl s,live,fun _ _=>rfl⟩

theorem prepare_identity {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (a : Tensor) (s : State) (dense : Dense a) : prepare (checked codec num (raw codec num)) a s=.ok a s := by
  have c : (checked codec num (raw codec num)).resolveConj a s=.ok a s := by
    dsimp only [checked,raw]
    apply validate_ok
    · rfl
    · exact materialized_identity codec num a s
  have n : (checked codec num (raw codec num)).resolveNeg a s=.ok a s := by
    dsimp only [checked,raw]
    apply validate_ok
    · rfl
    · intro live
      exact ⟨materialized_identity codec num a s live,dense.normal⟩
  have p : (checked codec num (raw codec num)).contiguous a s=.ok a s := by
    dsimp only [checked,raw]
    apply validate_ok
    · rfl
    · intro live _
      exact ⟨materialized_identity codec num a s live,dense⟩
  simp [prepare,normalize,Bind.bind,bind,c,n,p]

theorem zeros_contract {Word : Type} (codec : Codec Word) (num : Numerics Word) (m : Heap) :
    Allocated [1,1] 0 output (entry m) (allocated codec num (entry m)) ∧
      (∀ i,i<output.numel → value codec num.neg (allocated codec num (entry m)).heap output i=num.zero) := by
  constructor
  · refine ⟨rfl,rfl,output_dense,?_,by simp [allocated],?_⟩
    · refine ⟨by intro b hb; simp [allocated,hb],?_⟩
      intro b hb
      have eq : b=input.backing := by simpa [entry] using hb
      subst b
      intro p hp
      apply store_frame
      simp [input,Backing.contains] at hp
      omega
    · intro b hb
      have eq : b=input.backing := by simpa [entry] using hb
      subst b
      norm_num [output,input,Backing.disjoint]
  · intro i hi
    have iz : i=0 := by simpa [output,input,Tensor.numel] using hi
    subst i
    change load codec (store codec m 16 num.zero) 16=num.zero
    exact load_store codec m 16 num.zero

def compiler : Compiler where
  forward rows cols p t o red := fun s=>
    if h : Nonempty (ForwardPlan rows cols p t o red) then .ok (Classical.choice h) s else .error "no dense plan" s
  backward rows cols t o := fun s=>
    if h : Nonempty (BackwardPlan rows cols t o) then .ok (Classical.choice h) s else .error "no dense plan" s
  forward_state := by intro rows cols p t o red s plan after h; split at h <;> cases h <;> rfl
  backward_state := by intro rows cols t o s plan after h; split at h <;> cases h <;> rfl

def plan : ForwardPlan 1 1 input input output .none :=
  ⟨input_dense,input_dense,output_dense,rfl,rfl,by norm_num [input,output,Backing.disjoint],by norm_num [input,output,Backing.disjoint]⟩

/-- A full nonempty host call inhabits every Library and compiler hypothesis.
The inputs alias, and every output byte comes from the actual typed grid for
an arbitrary FP algebra and an ordinary four-byte codec. -/
theorem whole_forward_inhabited {Word : Type} [Inhabited Word] (codec : Codec Word)
    (num : Numerics Word) (M : Algebra Word) (m : Heap) (flag : Bool) (eps : Word) :
    ∃ api : Services Word,Library codec num api ∧ ∃ final,
      forward api (sourceActions codec M compiler) input input flag "none" eps (entry m)=.ok output final := by
  let api := checked codec num (raw codec num)
  have z : api.zeros [1,1] 0 (entry m)=.ok output (allocated codec num (entry m)) := by
    dsimp only [api,checked,raw]
    apply validate_ok
    · rfl
    · exact zeros_contract codec num m
  have cfg : api.configure 1 1 (entry m)=.ok () (entry m) := by
    dsimp only [api,checked,raw]
    apply validate_ok
    · rfl
    · rfl
  let chosen := Classical.choice (show Nonempty (ForwardPlan 1 1 input input output .none) from ⟨plan⟩)
  have setup : compiler.forward 1 1 input input output .none (allocated codec num (entry m))=
      .ok chosen (allocated codec num (entry m)) := by
    simp [compiler,chosen,show Nonempty (ForwardPlan 1 1 input input output .none) from ⟨plan⟩]
  obtain ⟨state,run,model⟩ := Program.grid_complete M (params 1 1) flag 0 eps
    (represented codec chosen.arena (allocated codec num (entry m)).heap) (block_positive 1)
  let final : State := {allocated codec num (entry m) with heap:=commit codec chosen.arena state (allocated codec num (entry m)).heap}
  have launch : launchForward codec M compiler 1 1 input input output flag .none eps
      (allocated codec num (entry m))=.ok () final := by
    simp [launchForward,bind,setup,Reduction.code,run,final]
  refine ⟨api,checked_library codec num _,final,?_⟩
  have shape : shape2 input (entry m)=.ok (1,1) (entry m) := rfl
  have cap : requireCapacity input (1*1) "target has fewer elements than prediction"
      (allocated codec num (entry m))=.ok () (allocated codec num (entry m)) := by
    simp [requireCapacity,input,Tensor.numel]
  have prep : prepare api input (allocated codec num (entry m))=.ok input (allocated codec num (entry m)) :=
    prepare_identity codec num input (allocated codec num (entry m)) input_dense
  simp [forward,Bind.bind,bind,shape,cfg,lookup,Reduction.outputShape,z,sourceActions,launch,pure,prep,cap,
    show input.device=0 from rfl]

#specNonCircular checked avoiding [forward,backward,Program.grid,Program.gridModel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host.Witness

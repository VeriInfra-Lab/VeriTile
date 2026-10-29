import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.DirectReal
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Binding

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.DirectNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Native Direct
noncomputable section
open Classical
set_option maxHeartbeats 6000000

abbrev Heap := Nat → BitVec 8
def load32 (heap : Heap) (base off : Nat) : Word :=
  BitVec.ofNat 32 (((List.range 4).map fun b => (heap (base+4*off+b)).toNat*256^b).sum)

/-- Original physical FP32 words plus the wrapper's intentional DG fill. -/
def initial (p : Metadata) (a : Direct.Allocation p) (heap : Heap) : State Word where
  mem := fun r off => .mk .fp32
    (if r="dg" ∧ off<extent (config p) "dg" then 0xce6e6b28
     else load32 heap (pointer p a (ofRegion r)) off)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

def runtime (p : Metadata) : Runtime := ⟨scaleBits p⟩

theorem scale_binding (p : Metadata) (h : InputContract p) : ScaleBinding (runtime p) (config p) := by
  cases hs : scaleRat p with
  | none => simpa [hs] using h.finite
  | some q =>
    refine ⟨q,hs,?_⟩
    simp [config,hs]

/-- Selected compiler/profile and layout mapping is an external structural
boundary. It contains no results, executions, or variable-size numerical laws. -/
structure CompilerMapping (p : Metadata) where
  descriptor : Fin 2 → Descriptor
  selected : Fin 2 → Descriptor
  binding : ∀ choice,Binding (descriptor choice) (config p) (selected choice)
  warps : ∀ choice,(descriptor choice).warps=if choice.val=0 then .four else .eight

def staged (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word) : State Stage :=
  (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose

theorem staged_run (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word) :
    Typed.runGrid (algebra d rt) dotModel c (allCTAs c) (injectState s)=some (staged d rt c s) :=
  (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose_spec.1

def one (I : Instructions) (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word) : State Word :=
  projectState I d rt s (staged d rt c s)

theorem one_q (I : Instructions) (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word)
    (hd : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    (one I d rt c s).mem "dq" ((hd.val*c.T+t.val)*c.K+e.val)=.mk .fp32 (outputQ I d rt c s hd t e) := by
  have value := (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose_spec.2.1 hd t e
  exact congrArg (projectCell I d rt) value

theorem one_k (I : Instructions) (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word)
    (hd : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    (one I d rt c s).mem "dk" ((hd.val*c.T+t.val)*c.K+e.val)=.mk .fp32 (outputK I d rt c s hd t e) := by
  have value := (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose_spec.2.2.1 hd t e
  exact congrArg (projectCell I d rt) value

theorem one_g (I : Instructions) (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word)
    (key : Fin c.NK) (hd : Fin c.heads) (t : Fin c.T) :
    (one I d rt c s).mem "dg" ((key.val*c.heads+hd.val)*c.T+t.val)=.mk .real (outputG I d rt c s key hd t) := by
  have value := (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose_spec.2.2.2.1 key hd t
  exact congrArg (projectCell I d rt) value

theorem one_frame (I : Instructions) (d : Descriptor) (rt : Runtime) (c : Config) (s : State Word)
    (r : RegionName) (off : Nat) (outside : ¬gridWrites c (allCTAs c) (r,off)) :
    (one I d rt c s).mem r off=s.mem r off := by
  have same := (Typed.whole_result (algebra d rt) dotModel c (injectState s) (allCTAs c) (List.Perm.refl _)).choose_spec.2.2.2.2 r off outside
  change (staged d rt c s).mem r off=(injectState s).mem r off at same
  change projectCell I d rt ((staged d rt c s).mem r off)=_
  rw [same,injectState_project]

def trials (I : Instructions) (p : Metadata) (mapping : CompilerMapping p) : List (Fin 2) → State Word → State Word
  | [],s => s
  | choice::rest,s => trials I p mapping rest (one I (mapping.selected choice) (runtime p) (config p) s)

def execute (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (s : State Word) : State Word :=
  one I (mapping.selected chosen) (runtime p) (config p) (trials I p mapping choices s)

def Certified (I : Instructions) (p : Metadata) (mapping : CompilerMapping p) : List (Fin 2) → State Word → Prop
  | [],_ => True
  | choice::rest,s =>
      (∃ t,Typed.runGrid (algebra (mapping.selected choice) (runtime p)) dotModel (config p)
        (allCTAs (config p)) (injectState s)=some t ∧
        (one I (mapping.selected choice) (runtime p) (config p) s).mem=
          (projectState I (mapping.selected choice) (runtime p) s t).mem) ∧
      Certified I p mapping rest (one I (mapping.selected choice) (runtime p) (config p) s)

theorem certified (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (s : State Word) : Certified I p mapping choices s := by
  induction choices generalizing s with
  | nil => trivial
  | cons choice rest ih =>
    exact ⟨⟨staged _ _ _ _,staged_run _ _ _ _,rfl⟩,ih _⟩

theorem trials_frame (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (s : State Word) (r : RegionName) (off : Nat)
    (outside : ¬gridWrites (config p) (allCTAs (config p)) (r,off)) :
    (trials I p mapping choices s).mem r off=s.mem r off := by
  induction choices generalizing s with
  | nil => rfl
  | cons choice rest ih =>
    exact (ih _).trans (one_frame I _ _ _ s r off outside)

theorem trials_inputs (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (s : State Word) :
    Typed.SameInputs (injectState (trials I p mapping choices s)) (injectState s) := by
  intro r hr
  funext off
  have same := trials_frame I p mapping choices s r off (by
    rintro ⟨a,_,hw⟩
    exact input_unwritten (config p) a r off hr hw)
  change (match (trials I p mapping choices s).mem r off with | .mk d v => (Cell.mk d (inject d v) : Cell Stage))=_
  rw [same]
  rfl

theorem outputs_from_original (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (s : State Word) :
    (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
      (execute I p mapping choices chosen s).mem "dq" ((hd.val*(config p).T+t.val)*(config p).K+e.val)=
        .mk .fp32 (outputQ I (mapping.descriptor chosen) (runtime p) (config p) s hd t e)) ∧
    (∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
      (execute I p mapping choices chosen s).mem "dk" ((hd.val*(config p).T+t.val)*(config p).K+e.val)=
        .mk .fp32 (outputK I (mapping.descriptor chosen) (runtime p) (config p) s hd t e)) ∧
    (∀ (key : Fin (config p).NK) (hd : Fin (config p).heads) (t : Fin (config p).T),
      (execute I p mapping choices chosen s).mem "dg" ((key.val*(config p).heads+hd.val)*(config p).T+t.val)=
        .mk .real (outputG I (mapping.descriptor chosen) (runtime p) (config p) s key hd t)) := by
  have same := trials_inputs I p mapping choices s
  have values (a : CTA (config p)) := Typed.results_inputs (algebra (mapping.descriptor chosen) (runtime p)) dotModel
    (config p) (injectState (trials I p mapping choices s)) (injectState s) a same
  constructor
  · intro hd t e
    rw [execute,one_q]
    rw [(mapping.binding chosen).selectedSyntax]
    simp only [outputQ,referenceQ,Typed.expectedQ,(values (qOwner (config p) hd t e)).1]
  constructor
  · intro hd t e
    rw [execute,one_k]
    rw [(mapping.binding chosen).selectedSyntax]
    simp only [outputK,referenceK,Typed.expectedK,(values (qOwner (config p) hd t e)).2.1]
  · intro key hd t
    rw [execute,one_g]
    rw [(mapping.binding chosen).selectedSyntax]
    simp only [outputG,referenceG,Typed.expectedG,(values (gOwner (config p) key hd t)).2.2]

def OutputBytes (p : Metadata) (a : Direct.Allocation p) (i : Nat) : Prop :=
  ∃ r : Direct.Role,¬IsInput r ∧ pointer p a r ≤ i ∧ i < pointer p a r+4*extent (config p) r.region

/-- Serialize only the three returned FP32 buffers. Original bytes outside
their spans are unchanged, with no byte-binding assumption from the caller. -/
def outputHeap (p : Metadata) (a : Direct.Allocation p) (heap : Heap) (s : State Word) (i : Nat) : BitVec 8 :=
  if OutputBytes p a i then
    match (flat p a).decode (i/4) with
    | some (r,off) => byteOf (encode (s.mem r off)) (i%4)
    | none => heap i
  else heap i

theorem output_frame (p : Metadata) (a : Direct.Allocation p) (heap : Heap) (s : State Word)
    (i : Nat) (outside : ¬OutputBytes p a i) : outputHeap p a heap s i=heap i := by
  simp [outputHeap,outside]

theorem output_byte (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (heap : Heap) (s : State Word) (r : Direct.Role) (out : ¬IsInput r)
    (off : Nat) (bound : off<extent (config p) r.region) (b : Fin 4) :
    outputHeap p a heap s (pointer p a r+4*off+b.val)=byteOf (encode (s.mem r.region off)) b.val := by
  have alignment := output_base p a r out
  have lt := b.isLt
  have inside : OutputBytes p a (pointer p a r+4*off+b.val) := ⟨r,out,by omega,by omega⟩
  have div : (pointer p a r+4*off+b.val)/4=(flat p a).addr r.region off := by
    unfold FlatAlloc.addr
    omega
  have mod : (pointer p a r+4*off+b.val)%4=b.val := by omega
  have listed : r.region∈(flat p a).regions := by
    change r.region∈regions
    rw [←region_list]
    exact List.mem_map.mpr ⟨r,role_list r,rfl⟩
  have decoded := (flat p a).decode_addr (denseAllocation p h a).apart listed bound
  simp [outputHeap,inside,div,decoded,mod]

theorem input_byte_frame (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (heap : Heap) (s : State Word) (r : Direct.Role) (input : IsInput r)
    (off : Nat) (bound : off<extent (config p) r.region) (b : Fin 4) :
    outputHeap p a heap s (pointer p a r+4*off+b.val)=heap (pointer p a r+4*off+b.val) := by
  apply output_frame
  rintro ⟨out,ho,low,high⟩
  have ne : out≠r := by intro eq; exact ho (eq ▸ input)
  have separate := pointer_separate p h a out r ne
  have lt := b.isLt
  omega

structure OutputWords (I : Instructions) (p : Metadata) (mapping : CompilerMapping p)
    (chosen : Fin 2) (before after : State Word) : Prop where
  q : ∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
    encode (after.mem "dq" ((hd.val*(config p).T+t.val)*(config p).K+e.val))=
      outputQ I (mapping.descriptor chosen) (runtime p) (config p) before hd t e
  k : ∀ (hd : Fin (config p).heads) (t : Fin (config p).T) (e : Fin (config p).K),
    encode (after.mem "dk" ((hd.val*(config p).T+t.val)*(config p).K+e.val))=
      outputK I (mapping.descriptor chosen) (runtime p) (config p) before hd t e
  g : ∀ (key : Fin (config p).NK) (hd : Fin (config p).heads) (t : Fin (config p).T),
    encode (after.mem "dg" ((key.val*(config p).heads+hd.val)*(config p).T+t.val))=
      outputG I (mapping.descriptor chosen) (runtime p) (config p) before key hd t

theorem words_if_local (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Metadata) (mapping : CompilerMapping p) (choices : List (Fin 2)) (chosen : Fin 2) (s : State Word) :
    OutputWords standard p mapping chosen s (execute actual p mapping choices chosen s) := by
  obtain ⟨hq,hk,hg⟩ := outputs_from_original actual p mapping choices chosen s
  have eq (g : Graph) : g.eval actual.ops=g.eval standard.ops :=
    transport id actual.ops standard.ops (agreement.transport actual standard) g
  constructor
  · intro hd t e
    rw [hq]
    exact eq _
  · intro hd t e
    rw [hk]
    exact eq _
  · intro key hd t
    rw [hg]
    exact eq _

structure Result where
  state : State Word
  tensors : ReturnedTensor × ReturnedTensor × ReturnedTensor
  heap : Heap

def result (I : Instructions) (p : Metadata) (a : Direct.Allocation p) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) : Result :=
  let final := execute I p mapping choices chosen (initial p a heap)
  ⟨final,(returned p a .dq,returned p a .dk,returned p a .dg),outputHeap p a heap final⟩

def wrapper (I : Instructions) (p : Metadata) (a : Direct.Allocation p) (mapping : CompilerMapping p)
    (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) : Except InputError Result := do
  checkInputs p
  checkOutputs p a
  pure (result I p a mapping choices chosen heap)

structure Certificate (actual standard : Instructions) (p : Metadata) (a : Direct.Allocation p)
    (mapping : CompilerMapping p) (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) (result : Result) : Prop where
  tensors : result.tensors=(returned p a .dq,returned p a .dk,returned p a .dg)
  words : OutputWords standard p mapping chosen (initial p a heap) result.state
  source : Nonempty p → Certified actual p mapping (choices++[chosen]) (initial p a heap)
  serialization : ∀ r,¬IsInput r → ∀ off,off<extent (config p) r.region → ∀ b : Fin 4,
    result.heap (pointer p a r+4*off+b.val)=byteOf (encode (result.state.mem r.region off)) b.val
  frame : ∀ i,¬OutputBytes p a i → result.heap i=heap i
  inputFrame : ∀ r,IsInput r → ∀ off,off<extent (config p) r.region → ∀ b : Fin 4,
    result.heap (pointer p a r+4*off+b.val)=heap (pointer p a r+4*off+b.val)
  scale : ScaleBinding (runtime p) (config p)
  compiler : ∀ choice,(mapping.selected choice).Admissible (config p) ∧
    (mapping.selected choice).warps=(if choice.val=0 then .four else .eight)
  safety : Nonempty p → sourceLayout p=(config p).layout ∧ Guard (config p) ∧
    (∀ r off,off<envelope (config p) → pointer p a r+4*off<2^64) ∧
    (∀ (cta : CTA (config p)) (s : BlockState),
      (kernel (config p)).eraseDType.toAlgKernel.TraceSafe (flat p a).extent (Whole.prepare (config p) s cta))

theorem complete (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p) (outputs : OutputGuard p a)
    (mapping : CompilerMapping p) (choices : List (Fin 2)) (chosen : Fin 2) (heap : Heap) :
    Certificate actual standard p a mapping choices chosen heap (result actual p a mapping choices chosen heap) := by
  constructor
  · rfl
  · exact words_if_local actual standard agreement p mapping choices chosen (initial p a heap)
  · intro _
    exact certified actual p mapping (choices++[chosen]) (initial p a heap)
  · exact output_byte p h a heap _
  · exact output_frame p a heap _
  · exact input_byte_frame p h a heap _
  · exact scale_binding p h
  · intro choice
    rw [(mapping.binding choice).selectedSyntax]
    exact ⟨(mapping.binding choice).admissible,mapping.warps choice⟩
  · intro nonempty
    exact ⟨sourceLayout_eq p h nonempty,native_guard p h nonempty,formed_safe p h a nonempty outputs,
      fun cta s => source_safe_dense (config p) s cta⟩

#axiomsClean staged_run
#axiomsClean certified
#axiomsClean outputs_from_original
#axiomsClean output_byte
#axiomsClean output_frame
#axiomsClean input_byte_frame
#axiomsClean words_if_local
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.DirectNative

import bench.repairs.tritonbench_g.fused_recurrent_retention.Contract
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.AddressAdmission
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Scope
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission

set_option maxRecDepth 4000
set_option maxHeartbeats 800000

theorem checked_tensor (current : Nat) (dims : Fin 4 → Nat) (dense : Bool) (x : TensorObject)
    (ok : tensorCheck current dims dense x=Bool.true) :
    x.RawFP32 current ∧ x.dimensions=dims ∧
    (dense=Bool.true → x.view.strides=contiguous4 (dims 0) (dims 1) (dims 2) (dims 3)) ∧
    lastOffset x<x.view.capacity := by
  simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at ok
  exact ⟨ok.1.1.1,ok.1.1.2,fun h=>by simpa [h] using ok.1.2,ok.2⟩

theorem checked_view (current : Nat) (dims : Fin 4 → Nat) (dense : Bool) (x : TensorObject)
    (ok : tensorCheck current dims dense x=Bool.true)
    (a h t c : Nat) (ha : a<dims 0) (hh : h<dims 1) (ht : t<dims 2) (hc : c<dims 3) :
    x.view.InBounds a h t c := by
  obtain ⟨_,shape,_,bound⟩ := checked_tensor current dims dense x ok
  unfold lastOffset at bound
  rw [shape] at bound
  have a' : a≤dims 0-1 := by omega
  have h' : h≤dims 1-1 := by omega
  have t' : t≤dims 2-1 := by omega
  have c' : c≤dims 3-1 := by omega
  unfold Tensor4.InBounds
  apply lt_of_le_of_lt _ bound
  gcongr

theorem checked_dense_read (s : Shape) (current width : Nat) (x : TensorObject)
    (ok : tensorCheck current ![s.B,s.H,s.T,width] Bool.true x=Bool.true)
    (hw : 0<width) (h t c : Nat) (hh : h<s.BH) (ht : t<s.T) (hc : c<width) :
    0≤(x.view.offset+h*x.view.strides 1+t*width+c : Nat) ∧
    x.view.offset+h*x.view.strides 1+t*width+c<x.view.capacity := by
  have hb := head_batch_bounds s h hh
  have read := checked_view current ![s.B,s.H,s.T,width] Bool.true x ok
    (h/s.H) (h%s.H) t c hb.1 hb.2 ht hc
  have dense := (checked_tensor current _ _ x ok).2.2.1 rfl
  change x.view.strides=contiguous4 s.B s.H s.T width at dense
  have addr := dense_source_address s width h t c hh ht hw
  unfold Tensor4.InBounds at read
  rw [dense] at read ⊢
  constructor
  · omega
  · calc
      _ = x.view.offset+((h/s.H)*contiguous4 s.B s.H s.T width 0+
        (h%s.H)*contiguous4 s.B s.H s.T width 1+t*contiguous4 s.B s.H s.T width 2+
        c*contiguous4 s.B s.H s.T width 3) := by rw [addr]; omega
      _ < _ := by omega

theorem checked_seed_read (s : Shape) (current : Nat) (x : TensorObject)
    (ok : tensorCheck current ![s.B,s.H,s.DK,s.DV] Bool.true x=Bool.true)
    (hk : 0<s.DK) (hv : 0<s.DV) (h k v : Nat)
    (hh : h<s.BH) (hkey : k<s.DK) (hval : v<s.DV) :
    0≤(x.view.offset+stateOffset s h k v : Int) ∧
    (x.view.offset+stateOffset s h k v : Int)<x.view.capacity := by
  let s' : Shape := ⟨s.B,s.H,s.DK,s.DV,s.DV⟩
  have read := checked_dense_read s' current s.DV x ok hv h k v hh hkey hval
  have dense := (checked_tensor current _ _ x ok).2.2.1 rfl
  change x.view.strides=contiguous4 s.B s.H s.DK s.DV at dense
  rw [dense,source_head_stride s' s.DV hk hv] at read
  have eq : x.view.offset+h*(s.DK*s.DV)+k*s.DV+v=x.view.offset+stateOffset s h k v := by
    unfold stateOffset
    ring
  change 0≤x.view.offset+h*(s.DK*s.DV)+k*s.DV+v ∧ _ at read
  rw [eq] at read
  constructor
  · positivity
  · exact_mod_cast read.2

theorem checked_forward (s : Shape) (args : ForwardArguments) (current : Nat) (bases : Bases)
    (ok : forwardCheck s args current bases=Bool.true) : ForwardScope s args current := by
  simp only [forwardCheck,Bool.and_eq_true] at ok
  rcases ok with ⟨⟨⟨⟨⟨⟨shape,q⟩,k⟩,v⟩,seed⟩,addr⟩,windows⟩
  simp only [shapeCheck,decide_eq_true_eq] at shape
  have qmeta := checked_tensor current _ _ args.q q
  have kmeta := checked_tensor current _ _ args.k k
  have vmeta := checked_tensor current _ _ args.v v
  have seedok : ∀ x,args.seed=some x → tensorCheck current ![s.B,s.H,s.DK,s.DV] Bool.true x=Bool.true := by
    intro x hx
    simpa [hx] using seed
  refine ⟨shape.2.2.2,qmeta.1,kmeta.1,vmeta.1,qmeta.2.1,kmeta.2.1,vmeta.2.1,
    qmeta.2.2.1 rfl,kmeta.2.2.1 rfl,vmeta.2.2.1 rfl,?_,checked_bounds _ _ _ _ _ addr,?_,?_,?_,?_⟩
  · intro x hx
    have m := checked_tensor current _ _ x (seedok x hx)
    exact ⟨m.1,m.2.1,m.2.2.1 rfl⟩
  · intro h hh t ht c hc
    have read := checked_dense_read s current s.DK args.q q shape.2.2.2.1 h t c hh ht hc
    simp only [ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,InputBuffers.qAddress]
    exact ⟨by positivity,by exact_mod_cast read.2⟩
  · intro h hh t ht c hc
    have read := checked_dense_read s current s.DK args.k k shape.2.2.2.1 h t c hh ht hc
    have dense : args.q.view.strides=args.k.view.strides :=
      (qmeta.2.2.1 rfl).trans (kmeta.2.2.1 rfl).symm
    simp only [ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,InputBuffers.kAddress,dense]
    exact ⟨by positivity,by exact_mod_cast read.2⟩
  · intro h hh t ht c hc
    have read := checked_dense_read s current s.DV args.v v shape.2.2.2.2.1 h t c hh ht hc
    simp only [ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,InputBuffers.vAddress]
    exact ⟨by positivity,by exact_mod_cast read.2⟩
  · intro hs h hh key hk val hv
    cases he : args.seed with
    | none => simp [he] at hs
    | some x =>
      have read := checked_seed_read s current x (seedok x he) shape.2.2.2.1 shape.2.2.2.2.1 h key val hh hk hv
      simpa [ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,InputBuffers.seedAddress,he,detach] using read

theorem checked_backward (s : Shape) (ctx : SavedContext) (do_ : TensorObject) (final : Option TensorObject)
    (current : Nat) (bases : Bases) (gradEnabled : Bool)
    (ok : backwardCheck s ctx do_ final current bases gradEnabled=Bool.true) :
    BackwardScope s ctx do_ final current := by
  simp only [backwardCheck,Bool.and_eq_true] at ok
  rcases ok with ⟨⟨⟨⟨⟨fwd,grad⟩,dout⟩,fmeta⟩,addr⟩,windows⟩
  have dm := checked_tensor current _ _ do_ dout
  have finals : ∀ f,final=some f → ctx.outputFinal=Bool.true ∧
      tensorCheck current ![s.B,s.H,s.DK,s.DV] Bool.false f=Bool.true := by
    intro f hf
    simpa [hf] using fmeta
  refine ⟨dm.1,dm.2.1,?_,checked_bounds _ _ _ _ _ addr,?_,?_⟩
  · intro f hf
    have fm := finals f hf
    have m := checked_tensor current _ _ f fm.2
    exact ⟨fm.1,m.1,m.2.1⟩
  · intro h hh t ht v hv
    have hb := head_batch_bounds s h hh
    exact checked_view current _ _ do_ dout _ _ _ _ hb.1 hb.2 ht hv
  · intro f hf h hh k hk v hv
    have hb := head_batch_bounds s h hh
    exact checked_view current _ _ f (finals f hf).2 _ _ _ _ hb.1 hb.2 hk hv

theorem member_le_foldMax (xs : List Nat) (x : Nat) (hx : x∈xs) : x≤xs.foldr max 0 := by
  induction xs with
  | nil => simp at hx
  | cons a rest ih =>
    rcases List.mem_cons.mp hx with rfl | hr
    · exact Nat.le_max_left _ _
    · exact (ih hr).trans (Nat.le_max_right _ _)

theorem formed_radius (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (positive : 0<s.B ∧ 0<s.H) (iv ik h lk lv r : Nat)
    (hiv : iv<s.NV) (hik : ik<s.NK) (hh : h<s.BH) (hlk : lk<s.BK) (hlv : lv<s.BV) (hr : r ≤ s.T)
    (offset : Int) (mem : offset∈SourceAddress.formed phase seed extra s b iv ik h lk lv r) :
    offset.natAbs≤pointerRadius phase seed extra s b := by
  rw [←instantiate_formed phase seed extra s b iv ik h lk lv r] at mem
  obtain ⟨e,he,rfl⟩ := List.mem_map.mp mem
  exact (radius_bounds (limits s) _
    (env_bounds s iv ik h lk lv r positive.1 positive.2 hiv hik hh hlk hlv hr) e).trans
    (member_le_foldMax _ _ (List.mem_map.mpr ⟨e,he,rfl⟩))

theorem checked_window (base radius capacity : Nat) (ok : windowCheck base radius capacity=Bool.true)
    (offset : Int) (bound : offset.natAbs≤radius) :
    base%4=0 ∧ 0≤(base : Int)+4*offset ∧ (base : Int)+4*offset+4≤2^64 ∧ base+4*capacity≤2^64 := by
  simp only [windowCheck,decide_eq_true_eq] at ok
  have hp := Int.le_natAbs (a:=offset)
  have hn := Int.le_natAbs (a:=-offset)
  simp only [Int.natAbs_neg] at hn
  have hb : (offset.natAbs : Int)≤radius := Int.ofNat_le.mpr bound
  have low : (4 : Int)*radius≤base := by exact_mod_cast ok.2.1
  have high : (base : Int)+4*radius+4≤2^64 := by exact_mod_cast ok.2.2.1
  exact ⟨ok.1,by omega,by omega,ok.2.2.2⟩

theorem forward_native_pointer (s : Shape) (args : ForwardArguments) (current : Nat) (bases : Bases)
    (ok : forwardCheck s args current bases=Bool.true) (x : TensorObject) (member : x∈forwardObjects args)
    (iv ik h lk lv r : Nat) (hiv : iv<s.NV) (hik : ik<s.NK) (hh : h<s.BH)
    (hlk : lk<s.BK) (hlv : lv<s.BV) (hr : r ≤ s.T) (offset : Int)
    (formed : offset∈SourceAddress.formed .forward args.seed.isSome args.outputFinal s args.buffers iv ik h lk lv r) :
    (bases x.view.storage)%4=0 ∧ 0≤(bases x.view.storage : Int)+4*offset ∧
      (bases x.view.storage : Int)+4*offset+4≤2^64 ∧ bases x.view.storage+4*x.view.capacity≤2^64 := by
  simp only [forwardCheck,Bool.and_eq_true] at ok
  have shapes := ok.1.1.1.1.1.1
  simp only [shapeCheck,decide_eq_true_eq] at shapes
  have bound := formed_radius _ _ _ s args.buffers ⟨shapes.1,shapes.2.1⟩
    iv ik h lk lv r hiv hik hh hlk hlv hr offset formed
  exact checked_window _ _ _ (List.all_eq_true.mp ok.2 x member) offset bound

theorem backward_native_pointer (s : Shape) (ctx : SavedContext) (do_ : TensorObject)
    (final : Option TensorObject) (current : Nat) (bases : Bases) (gradEnabled : Bool)
    (ok : backwardCheck s ctx do_ final current bases gradEnabled=Bool.true)
    (x : TensorObject) (member : x∈backwardObjects ctx do_ final)
    (iv ik h lk lv r : Nat) (hiv : iv<s.NV) (hik : ik<s.NK) (hh : h<s.BH)
    (hlk : lk<s.BK) (hlv : lv<s.BV) (hr : r ≤ s.T) (offset : Int)
    (formed : offset∈SourceAddress.formed .backward ctx.initial.isSome final.isSome s
      (ctx.buffers do_.view (final.map TensorObject.view)) iv ik h lk lv r) :
    (bases x.view.storage)%4=0 ∧ 0≤(bases x.view.storage : Int)+4*offset ∧
      (bases x.view.storage : Int)+4*offset+4≤2^64 ∧ bases x.view.storage+4*x.view.capacity≤2^64 := by
  simp only [backwardCheck,Bool.and_eq_true] at ok
  have fwd := ok.1.1.1.1.1
  simp only [forwardCheck,Bool.and_eq_true] at fwd
  have shapes := fwd.1.1.1.1.1.1
  simp only [shapeCheck,decide_eq_true_eq] at shapes
  have bound := formed_radius _ _ _ s _ ⟨shapes.1,shapes.2.1⟩
    iv ik h lk lv r hiv hik hh hlk hlv hr offset formed
  exact checked_window _ _ _ (List.all_eq_true.mp ok.2 x member) offset bound

theorem allocated_native_pointer (bases : Bases) (radius : Nat) (objects : List (Nat × Nat))
    (ok : allocationCheck bases radius objects=Bool.true) (storage capacity : Nat)
    (member : (storage,capacity)∈objects) (offset : Int) (bound : offset.natAbs≤radius) :
    (bases storage)%4=0 ∧ 0≤(bases storage : Int)+4*offset ∧
      (bases storage : Int)+4*offset+4≤2^64 ∧ bases storage+4*capacity≤2^64 :=
  checked_window _ _ _ (List.all_eq_true.mp ok (storage,capacity) member) offset bound

#axiomsClean forward_native_pointer
#axiomsClean backward_native_pointer
#axiomsClean allocated_native_pointer
#axiomsClean formed_radius
#axiomsClean checked_window
#axiomsClean checked_view
#axiomsClean checked_forward
#axiomsClean checked_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission

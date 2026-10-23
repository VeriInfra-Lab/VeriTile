import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Native

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

def allocate (api : API Word Store Byte) (lib : Library api backings) (x : Shaped request) :
    Action (world api backings) (Output backings x) :=
  (call (.transfer .x) (lib.device x).call).bind fun device=>
  (call .residualDtype (lib.dtype x).call).bind fun dtype=>
  call (.allocate .y) (lib.zeros x device dtype).call

theorem allocate_success (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (s : State Store) (y : Output backings x)
    (input : Values api s x.tensor arangeWord)
    (ok : ((allocate api lib x).run s).result=.ok y) :
    Values api ((allocate api lib x).run s).state x.tensor arangeWord ∧
    Values api ((allocate api lib x).run s).state y.tensor (fun _=>0) := by
  unfold allocate at ok ⊢
  obtain ⟨device,hd,tail,hs,_⟩:=bind_success _ _ s y ok
  obtain ⟨dtype,ht,hz,hs2,_⟩:=bind_success _ _ _ y tail
  have hz':=call_ok (lib.zeros x device dtype) _ _ _ hz
  rw [hs,hs2]
  exact ⟨effect_values api (lib.zeros x device dtype) _ _ _
      (effect_values api (lib.dtype x) _ _ _ (effect_values api (lib.device x) s _ _ input)),
    lib.zeros_values _ _ _ _ _ _ hz'⟩

def launch (api : API Word Store Byte) (lib : Library api backings) (x : Shaped request)
    (y : Output backings x) (count : Number (x.rows*x.cols)) : Action (world api backings) Unit :=
  (call (.stride .x 0) (lib.stride0 x).call).bind fun stride0=>
  (call (.stride .x 1) (lib.stride1 x).call).bind fun stride1=>
  (call .selectKernel (lib.setup x y count stride0 stride1).call).bind fun trial=>
  trialAction api x y trial (lib.finish x y trial)

theorem launch_arguments (x : Shaped request) (count : Number (x.rows*x.cols))
    (stride0 : Number (max x.cols 1)) (stride1 : Number 1) (nonempty : count.actual≠0) :
    count.actual=x.rows*x.cols ∧ stride0.actual=x.cols ∧ stride1.actual=1 ∧ 0<x.cols := by
  have pos:0<x.rows*x.cols:=by rw [count.exact] at nonempty;omega
  have columns:0<x.cols:=Nat.pos_of_mul_pos_left pos
  exact ⟨count.exact,stride0.exact.trans (max_eq_left (by omega)),stride1.exact,columns⟩

theorem launch_success (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) (count : Number (x.rows*x.cols))
    (s : State Store) (positive : 0<x.rows*x.cols)
    (input : Values api s x.tensor arangeWord) (initial : Values api s y.tensor (fun _=>0))
    (ok : ((launch api lib x y count).run s).result=.ok ()) :
    Values api ((launch api lib x y count).run s).state y.tensor
      (target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0)) := by
  unfold launch at ok ⊢
  obtain ⟨stride0,h0,tail0,hs0,_⟩:=bind_success _ _ s () ok
  obtain ⟨stride1,h1,tail1,hs1,_⟩:=bind_success _ _ _ () tail0
  obtain ⟨trial,ht,hk,hs2,_⟩:=bind_success _ _ _ () tail1
  rw [hs0,hs1,hs2]
  exact trial_output api x y trial (lib.finish x y trial) _ positive
    (effect_values api (lib.setup x y count stride0 stride1) _ _ _
      (effect_values api (lib.stride1 x) _ _ _ (effect_values api (lib.stride0 x) s _ _ input)))
    (effect_values api (lib.setup x y count stride0 stride1) _ _ _
      (effect_values api (lib.stride1 x) _ _ _ (effect_values api (lib.stride0 x) s _ _ initial))) hk

def dispatch (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) : Action (world api backings) Unit :=
  (call .unpackShape (lib.numel x).call).bind fun count=>
  if count.actual=0 then Action.pure () else launch api lib x y count

theorem dispatch_success (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) (s : State Store)
    (input : Values api s x.tensor arangeWord) (initial : Values api s y.tensor (fun _=>0))
    (ok : ((dispatch api lib x y).run s).result=.ok ()) :
    Values api ((dispatch api lib x y).run s).state y.tensor
      (target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0)) := by
  unfold dispatch at ok ⊢
  obtain ⟨count,hn,tail,hs,_⟩:=bind_success _ _ s () ok
  rw [hs]
  by_cases empty:count.actual=0
  · simp only [empty,↓reduceIte,Action.pure]
    refine ⟨(lib.numel x).retained s _ initial.1,?_⟩
    intro i hi
    have zero:x.rows*x.cols=0:=count.exact.symm.trans empty
    simp [zero] at hi
  · simp only [empty,↓reduceIte] at tail ⊢
    apply launch_success api lib x y count _ (by rw [count.exact] at empty;omega)
      (effect_values api (lib.numel x) s _ _ input)
      (effect_values api (lib.numel x) s _ _ initial) tail

/-- The source returns Python None. The allocated tensor is observed through
its actual print call, and is never substituted for the return value. -/
def wrapper (api : API Word Store Byte) (lib : Library api backings) (request : Request) :
    Action (world api backings) Unit :=
  (input api lib request).bind fun x=>
  (allocate api lib x).bind fun y=>
  (dispatch api lib x y).bind fun _=>
  call .returnTuple (lib.print x y).call

theorem wrapper_success (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store)
    (ok : ((wrapper api lib request).run s).result=.ok ()) :
    ∃ (x : Shaped request) (y : Output backings x),
      ((allocate api lib x).run ((input api lib request).run s).state).result=.ok y ∧
      ((wrapper api lib request).run s).state.printed=some ⟨[x.rows,x.cols],y.tensor⟩ ∧
      Values api ((wrapper api lib request).run s).state y.tensor
        (target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0)) := by
  unfold wrapper at ok ⊢
  obtain ⟨x,hx,tail,hs,_⟩:=bind_success _ _ s () ok
  obtain ⟨y,hy,tail2,hs2,_⟩:=bind_success _ _ _ () tail
  obtain ⟨u,hd,hp,hs3,_⟩:=bind_success _ _ _ () tail2
  cases u
  have input:=input_success api lib request s x hx
  have allocated:=allocate_success api lib x _ y input hy
  have dispatched:=dispatch_success api lib x y _ allocated.1 allocated.2 hd
  refine ⟨x,y,hy,?_,?_⟩
  · rw [hs,hs2,hs3]
    exact lib.print_exact _ _ _ _ (call_ok (lib.print x y) _ _ _ hp)
  · rw [hs,hs2,hs3]
    exact effect_values api (lib.print x y) _ _ _ dispatched

/-- Complete owner-tagged caller backing bytes, for every success or error,
including errors before allocations and errors after completed native writes. -/
theorem wrapper_frame (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store) :
    Frame api backings s.memory ((wrapper api lib request).run s).state.memory :=
  (wrapper api lib request).frame s

theorem wrapper_error_state (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store) (code : Nat)
    (error : ((lib.arange request).call.run s).2=.error code) :
    ((wrapper api lib request).run s).state=((lib.arange request).call.run s).1 ∧
    ((wrapper api lib request).run s).result=.error (.api (.allocate .x) code) := by
  simp [wrapper,input,Action.bind,call,error,Except.mapError]

#axiomsClean allocate_success
#axiomsClean launch_arguments
#axiomsClean launch_success
#axiomsClean dispatch_success
#axiomsClean wrapper_success
#axiomsClean wrapper_frame
#axiomsClean wrapper_error_state
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Host

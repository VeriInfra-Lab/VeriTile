import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address (Dense)
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

/-- The ordinary reshape -1 rules, specialized only by the actual arange
length, imply the nonnegative success descriptor used by Library.reshape. -/
theorem reshape_nonnegative (r c : Int) (rows cols : Nat)
    (size : (rows*cols:Nat)=r*c)
    (row : r=-1 ∨ r=(rows:Int)) (col : c=-1 ∨ c=(cols:Int))
    (single : ¬(r=-1 ∧ c=-1))
    (inferRow : r=-1 → 0<cols) (inferCol : c=-1 → 0<rows) :
    r=(rows:Int) ∧ c=(cols:Int) := by
  rcases row with hr|hr
  · have hc:c=(cols:Int):=col.resolve_left (fun hc=>single ⟨hr,hc⟩)
    have pos:=inferRow hr
    have positive:(0:Int)<cols:=by exact_mod_cast pos
    have nonneg:(0:Int)≤rows*cols:=by positivity
    push_cast at size
    rw [hr,hc] at size
    nlinarith
  · rcases col with hc|hc
    · have pos:=inferCol hc
      have positive:(0:Int)<rows:=by exact_mod_cast pos
      have nonneg:(0:Int)≤rows*cols:=by positivity
      push_cast at size
      rw [hr,hc] at size
      nlinarith
    · exact ⟨hr,hc⟩

def Effect.failed (api : API Word Store Byte) (backings : List Backing) (Value : Type) (code : Nat) :
    Effect api backings Value where
  call:=⟨fun s=>(s,.error code),fun _ _ _=>rfl⟩
  retained:=fun _ _ h=>h
  oldRead:=fun _ _ _ _ _ _=>rfl

def Effect.returned (api : API Word Store Byte) (backings : List Backing) (value : Value) :
    Effect api backings Value where
  call:=⟨fun s=>(s,.ok value),fun _ _ _=>rfl⟩
  retained:=fun _ _ h=>h
  oldRead:=fun _ _ _ _ _ _=>rfl

def Effect.printed (api : API Word Store Byte) (backings : List Backing) (p : Packed shape) :
    Effect api backings Unit where
  call:=⟨fun s=>({s with printed:=some ⟨shape,p⟩},.ok ()),fun _ _ _=>rfl⟩
  retained:=fun _ _ h=>h
  oldRead:=fun _ _ _ _ _ _=>rfl

def Trial.completedAll (count cols : Nat) : Trial count cols :=
  ⟨Schedule.canonical count cols,(copies count cols).length,none,[],fun _=>le_rfl,CtaBarrierContract.canonical⟩

def Trial.failed (count cols completed code : Nat) : Trial count cols :=
  ⟨Schedule.canonical count cols,completed,some code,(copies count cols).take completed,by simp,CtaBarrierContract.canonical⟩

/-- An inhabited whole callback family, with fallible allocation/reshape and
successful ordinary metadata. No total successful allocation family is needed. -/
def Library.metadata (api : API Word Store Byte) (backings : List Backing) : Library api backings where
  arange:=fun _=>Effect.failed api backings _ 1
  reshape:=fun _ _=>Effect.failed api backings _ 2
  device:=fun x=>Effect.returned api backings ⟨x.tensor.owner,rfl⟩
  dtype:=fun _=>Effect.returned api backings .int32
  zeros:=fun _ _ _=>Effect.failed api backings _ 3
  numel:=fun x=>Effect.returned api backings ⟨x.rows*x.cols,rfl⟩
  stride0:=fun x=>Effect.returned api backings ⟨max x.cols 1,rfl⟩
  stride1:=fun _=>Effect.returned api backings ⟨1,rfl⟩
  setup:=fun x _ _ _ _=>Effect.returned api backings (Trial.completedAll (x.rows*x.cols) x.cols)
  finish:=fun _ _ _=>Effect.returned api backings ()
  print:=fun _ y=>Effect.printed api backings y.tensor
  arange_values:=by intros; contradiction
  reshape_live:=by intros; contradiction
  reshape_values:=by intros; contradiction
  reshape_storage:=by intros; contradiction
  zeros_values:=by intros; contradiction
  zeros_fresh:=by intros; contradiction
  print_exact:=by intros;rfl

theorem library_inhabited (api : API Word Store Byte) (backings : List Backing) :
    Nonempty (Library api backings) := ⟨Library.metadata api backings⟩

theorem any_integer_failure (api : API Word Store Byte) (backings : List Backing)
    (request : Request) (s : State Store) :
    (wrapper api (Library.metadata api backings) request).run s=
      ⟨s,.error (.api (.allocate .x) 1),[.allocate .x]⟩ := by
  simp [wrapper,input,Library.metadata,Effect.failed,Action.bind,call,Except.mapError]

theorem empty_dispatch_exact (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) (s : State Store)
    (number : Number (x.rows*x.cols))
    (ok : ((lib.numel x).call.run s).2=.ok number) (empty : number.actual=0) :
    (dispatch api lib x y).run s=⟨((lib.numel x).call.run s).1,.ok (),[.unpackShape]⟩ := by
  simp [dispatch,Action.bind,call,ok,empty,Except.mapError,Action.pure]

theorem native_success_inhabited (api : API Word Store Byte) (x : Shaped request)
    (y : Output backings x) (s : State Store) :
    ((trialAction api x y (Trial.completedAll (x.rows*x.cols) x.cols)
      (Effect.returned api backings ())).run s).result=.ok () := by
  simp [trialAction,Trial.completedAll,Effect.returned,call,Except.mapError]

theorem native_failure_exact (api : API Word Store Byte) (x : Shaped request)
    (y : Output backings x) (completed code : Nat) (s : State Store) :
    (trialAction api x y (Trial.failed (x.rows*x.cols) x.cols completed code)
      (Effect.returned api backings ())).run s=
      ⟨trialState api x y (Trial.failed (x.rows*x.cols) x.cols completed code) s,
       .error (.api .execute code),[.execute,.execute]⟩ := by
  simp [trialAction,Trial.failed,Effect.returned,call,Except.mapError]

/-- A native fault can complete an arbitrary subset/order of lanes of its
current vector store. This constructor records those actual completed words;
it requires no canonical lane-prefix assumption. -/
def Trial.failedActual (count cols code : Nat) (writes : List (Copy count)) : Trial count cols :=
  ⟨Schedule.canonical count cols,0,some code,writes,by simp,CtaBarrierContract.canonical⟩

theorem arbitrary_native_failure (api : API Word Store Byte) (x : Shaped request)
    (y : Output backings x) (code : Nat) (writes : List (Copy (x.rows*x.cols))) (s : State Store) :
    ((trialAction api x y (Trial.failedActual (x.rows*x.cols) x.cols code writes)
      (Effect.returned api backings ())).run s).state.memory=
      writeOwner s.memory y.tensor.owner (execute api 0 y.placement writes (s.memory y.tensor.owner)) := by
  simp [trialAction,trialState,Trial.failedActual,Trial.writes,Effect.returned,call]

theorem setup_failure_exact (api : API Word Store Byte)
    (setup : Effect api backings Value) (next : Value → Action (world api backings) Unit)
    (s : State Store) (code : Nat) (error : (setup.call.run s).2=.error code) :
    ((call .selectKernel setup.call).bind next).run s=
      ⟨(setup.call.run s).1,.error (.api .selectKernel code),[.selectKernel]⟩ := by
  simp [Action.bind,call,error,Except.mapError]

theorem print_failure_preserves (api : API Word Store Byte) (lib : Library api backings)
    (x : Shaped request) (y : Output backings x) (s : State Store) (f : Nat → Word)
    (values : Values api s y.tensor f) : Values api ((lib.print x y).call.run s).1 y.tensor f :=
  effect_values api (lib.print x y) s y.tensor f values

/-- Metadata-only descriptors witness small, empty, one-column and large
index cases. Their explicit inequalities are examples, not a public cap. -/
def witnessInput (rows cols : Nat) (fits : 2*(rows*cols)*4≤2^64) : Shaped ⟨rows,cols⟩ where
  rows:=rows
  cols:=cols
  rows_eq:=rfl
  cols_eq:=rfl
  tensor:=⟨1,0,⟨0,by simp;omega,rfl⟩⟩

def witnessOutput (rows cols : Nat) (fits : 2*(rows*cols)*4≤2^64) :
    Output [] (witnessInput rows cols fits) where
  tensor:=⟨2,0,⟨rows*cols*4,by simp [witnessInput];omega,by simp⟩⟩
  owner:=rfl
  apart:=by simp [VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution.Disjoint,witnessInput]
  callerApart:=by intro key hk;obtain ⟨b,hb,_⟩:=hk;simp at hb

def smallInput:=witnessInput 1 1 (by norm_num)
def smallOutput:=witnessOutput 1 1 (by norm_num)
def oneColumnInput:=witnessInput 49 1 (by norm_num)
def oneColumnOutput:=witnessOutput 49 1 (by norm_num)
def wideEmptyInput:=witnessInput 0 (2^32) (by norm_num)
def wideEmptyOutput:=witnessOutput 0 (2^32) (by norm_num)
def largeInput:=witnessInput 1 (2^32) (by norm_num)
def largeOutput:=witnessOutput 1 (2^32) (by norm_num)

theorem small_native (api : API Word Store Byte) (s : State Store) :
    ((trialAction api smallInput smallOutput (Trial.completedAll 1 1)
      (Effect.returned api [] ())).run s).result=.ok () := native_success_inhabited api _ _ s

theorem one_column_native (api : API Word Store Byte) (s : State Store) :
    ((trialAction api oneColumnInput oneColumnOutput (Trial.completedAll 49 1)
      (Effect.returned api [] ())).run s).result=.ok () := native_success_inhabited api _ _ s

theorem large_native (api : API Word Store Byte) (s : State Store) :
    ((trialAction api largeInput largeOutput (Trial.completedAll (2^32) (2^32))
      (Effect.returned api [] ())).run s).result=.ok () := native_success_inhabited api _ _ s

theorem wide_empty_skip (api : API Word Store Byte) (s : State Store) :
    (dispatch api (Library.metadata api []) wideEmptyInput wideEmptyOutput).run s=
      ⟨s,.ok (),[.unpackShape]⟩ :=
  empty_dispatch_exact api _ _ _ s ⟨0,rfl⟩ rfl rfl

#axiomsClean reshape_nonnegative
#axiomsClean library_inhabited
#axiomsClean any_integer_failure
#axiomsClean empty_dispatch_exact
#axiomsClean native_success_inhabited
#axiomsClean native_failure_exact
#axiomsClean arbitrary_native_failure
#axiomsClean setup_failure_exact
#axiomsClean print_failure_preserves
#axiomsClean small_native
#axiomsClean one_column_native
#axiomsClean large_native
#axiomsClean wide_empty_skip
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Host

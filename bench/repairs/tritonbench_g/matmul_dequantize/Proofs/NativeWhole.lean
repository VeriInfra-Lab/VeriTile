import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.TensorBinding
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.GPUFPEquiv
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.NativeWhole
open VeriTile.Triton FP.Structural Host Checked ByteBridge
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

/-- Independent instruction-graph result from original input words, followed
by the ordinary Tensor store ABI. It never reads the executed output state. -/
def expected (out : Host.Tensor) (g : Geometry.Layout) (width : Nat) (plan : Native.Plan)
    (isa : Native.WordISA) (before : Memory) (reference : Grid.Index g → Native.Expr) : Memory := fun r b =>
  if hr : r=out.region then
    if hx : ∃ x : Store g width,address out g width x=b then
      let x := Classical.choose hx
      ((Native.lower plan (reference x.1) x.1.val.1).eval isa).extractLsb' (8*x.2.val) 8
    else before r b
  else before r b

def observe (a : Arguments) (route : Route) (t : TensorBinding.Tensors a route)
    (g : Geometry.Layout) (width : Nat) (half : Bool) (plan : Native.Plan) (isa : Native.WordISA)
    (before : Memory) (state : State Native.Expr) : TensorBinding.Result where
  memory := serialize (TensorBinding.output a route) g width half plan isa before state
  output := t.get (TensorBinding.output a route).region
  returned := TensorBinding.returned a route t
  allocated := TensorBinding.allocated a route
  scratchReleased := decide (route=.s1)

def invoke (a : Arguments) (route : Route) (t : TensorBinding.Tensors a route)
    (g : Geometry.Layout) (width : Nat) (half : Bool) (plan : Native.Plan) (isa : Native.WordISA)
    (before : Memory) (action : Option (State Native.Expr)) : Except Error (Option TensorBinding.Result) := do
  checkInputs a route
  pure (action.map (observe a route t g width half plan isa before))

def Guarantee (a : Arguments) (route : Route) (cfg : FirstConfig) (t : TensorBinding.Tensors a route)
    (g : Geometry.Layout) (width : Nat) (plan : Native.Plan) (target : Native.WordISA)
    (before : Memory) (reference : Grid.Index g → Native.Expr) (result : TensorBinding.Result) : Prop :=
  result.output=t.get (TensorBinding.output a route).region ∧
  result.returned=TensorBinding.returned a route t ∧ result.allocated=TensorBinding.allocated a route ∧
  result.scratchReleased=decide (route=.s1) ∧
  TensorBinding.Binding (TensorBinding.output a route) result.output ∧
  ABI.SourceExact a route cfg ∧
  result.memory=expected (TensorBinding.output a route) g width plan target before reference ∧
  (∀ x : Store g width,result.memory (TensorBinding.output a route).region
    (address (TensorBinding.output a route) g width x)=
      ((Native.lower plan (reference x.1) x.1.val.1).eval target).extractLsb' (8*x.2.val) 8) ∧
  (∀ r b,Outside (TensorBinding.output a route) g width r b → result.memory r b=before r b)

/-- Internal composition lemma; every source success/value premise is supplied
by the existing whole-grid proofs in the route theorems below. -/
theorem complete (a : Arguments) (route : Route) (cfg : FirstConfig) (t : TensorBinding.Tensors a route)
    (h : InputContract a route) (candidate : cfg∈candidates route)
    (g : Geometry.Layout) (hg : Geometry.Valid g) (width : Nat) (hw : 0<width) (half : Bool)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target)
    (before : Memory) (reference : Grid.Index g → Native.Expr) (action : Option (State Native.Expr))
    (state : State Native.Expr) (run : action=some state)
    (values : ∀ x : Grid.Index g,state.mem (TensorBinding.output a route).region (Grid.address g x)=floatCell half (reference x)) :
    ∃ result,invoke a route t g width half plan actual before action=.ok (some result) ∧
      Guarantee a route cfg t g width plan target before reference result := by
  have eq := numerics_eq actual target ports
  subst actual
  refine ⟨observe a route t g width half plan target before state,?_,rfl,rfl,rfl,rfl,
    TensorBinding.output_bound a route t,ABI.source_exact a route cfg (indices a route cfg candidate h),?_,?_,?_⟩
  · simp [invoke,(checkInputs_iff a route).mpr h,run]
  · funext r b
    simp only [observe,serialize,expected]
    split
    · split
      · rename_i hx
        rw [values (Classical.choose hx).1,float_read_cell]
      · rfl
    · rfl
  · intro x
    rw [show (observe a route t g width half plan target before state).memory=
      serialize (TensorBinding.output a route) g width half plan target before state from rfl]
    rw [serialize_read _ g hg width hw half plan target before state x,values x.1,float_read_cell]
  · exact serialize_frame (TensorBinding.output a route) g width half plan target before state

def gptqAction (a : Arguments) (cfg : FirstConfig) (h : InputContract a .gptq) (m : Memory) :=
  let ha := admitted a .gptq h
  Grid.run (FirstGrid.program Native.scalar Native.firstDot (first a cfg) (first_valid a cfg ha)
    (distinct a ha).1 (distinct a ha).2.1 (distinct a ha).2.2.1 (distinct a ha).2.2.2.1)
    (List.range (Geometry.count (FirstGrid.geometry (first a cfg)))) (entry a m)

def gptqReference (a : Arguments) (cfg : FirstConfig) (m : Memory)
    (x : Grid.Index (FirstGrid.geometry (first a cfg))) : Native.Expr :=
  VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.outputValue Native.scalar Native.firstDot
    (first a cfg) (GridExecution.atProgram (entry a m) x.val.1) x.val.2

theorem gptq (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq)
    (h : InputContract a .gptq) (plan : Native.Plan) (actual target : Native.WordISA)
    (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .gptq t (FirstGrid.geometry (first a cfg)) 2 Bool.true plan actual m (gptqAction a cfg h m)=.ok (some result) ∧
      Guarantee a .gptq cfg t (FirstGrid.geometry (first a cfg)) 2 plan target m (gptqReference a cfg m) result := by
  have ha := admitted a .gptq h
  have candidate : cfg∈candidates .gptq := by simp [candidates,first_exhaustive]
  obtain ⟨_,state,run,values,_,_,_⟩ := FPEquiv.gptq a cfg ha (indices a .gptq cfg candidate h)
    (entry a m) (packed a .gptq ha m) (input_words a .gptq ha m) plan actual
    (List.range (Geometry.count (FirstGrid.geometry (first a cfg)))) (List.Perm.refl _)
  apply complete a .gptq cfg t h candidate _ (first_valid a cfg ha) 2 (by decide) Bool.true
    plan actual target ports m (gptqReference a cfg m) (gptqAction a cfg h m) state run
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

def s2Action (a : Arguments) (h : InputContract a .s2) (m : Memory) :=
  let ha := admitted a .s2 h
  Grid.run (SecondGrid.program Native.scalar Native.secondDot (second a) (second_valid a ha)
    (distinct a ha).1 (distinct a ha).2.1 (distinct a ha).2.2.1 (distinct a ha).2.2.2.1)
    (List.range (Geometry.count (SecondGrid.geometry (second a)))) (entry a m)

def s2Reference (a : Arguments) (m : Memory) (x : Grid.Index (SecondGrid.geometry (second a))) : Native.Expr :=
  FPTypedSecond.outputValue Native.scalar Native.secondDot (second a) (GridExecution.atProgram (entry a m) x.val.1) x.val.2

theorem s2 (a : Arguments) (t : TensorBinding.Tensors a .s2) (h : InputContract a .s2)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .s2 t (SecondGrid.geometry (second a)) 2 Bool.true plan actual m (s2Action a h m)=.ok (some result) ∧
      Guarantee a .s2 .m64n256 t (SecondGrid.geometry (second a)) 2 plan target m (s2Reference a m) result := by
  have ha := admitted a .s2 h
  have candidate : FirstConfig.m64n256∈candidates .s2 := by simp [candidates]
  obtain ⟨_,state,run,values,_,_,_⟩ := FPEquiv.s2 a ha (indices a .s2 .m64n256 candidate h)
    (entry a m) (packed a .s2 ha m) (input_words a .s2 ha m) plan actual
    (List.range (Geometry.count (SecondGrid.geometry (second a)))) (List.Perm.refl _)
  apply complete a .s2 .m64n256 t h candidate _ (second_valid a ha) 2 (by decide) Bool.true
    plan actual target ports m (s2Reference a m) (s2Action a h m) state run
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

def dequant32Action (a : Arguments) (h : InputContract a .dequant32) (m : Memory) :=
  let ha := admitted a .dequant32 h
  Grid.run (DequantGrid32.program Native.scalar (Host.dequant32 a) (d32_valid a ha)
    (distinct a ha).2.2.2.2.1 (distinct a ha).2.2.2.2.2.1 (distinct a ha).2.2.2.2.2.2)
    (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a)))) (entry a m)

def dequant32Reference (a : Arguments) (m : Memory) (x : Grid.Index (DequantGrid32.geometry (Host.dequant32 a))) : Native.Expr :=
  Dequant32.value Native.scalar (Host.dequant32 a) (DequantGrid32.atProgram (Host.dequant32 a) (entry a m) x.val.1) x.val.2

theorem dequant32 (a : Arguments) (t : TensorBinding.Tensors a .dequant32) (h : InputContract a .dequant32)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .dequant32 t (DequantGrid32.geometry (Host.dequant32 a)) 4 Bool.false plan actual m (dequant32Action a h m)=.ok (some result) ∧
      Guarantee a .dequant32 .m64n256 t (DequantGrid32.geometry (Host.dequant32 a)) 4 plan target m (dequant32Reference a m) result := by
  have ha := admitted a .dequant32 h
  have candidate : FirstConfig.m64n256∈candidates .dequant32 := by simp [candidates]
  obtain ⟨_,state,run,values,_,_,_⟩ := FPEquiv.dequant32 a ha (indices a .dequant32 .m64n256 candidate h)
    (entry a m) (packed a .dequant32 ha m) (input_words a .dequant32 ha m) plan actual
    (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a)))) (List.Perm.refl _)
  apply complete a .dequant32 .m64n256 t h candidate _ (d32_valid a ha) 4 (by decide) Bool.false
    plan actual target ports m (dequant32Reference a m) (dequant32Action a h m) state run
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

def dequant16Action (a : Arguments) (h : InputContract a .dequant16) (m : Memory) :=
  let ha := admitted a .dequant16 h
  Grid.run (DequantGrid16.program Native.scalar (Host.dequant16 a) (d16_valid a ha)
    (distinct a ha).2.2.2.2.1 (distinct a ha).2.2.2.2.2.1 (distinct a ha).2.2.2.2.2.2)
    (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (entry a m)

def dequant16Reference (a : Arguments) (m : Memory) (x : Grid.Index (DequantGrid16.geometry (Host.dequant16 a))) : Native.Expr :=
  VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar (Host.dequant16 a)
    (DequantGrid16.atProgram (Host.dequant16 a) (entry a m) x.val.1) x.val.2

theorem dequant16 (a : Arguments) (t : TensorBinding.Tensors a .dequant16) (h : InputContract a .dequant16)
    (plan : Native.Plan) (actual target : Native.WordISA) (ports : Numerics actual target) (m : Memory) :
    ∃ result,invoke a .dequant16 t (DequantGrid16.geometry (Host.dequant16 a)) 2 Bool.true plan actual m (dequant16Action a h m)=.ok (some result) ∧
      Guarantee a .dequant16 .m64n256 t (DequantGrid16.geometry (Host.dequant16 a)) 2 plan target m (dequant16Reference a m) result := by
  have ha := admitted a .dequant16 h
  have candidate : FirstConfig.m64n256∈candidates .dequant16 := by simp [candidates]
  obtain ⟨_,state,run,values,_,_,_⟩ := FPEquiv.dequant16 a .dequant16 ha (Or.inl rfl)
    (indices a .dequant16 .m64n256 candidate h) (entry a m) (packed a .dequant16 ha m)
    (input_words a .dequant16 ha m) plan actual
    (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (List.Perm.refl _)
  apply complete a .dequant16 .m64n256 t h candidate _ (d16_valid a ha) 2 (by decide) Bool.true
    plan actual target ports m (dequant16Reference a m) (dequant16Action a h m) state run
  intro x
  exact values x.val.1 x.property.1 x.val.2 x.property.2

#specNonCircular Guarantee avoiding [invoke,observe,serialize,Grid.run]
#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant32
#axiomsClean dequant16
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.NativeWhole

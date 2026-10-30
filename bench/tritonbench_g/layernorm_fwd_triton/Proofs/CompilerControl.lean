import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Compiler

/-! The selected object's decoded loop, mask and address recipes are consumed
here, then normalized by local syntactic equalities. No bytes-to-result or
whole-output equality is supplied by the compiler adapter. The interpretation
is for the certified repaired source graph; unsupported graph recipes cannot
inhabit `Selected`. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace CompilerControl
open Address Memory Execution Native Compiler
open VeriTile.Bench.Repairs.LayerNormFwd.Address
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
open scoped Classical

def evalWord (env : IntReg → Nat) : IntExpr → Nat
  | .widened r => wrap64 (env r)
  | .add a b => add64 (evalWord env a) (evalWord env b)
  | .mul a b => mul64 (evalWord env a) (evalWord env b)

def intEnvironment (st : Strides) (row : Row shape) (col : Nat) : IntReg → Nat
  | .seq => row.1.val | .head => row.2.val | .off => col | .lane => 0 | .n => shape.columns
  | .xSeq => st.xSeq | .xHead => st.xHead | .xDim => st.xDim
  | .ySeq => st.ySeq | .yHead => st.yHead | .yDim => st.yDim
  | .wHead => st.wHead | .wDim => st.wDim

def pointer (a : Address.Allocation) (env : IntReg → Nat) (expr : IntExpr) : Nat :=
  add64 (wrap64 a.base) (mul64 (evalWord env expr) a.width)

theorem x_pointer (code : Selected decoder shape.columns) (layout : Layout shape st)
    (row : Row shape) (col : Fin shape.columns) :
    pointer layout.x (intEnvironment st row col.val) (decoder.inspect code.object).xAddress=
      xAddress layout row col := by
  rw [code.recipes.xRecipe]
  have hh : pointer layout.x (intEnvironment st row col.val) (addressExpr .xSeq .xHead .xDim)=
      nativePointer layout.x row.1.val row.2.val col.val st.xSeq st.xHead st.xDim := by
    simp [pointer,evalWord,intEnvironment,addressExpr,columnExpr,nativePointer,wordOffset,
      add64,mul64,wrap64,Nat.add_mod,Nat.mul_mod]
  exact hh.trans (native_x layout row col)

theorem w_pointer (code : Selected decoder shape.columns) (layout : Layout shape st)
    (row : Row shape) (col : Fin shape.columns) :
    pointer layout.w (intEnvironment st row col.val) (decoder.inspect code.object).wAddress=
      wAddress layout row col := by
  rw [code.recipes.wRecipe]
  have hh : pointer layout.w (intEnvironment st row col.val) weightAddressExpr=
      nativePointer layout.w 0 row.2.val col.val 0 st.wHead st.wDim := by
    simp [pointer,evalWord,intEnvironment,weightAddressExpr,columnExpr,nativePointer,wordOffset,
      add64,mul64,wrap64,Nat.add_mod,Nat.mul_mod]
  exact hh.trans (native_w layout row col)

theorem y_pointer (code : Selected decoder shape.columns) (layout : Layout shape st)
    (row : Row shape) (col : Fin shape.columns) :
    pointer layout.y (intEnvironment st row col.val) (decoder.inspect code.object).yAddress=
      yAddress layout row col := by
  rw [code.recipes.yRecipe]
  have hh : pointer layout.y (intEnvironment st row col.val) (addressExpr .ySeq .yHead .yDim)=
      nativePointer layout.y row.1.val row.2.val col.val st.ySeq st.yHead st.yDim := by
    simp [pointer,evalWord,intEnvironment,addressExpr,columnExpr,nativePointer,wordOffset,
      add64,mul64,wrap64,Nat.add_mod,Nat.mul_mod]
  exact hh.trans (native_y layout row col)

/-- Actual decoded start, increment and lane count. Every inactive lane is
filtered before becoming a physical memory event. -/
def decodedColumns (N off step lanes : Nat) : List (Fin N) :=
  if h : off<N ∧ 0<step then
    ((List.finRange lanes).filterMap fun lane =>
      if hc : off+lane.val<N then some ⟨off+lane.val,hc⟩ else none) ++ decodedColumns N (off+step) step lanes
  else []
termination_by N-off
decreasing_by omega

theorem fixed_columns (N off : Nat) : decodedColumns N off 128 128=columns N off := by
  rw [decodedColumns,columns]
  by_cases h : off<N
  · rw [dif_pos ⟨h,by decide⟩,dif_pos h]
    rw [fixed_columns N (off+128)]
  · rw [dif_neg (by omega),dif_neg h]
termination_by N-off
decreasing_by omega

def decodedMean (ops : Ops α) (code : Selected decoder N) (x : Nat → α) : α :=
  let g := decoder.inspect code.object
  let acc := foldRange (fun off acc lane =>
    let value := if g.meanMasked then maskedX ops x N (off+lane.val) else cast32 (x (off+lane.val))
    eval ops (environment ops (acc lane) value ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero)
      g.meanUpdate) g.loopStart N g.loopStep (fun _ : Fin 128 => ops.zero)
  eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero
    (g.meanTree.eval ops.add acc) (ops.count N) ops.zero) g.meanDivide

def decodedVariance (ops : Ops α) (code : Selected decoder N) (x : Nat → α) (mu : α) : α :=
  let g := decoder.inspect code.object
  let acc := foldRange (fun off acc lane =>
    let value := if g.varianceMasked then maskedX ops x N (off+lane.val) else cast32 (x (off+lane.val))
    let z := eval ops (environment ops ops.zero value mu ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero) g.center
    let z := if g.centerMasked then (if off+lane.val<N then z else ops.zero) else z
    eval ops (environment ops (acc lane) ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero z)
      g.varianceUpdate) g.loopStart N g.loopStep (fun _ : Fin 128 => ops.zero)
  eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero
    (g.varianceTree.eval ops.add acc) (ops.count N) ops.zero) g.varianceDivide

theorem decoded_mean (ops : Ops α) (code : Selected decoder N) (x : Nat → α) :
    decodedMean ops code x=compiledMean ops code x := by
  unfold decodedMean compiledMean
  dsimp only
  rw [code.recipes.start,code.recipes.step,code.recipes.masks.1]
  rfl

theorem decoded_variance (ops : Ops α) (code : Selected decoder N) (x : Nat → α) (mu : α) :
    decodedVariance ops code x mu=compiledVariance ops code x mu := by
  unfold decodedVariance compiledVariance
  dsimp only
  rw [code.recipes.start,code.recipes.step,code.recipes.masks.2.1,code.recipes.masks.2.2.2]
  rfl

def snapshot (ops : Ops α) (code : Selected decoder shape.columns) (api : API α Store Byte)
    (layout : Layout shape st) (m : Store) (row : Row shape) (col : Nat) : α :=
  if h : col<shape.columns then api.read m
    (pointer layout.x (intEnvironment st row col) (decoder.inspect code.object).xAddress) else ops.zero

theorem snapshot_eq (ops : Ops α) (code : Selected decoder shape.columns) (api : API α Store Byte)
    (layout : Layout shape st) (m : Store) (row : Row shape) :
    snapshot ops code api layout m row=xSnapshot ops api layout m row := by
  funext col
  by_cases h : col<shape.columns
  · simpa only [snapshot,xSnapshot,dif_pos h] using congrArg (api.read m) (x_pointer code layout row ⟨col,h⟩)
  · simp [snapshot,xSnapshot,h]

def execute (ops : Ops α) (code : Selected decoder shape.columns)
    (api : API α Store Byte) (layout : Layout shape st) (eps : α)
    (rows : List (Row shape)) (m : Store) : Store :=
  let g := decoder.inspect code.object
  if g.passCount=3 then
    rows.foldl (fun state row =>
      let x := snapshot ops code api layout state row
      let mu := decodedMean ops code x
      let v := decodedVariance ops code x mu
      let inv := eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero eps v ops.zero ops.zero) g.inverse
      let cols := if g.outputMasked then decodedColumns shape.columns g.loopStart g.loopStep g.laneCount else []
      cols.foldl (fun current col =>
        let env := intEnvironment st row col.val
        api.write current (pointer layout.y env g.yAddress)
          (eval ops (environment ops ops.zero (api.read current (pointer layout.x env g.xAddress)) mu inv
            (api.read current (pointer layout.w env g.wAddress)) ops.zero ops.zero ops.zero ops.zero) g.output)) state) m
  else m

theorem execute_selected (ops : Ops α) (code : Selected decoder shape.columns)
    (api : API α Store Byte) (layout : Layout shape st) (eps : α)
    (rows : List (Row shape)) (m : Store) :
    execute ops code api layout eps rows m=Compiler.execute ops code api layout eps rows m := by
  unfold execute
  dsimp only
  simp only [code.recipes.passes,ite_true,code.recipes.masks.2.2.1,code.recipes.start,
    code.recipes.step,code.recipes.lanes,fixed_columns]
  simp_rw [snapshot_eq,decoded_mean,decoded_variance,x_pointer,w_pointer,y_pointer]
  rfl

#axiomsClean x_pointer
#axiomsClean w_pointer
#axiomsClean y_pointer
#axiomsClean fixed_columns
#axiomsClean execute_selected
#guard_msgs (drop info) in
#auditModuleAxioms
end
end CompilerControl
end VeriTile.Bench.Repairs.LayernormFwdTriton

import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Native
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.FP
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def logicalScalar (op : Operations α) (r : Request) (input : Bool→Nat→Nat→Nat→Nat→α)
    (cosine sine : Nat→Nat→α) (key : Bool) (b h time col : Nat) : α :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    scalarPair op (decide (r.half≤col)) (input key b h time j) (input key b h time (j+r.half))
      (cosine time j) (sine time j)
  else input key b h time col

def mathematicalTarget (api : API β Store Byte) (decode : β→ℝ) (r : Request)
    (before : Store) (input : Bool→Native.Tensor) (cosine sine : Native.Tensor)
    (key : Bool) (b h time col : Nat) :=
  Spec.result r (fun k b h t c => decode (Native.logicalRead api before (input k) b h t c))
    (fun t j => decode (api.read before (Native.tableAddress cosine t j)))
    (fun t j => decode (api.read before (Native.tableAddress sine t j))) key b h time col

theorem row_decompose (b time sequence : Nat) (ht : time<sequence) :
    (b*sequence+time)/sequence=b ∧ (b*sequence+time)%sequence=time := by
  constructor
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt ht]; simp
  · simp [Nat.add_mod,Nat.mod_eq_of_lt ht]

theorem scalar_binding {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m)
    (key : Bool) (b h time col : Nat) (hb : b<r.batches) (hh : h<r.heads key)
    (ht : time<r.sequence) (hc : col<r.dimension) :
    scalarResult op r m key (b*r.sequence+time) h col=
      logicalScalar op r (fun k b h t c => embed (Native.logicalRead api call.before (call.input k) b h t c))
        (fun t j => embed (api.read call.before (Native.tableAddress call.cosine t j)))
        (fun t j => embed (api.read call.before (Native.tableAddress call.sine t j))) key b h time col := by
  have ha := row_bound r b time hb ht
  have hd := row_decompose b time r.sequence ht
  have input (c : Nat) (hc : c<r.dimension) :
      (m (region key) (r.offset key (b*r.sequence+time) h c)).read .fp32=
        embed (Native.logicalRead api call.before (call.input key) b h time c) := by
    rw [call.inputBinding key _ h c ha hh hc,hd.1,hd.2]
    unfold Native.logicalRead
    rw [call.inputAddress key b h time c hb hh ht hc]
  unfold scalarResult logicalScalar
  split
  · rename_i active
    let j := if col<r.half then col else col-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    have h0 := half_column r Bool.false j hj
    have h1 := half_column r Bool.true j hj
    simp only [Bool.false_eq_true,↓reduceIte,Nat.add_zero] at h0 h1
    dsimp only
    rw [input j h0,input (j+r.half) h1,hd.2,
      call.cosineBinding time j ht hj,call.sineBinding time j ht hj]
  · exact input col hc

theorem ideal_logical (r : Request) (input : Bool→Nat→Nat→Nat→Nat→ℝ)
    (cosine sine : Nat→Nat→ℝ) (key : Bool) (b h time col : Nat) :
    logicalScalar ideal r input cosine sine key b h time col=Spec.result r input cosine sine key b h time col := by
  unfold logicalScalar Spec.result
  split
  · dsimp only
    rw [ideal_pair]
    by_cases h : col<r.half <;> simp [h,show (r.half≤col)↔¬col<r.half from by omega]
  · rfl

def LocalNodes (api : API Numerics.Word Store Byte) (p : Numerics.Primitives) (r : Request)
    (before : Store) (input : Bool→Native.Tensor) (cosine sine : Native.Tensor) : Prop :=
  ∀ key b h time col,b<r.batches → h<r.heads key → time<r.sequence → col<2*r.half →
    let j := if col<r.half then col else col-r.half
    Numerics.LocalExact p (decide (r.half≤col))
      (Native.logicalRead api before (input key) b h time j)
      (Native.logicalRead api before (input key) b h time (j+r.half))
      (api.read before (Native.tableAddress cosine time j))
      (api.read before (Native.tableAddress sine time j))

theorem fp_logical (api : API Numerics.Word Store Byte) (p : Numerics.Primitives) (r : Request)
    (before : Store) (input : Bool→Native.Tensor) (cosine sine : Native.Tensor)
    (nodes : LocalNodes api p r before input cosine sine)
    (key : Bool) (b h time col : Nat) (hb : b<r.batches) (hh : h<r.heads key) (ht : time<r.sequence) :
    Numerics.decode (logicalScalar (Numerics.operations p) r
      (fun k b h t c => Numerics.leaf (Native.logicalRead api before (input k) b h t c))
      (fun t j => Numerics.leaf (api.read before (Native.tableAddress cosine t j)))
      (fun t j => Numerics.leaf (api.read before (Native.tableAddress sine t j))) key b h time col).word=
      mathematicalTarget api Numerics.decode r before input cosine sine key b h time col := by
  unfold logicalScalar mathematicalTarget Spec.result
  split
  · rename_i active
    dsimp only
    rw [Numerics.source_pair]
    dsimp only [Numerics.leaf]
    rw [Numerics.pair_decode p _ _ _ _ _ (nodes key b h time col hb hh ht active)]
    by_cases h : col<r.half <;> simp [h,show (r.half≤col)↔¬col<r.half from by omega]
  · rfl

#specNonCircular mathematicalTarget avoiding [Source.kernel,scalarPair,scalarResult,Native.entry,Memory.run]
#specNonCircular LocalNodes avoiding [Source.kernel,scalarPair,scalarResult,Native.entry,Memory.run,mathematicalTarget]
#axiomsClean scalar_binding
#axiomsClean ideal_logical
#axiomsClean fp_logical
end
end VeriTile.Bench.Repairs.RopeBackwardTransform

import bench.repairs.tritonbench_g.rope_embedding.Proofs.Complete
import bench.repairs.tritonbench_g.rope_embedding.Proofs.FP
namespace VeriTile.Bench.Repairs.RopeEmbedding.Checked
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def logicalTarget (op : Operations α) (r : Request) (back : Bool)
    (input : Nat→Nat→Nat→Nat→α) (cosine sine : Nat→Nat→α) (b h time col : Nat) : α :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    scalarPair op back (input b h time j) (input b h time (j+r.half))
      (cosine time j) (sine time j) (decide (r.half≤col))
  else input b h time col

def tableAddress (r : Request) (t : Native.Tensor) (time j : Nat) := t.base+4*(time*r.half+j)

theorem scalar_binding {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Native.Preparation api embed r s)
    (b h time col : Nat) (hb : b<r.batches) (hh : h<r.heads) (ht : time<r.sequence) (hc : col<r.dimension) :
    scalarResult op r back (fun o => (s.mem "Q" o).read .real)
      (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) (b*r.sequence+time) h col=
      logicalTarget op r back (fun b h t c => embed (Native.logicalRead api call.before call.input b h t c))
        (fun t j => embed (api.read call.before (tableAddress r call.cosine t j)))
        (fun t j => embed (api.read call.before (tableAddress r call.sine t j))) b h time col := by
  have ha := Native.row_bound r b time hb ht
  have hm : (b*r.sequence+time)%r.sequence=time := by simp [Nat.add_mod,Nat.mod_eq_of_lt ht]
  have hd : (b*r.sequence+time)/r.sequence=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt ht]; simp
  have input (c : Nat) (hc : c<r.dimension) :
      (s.mem "Q" (r.offset (b*r.sequence+time) h c)).read .real=
        embed (Native.logicalRead api call.before call.input b h time c) := by
    rw [call.inputBinding _ h c ha hh hc,hd,hm]
    unfold Native.logicalRead
    rw [call.inputAddress b h time c hb hh ht hc]
  unfold scalarResult logicalTarget
  split
  · rename_i active
    let j := if col<r.half then col else col-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    have dimension : 2*r.half≤r.dimension := by unfold Request.half; omega
    have coeff : time*r.half+j<r.sequence*r.half := by
      have c := Native.coefficient_bound r (b*r.sequence+time) j ha hj
      simpa [hm] using c
    dsimp only
    rw [input j (by omega),input (j+r.half) (by omega),hm,
      call.cosineBinding _ coeff,call.sineBinding _ coeff]
    rfl
  · exact input col hc

/-- Independent caller-indexed polynomial with the current coefficient views. -/
def mathematicalTarget (api : Memory.API ℝ Store Byte) (r : Request) (back : Bool)
    (before : Store) (input cosine sine : Native.Tensor) (b h time col : Nat) : ℝ :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    let pair := Spec.pair back (Native.logicalRead api before input b h time j)
      (Native.logicalRead api before input b h time (j+r.half))
      (api.read before (tableAddress r cosine time j)) (api.read before (tableAddress r sine time j))
    if col<r.half then pair.1 else pair.2
  else Native.logicalRead api before input b h time col

/-- Independent raw-word native graph on the current caller inputs. -/
def machineTarget (api : Memory.API Numerics.Word Store Byte) (p : Numerics.Primitives)
    (r : Request) (back : Bool) (before : Store) (input cosine sine : Native.Tensor)
    (b h time col : Nat) : Numerics.Word :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    Numerics.pair p back (decide (r.half≤col)) (Native.logicalRead api before input b h time j)
      (Native.logicalRead api before input b h time (j+r.half))
      (api.read before (tableAddress r cosine time j)) (api.read before (tableAddress r sine time j))
  else Native.logicalRead api before input b h time col

theorem ideal_logical (api : Memory.API ℝ Store Byte) (r : Request) (back : Bool)
    (before : Store) (input cosine sine : Native.Tensor) (b h time col : Nat) :
    logicalTarget ideal r back (Native.logicalRead api before input)
      (fun t j => api.read before (tableAddress r cosine t j))
      (fun t j => api.read before (tableAddress r sine t j)) b h time col=
      mathematicalTarget api r back before input cosine sine b h time col := by
  unfold logicalTarget mathematicalTarget
  split
  · dsimp only
    rw [ideal_pair]
    by_cases h : col<r.half <;> simp [h,show (r.half≤col)↔¬col<r.half from by omega]
  · rfl

theorem native_logical (api : Memory.API Numerics.Word Store Byte) (p : Numerics.Primitives)
    (r : Request) (back : Bool) (before : Store) (input cosine sine : Native.Tensor) (b h time col : Nat) :
    (logicalTarget (Numerics.operations p) r back
      (fun b h t c => Numerics.leaf (Native.logicalRead api before input b h t c))
      (fun t j => Numerics.leaf (api.read before (tableAddress r cosine t j)))
      (fun t j => Numerics.leaf (api.read before (tableAddress r sine t j))) b h time col).word=
      machineTarget api p r back before input cosine sine b h time col := by
  unfold logicalTarget machineTarget
  split
  · exact Numerics.source_pair p back _ _ _ _ _
  · rfl

#axiomsClean scalar_binding
#axiomsClean ideal_logical
#axiomsClean native_logical
#specNonCircular mathematicalTarget avoiding [Native.entry,Memory.run,FP.Structural.exec,scalarResult,scalarPair]
#specNonCircular machineTarget avoiding [Native.entry,Memory.run,FP.Structural.exec,scalarResult,scalarPair]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeEmbedding.Checked

import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedResult
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Native
import bench.tritonbench_g.rotary_emb.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section

/-- Independent selected native graphs, preserving their observed operand and
FMA contraction order. Every argument ranges over all 32-bit words. -/
def splitGraph (p : Native.Instructions) (cached : Bool) (a b c z : Native.Word)
    (odd : Bool) : Native.Word :=
  if cached then
    if odd then p.fma b c (p.mul a z) else p.fma a c (p.neg (p.mul b z))
  else
    if odd then p.fma z a (p.mul c b) else p.fma c a (p.neg (p.mul z b))

def uGraphValue (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key odd : Bool) (i : ULane (uLayout r)) : Cell Native.Node :=
  let l := uLayout r
  let a := ((s.mem (uRegion l key) (uOffset l s key Bool.false i)).read .real).value
  let b := ((s.mem (uRegion l key) (uOffset l s key Bool.true i)).read .real).value
  let c := ((s.mem l.cos (uCoefficient l s Bool.false i)).read .real).value
  let z := ((s.mem l.sin (uCoefficient l s Bool.true i)).read .real).value
  .mk .real (Native.atom (splitGraph p Bool.false a b c z odd))

def cGraphValue (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key odd : Bool) (i : CLane (cLayout r)) : Cell Native.Node :=
  let l := cLayout r
  let region := if key then l.k else l.q
  let a := ((s.mem region (cInput l s key Bool.false i)).read .real).value
  let b := ((s.mem region (cInput l s key Bool.true i)).read .real).value
  let c := ((s.mem l.cos (cCoefficient l s Bool.false i)).read .real).value
  let z := ((s.mem l.sin (cCoefficient l s Bool.true i)).read .real).value
  .mk .real (Native.atom (splitGraph p Bool.true a b c z odd))

theorem u_value_graph (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key odd : Bool) (i : ULane (uLayout r)) :
    uValue (Native.ops p Bool.false) (uLayout r) s s key odd i=uGraphValue p r s key odd i := by
  cases odd <;> rfl

theorem c_value_graph (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key odd : Bool) (i : CLane (cLayout r)) :
    cValue (Native.ops p Bool.true) (cLayout r) s key odd i=cGraphValue p r s key odd i := by
  cases odd <;> rfl

def uGraphBranch (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key : Bool) (entry : State Native.Node) : State Native.Node :=
  Pair.written (uRegion (uLayout r) key) (uActive (uLayout r) s key)
    (uOffset (uLayout r) s key) (uGraphValue p r s key) entry

def cGraphBranch (p : Native.Instructions) (r : Request) (s : State Native.Node)
    (key : Bool) (entry : State Native.Node) : State Native.Node :=
  Pair.written (cRegion (cLayout r) key) (cActive (cLayout r) s key)
    (cOffset (cLayout r) s key) (cGraphValue p r s key) entry

def uGraphTarget (p : Native.Instructions) (r : Request) (s : State Native.Node) : State Native.Node :=
  let q := uGraphBranch p r s Bool.false s
  if s.pids 0%r.groups=0 then uGraphBranch p r s Bool.true q else q

def cGraphTarget (p : Native.Instructions) (r : Request) (s : State Native.Node) : State Native.Node :=
  if s.pids 0<r.qHeads then cGraphBranch p r s Bool.false (cGraphBranch p r s Bool.true s) else s

theorem u_target_graph (p : Native.Instructions) (r : Request) (s : State Native.Node) :
    uTarget (Native.ops p Bool.false) r s=uGraphTarget p r s := by
  have values (key : Bool) : uValue (Native.ops p Bool.false) (uLayout r) s s key=uGraphValue p r s key :=
    funext fun odd => funext (u_value_graph p r s key odd)
  simp only [uTarget,uGraphTarget,uBranch,uGraphBranch,values]

theorem c_target_graph (p : Native.Instructions) (r : Request) (s : State Native.Node) :
    cReference (Native.ops p Bool.true) (cLayout r) s=cGraphTarget p r s := by
  have values (key : Bool) : cValue (Native.ops p Bool.true) (cLayout r) s key=cGraphValue p r s key :=
    funext fun odd => funext (c_value_graph p r s key odd)
  simp only [cReference,cGraphTarget,cBranch,cGraphBranch,values]
  rfl

#specNonCircular splitGraph avoiding [Kernels.uncached,Kernels.cached,uncachedExecution,cachedExecution]
#specNonCircular uGraphTarget avoiding [Kernels.uncached,uncachedExecution,uReference,uTarget]
#specNonCircular cGraphTarget avoiding [Kernels.cached,cachedExecution,cReference]
#axiomsClean u_target_graph
#axiomsClean c_target_graph
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed

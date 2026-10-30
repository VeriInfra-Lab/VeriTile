import bench.tritonbench_g.lightning_attention.Proofs.Model
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]
def intraQ (M : Algebra α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.D p.BT :=
  fun i => read M m "query" (head*p.N*p.D+(block*p.BT+i.2.1.val)*p.D+i.1.val)
    (block*p.BT+i.2.1.val<p.N)
def intraK (M : Algebra α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.D :=
  fun i => read M m "key" (head*p.N*p.D+(block*p.BT+i.1.val)*p.D+i.2.1.val)
    (block*p.BT+i.1.val<p.N)
def intraV (M : Algebra α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.E p.BT :=
  fun i => read M m "value" (head*p.N*p.E+(block*p.BT+i.2.1.val)*p.E+i.1.val)
    (block*p.BT+i.2.1.val<p.N)
def intraG (M : Algebra α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.E :=
  fun i => read M m "incoming" (head*p.N*p.E+(block*p.BT+i.1.val)*p.E+i.2.1.val)
    (block*p.BT+i.1.val<p.N)
def intraDS (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.BT :=
  let score := dot D (intraG M p head block m) (intraV M p head block m)
  fun i => if i.2.1.val ≤ i.1.val then score i else zero M
def intraST (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.BT :=
  let score := dot D (intraK M p head block m) (intraQ M p head block m)
  fun i => if i.1.val ≤ i.2.1.val then score i else zero M
def intraDQ (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.D :=
  dot D (intraDS M D p head block m) (intraK M p head block m)
def intraDKT (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.D p.BT :=
  dot D (intraQ M p head block m) (intraDS M D p head block m)
def intraDV (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Matrix α p.BT p.E :=
  dot D (intraST M D p head block m) (intraG M p head block m)

def writeRows (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [rows,width]) r (fun i => start+i.1.val<N)
    (fun i => head*N*width+(start+i.1.val)*width+i.2.1.val)
    (fun i => .mk .real (cast M (values i))) m
def writeTrans (M : Algebra α) (r : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [width,rows]) r (fun i => start+i.2.1.val<N)
    (fun i => head*N*width+(start+i.2.1.val)*width+i.1.val)
    (fun i => .mk .real (cast M (values i))) m
def intraMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat) (m : Memory α) : Memory α :=
  writeRows M "valueGradient" p.N p.E head (block*p.BT) p.BT (intraDV M D p head block m)
    (writeTrans M "keyGradient" p.N p.D head (block*p.BT) p.BT (intraDKT M D p head block m)
      (writeRows M "queryGradient" p.N p.D head (block*p.BT) p.BT (intraDQ M D p head block m) m))
theorem writeRows_inputs (M : Algebra α) (r : RegionName)
    (hr : r∉(["query","key","value","incoming"] : List RegionName))
    (N width head start rows : Nat) (values : Matrix α rows width) (m : Memory α) :
    Inputs (writeRows M r N width head start rows values m) m :=
  scatter_inputs _ _ hr _ _ _ _
theorem writeTrans_inputs (M : Algebra α) (r : RegionName)
    (hr : r∉(["query","key","value","incoming"] : List RegionName))
    (N width head start rows : Nat) (values : Matrix α width rows) (m : Memory α) :
    Inputs (writeTrans M r N width head start rows values m) m :=
  scatter_inputs _ _ hr _ _ _ _
theorem intraMemory_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (m : Memory α) : Inputs (intraMemory M D p head block m) m :=
  (writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _).trans
    ((writeTrans_inputs _ _ (by decide) _ _ _ _ _ _ _).trans
      (writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _))
#axiomsClean intraMemory_inputs
end
end VeriTile.Bench.Repairs.LightningAttention

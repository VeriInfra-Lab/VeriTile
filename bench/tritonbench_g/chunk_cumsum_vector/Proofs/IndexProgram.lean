import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.IndexProgram
open VeriTile.Triton

def startOp (nm : RegName) (width : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] nm) (.constNat width)
def activeOp (length width : Nat) (lanes start : RegName) : Op .bool [width] :=
  .lt .nat .scalarR (.ref .nat [width] lanes)
    (.sub .nat .nil (.constNat length) (.ref .nat [] start))
def coordinateOp (width : Nat) (start active lanes : RegName) : Op .nat [width] :=
  .add .nat .scalarL (.ref .nat [] start)
    (.where (.ref .bool [width] active) (.ref .nat [width] lanes)
      (.broadcast (.constNat 0) [width]))
def maskOp (c : Config) : Op .bool [c.BT,BS] :=
  .boolAnd (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .bool [c.BT] "active_t"))
    (.expandDim ⟨0,by simp⟩ (.ref .bool [BS] "active_s"))
def pointerOp (c : Config) (r : RegionName) (strides : Strides) : Op .ptr [c.BT,BS] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL
        (.add .nat .nil
          (.mul .nat .nil (.ref .nat [] "i_b") (.constNat strides.batch))
          (.mul .nat .nil (.ref .nat [] "i_h") (.constNat strides.head)))
        (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (.ref .nat [c.BT] "times"))
          (.constNat strides.time)))
      (.mul .nat .scalarR (.expandDim ⟨0,by simp⟩ (.ref .nat [BS] "columns"))
        (.constNat strides.channel)))
def countOp (h : Call) (c : Config) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat h.T) (.constNat c.BT))
    (.constNat 1)) (.constNat c.BT)
def triangularOp (c : Config) : Op .real [c.BT,c.BT] :=
  .where (.ge .nat (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .nat [c.BT] "o_i"))
    (.expandDim ⟨0,by simp⟩ (.ref .nat [c.BT] "o_i")))
    (.broadcast (.const 1.0) [c.BT,c.BT]) (.broadcast (.const 0.0) [c.BT,c.BT])
def outputOp (c : Config) : Op .real [c.BT,BS] :=
  .add .real (.consL (.consSame .nil))
    (.expandDim ⟨0,by simp⟩ (.ref .real [BS] "b_z"))
    (.dot (batch:=[]) (.ref .real [c.BT,c.BT] "m_s") (.ref .real [c.BT,BS] "b_s")
      {allowTF32:=some Bool.false})
def carryOp (c : Config) : Op .real [BS] :=
  .add .real (.consSame .nil) (.ref .real [BS] "b_z")
    (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT,BS] "b_s"))
end VeriTile.Bench.Repairs.ChunkCumsumVector.IndexProgram

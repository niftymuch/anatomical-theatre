"""GLB -> USDZ for AR Quick Look.  Y-up, metres, UsdPreviewSurface materials."""
import os, sys, shutil, tempfile
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils


def convert(glb_path, usdz_path):
    scene = trimesh.load(glb_path)
    work = tempfile.mkdtemp()
    usd_path = os.path.join(work, "model.usdc")

    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Theatre")
    stage.SetDefaultPrim(root.GetPrim())
    looks = UsdGeom.Scope.Define(stage, "/Theatre/Looks")

    made = {}
    for name, g in scene.geometry.items():
        safe = "".join(c if c.isalnum() else "_" for c in name)
        mat_name = safe.split("_")[0]

        # ---- material, once per palette entry ----
        if mat_name not in made:
            mpath = f"/Theatre/Looks/{mat_name}"
            mat = UsdShade.Material.Define(stage, mpath)
            sh = UsdShade.Shader.Define(stage, mpath + "/Surface")
            sh.CreateIdAttr("UsdPreviewSurface")
            c = np.array(g.visual.material.baseColorFactor[:3]) / 255.0
            a = float(g.visual.material.baseColorFactor[3]) / 255.0
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(g.visual.material.roughnessFactor or 0.85))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if a < 0.99:
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(a)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            made[mat_name] = mat

        # ---- mesh ----
        mesh = UsdGeom.Mesh.Define(stage, f"/Theatre/{safe}")
        v = np.asarray(g.vertices, dtype=np.float32)
        f = np.asarray(g.faces, dtype=np.int32)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v.tolist()])
        mesh.CreateFaceVertexCountsAttr([3] * len(f))
        mesh.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        n = np.asarray(g.vertex_normals, dtype=np.float32)
        mesh.CreateNormalsAttr([Gf.Vec3f(*p) for p in n.tolist()])
        mesh.SetNormalsInterpolation("vertex")
        ext = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float32)
        mesh.CreateExtentAttr([Gf.Vec3f(*ext.tolist()[0]), Gf.Vec3f(*ext.tolist()[1])])
        UsdShade.MaterialBindingAPI(mesh).Bind(made[mat_name])

    stage.GetRootLayer().Save()
    if os.path.exists(usdz_path):
        os.remove(usdz_path)
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(usd_path), usdz_path)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{usdz_path}  {os.path.getsize(usdz_path)/1024:.0f} KB")


if __name__ == "__main__":
    convert("/home/claude/theatre.glb", "/home/claude/theatre.usdz")

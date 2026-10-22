"""Airflow ground truth: gmsh mesh plus a steady OpenFOAM run per nasal side.

Each side is its own domain: the elliptical tube from `nasal.geometry`, lofted
in gmsh and filled with tetrahedra sized to the local tube width. OpenFOAM
(simpleFoam in Docker) drives it with the reference pressure drop: total
pressure REFERENCE_PRESSURE at the nostril, static 0 at the choanal exit.
At 150 Pa the valve Reynolds number is around 10^4, so the default model is
k-omega SST RANS; laminar is available for comparison.

Region pressure drops come from the flux-weighted total pressure on planes at
the region boundaries (x positions from REGION_LENGTH), so they are losses
and sum to the side's drop. The output record uses the same column names and
units (Pa*s/cm^3) as `nasal.resistance.evaluate`, plus solver checks.
"""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .geometry import REGION_LENGTH, SideGeometry, airway, area_profile
from .params import REGIONS, SIDES, CaseParams
from .resistance import AIR_DENSITY, AIR_VISCOSITY, CM3, REFERENCE_PRESSURE

IMAGE = "opencfd/openfoam-run:2606"
NU = AIR_VISCOSITY / AIR_DENSITY  # kinematic viscosity, m^2/s
TURBULENCE = ("kOmegaSST", "laminar")
CELLS_ACROSS = 6  # tetrahedra across the narrow axis of the tube
SECTIONS = 60  # elliptical sections lofted along the tube
MAX_ITERATIONS = 3000
AVERAGE_LAST = 300  # iterations averaged when the residual targets are not met
RAMP_ITERATIONS = 300  # inlet total pressure ramps up from 0 so the start stays stable
INLET_INTENSITY = 0.05
INLET_VELOCITY = 10.0  # m/s, typical nostril velocity at 150 Pa; sets inflow k only
INLET_MIXING_LENGTH = 1e-3  # m

FOAM_HEADER = """FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {obj};
}}
"""


def region_boundaries() -> list[float]:
    """x positions (m) of the planes between consecutive regions."""
    return [float(v) for v in np.cumsum([REGION_LENGTH[r] for r in REGIONS])[:-1]]


def mesh_side(geom: SideGeometry, path: Path, cells_across: int = CELLS_ACROSS) -> int:
    """Write a gmsh 2.2 tetrahedral mesh with patches inlet, outlet and wall.

    The tube axis is x and the side's centreline is y=0, so both sides share
    one frame. Returns the number of tetrahedra.
    """
    import gmsh

    x, area, aspect = area_profile(geom)
    width = np.sqrt(area / (np.pi * aspect))
    height = aspect * width
    idx = np.unique(np.linspace(0, len(x) - 1, SECTIONS).round().astype(int))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        loops = []
        for i in idx:
            # gmsh puts the major radius on xAxis, so point it along z (height).
            c = occ.addEllipse(x[i], 0, 0, height[i], width[i], zAxis=[1, 0, 0], xAxis=[0, 0, 1])
            loops.append(occ.addCurveLoop([c]))
        vol = occ.addThruSections(loops, makeSolid=True, makeRuled=False)[0][1]
        occ.synchronize()

        patches = {"inlet": [], "outlet": [], "wall": []}
        for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False):
            xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, tag)
            if xmax - xmin > 1e-6:
                patches["wall"].append(tag)
            else:
                patches["inlet" if xmax < x[-1] / 2 else "outlet"].append(tag)
        for name, tags in patches.items():
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.model.addPhysicalGroup(3, [vol], name="fluid")

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, px, py, pz, lc: 2 * float(np.interp(px, x, width)) / cells_across
        )
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
        _, tets, _ = gmsh.model.mesh.getElements(3)
        return int(sum(len(t) for t in tets))
    finally:
        gmsh.finalize()


def _dict(cls: str, obj: str, body: str) -> str:
    return FOAM_HEADER.format(cls=cls, obj=obj) + body


def _field(cls: str, name: str, dims: str, internal: str, inlet: str, outlet: str, wall: str,
           inlet_key: str = "inlet") -> str:
    return _dict(cls, name, f"""
dimensions {dims};
internalField uniform {internal};
boundaryField
{{
    {inlet_key} {{ {inlet} }}
    outlet {{ {outlet} }}
    wall {{ {wall} }}
}}
""")


def case_files(processes: int = 8, turbulence: str = "kOmegaSST", max_iterations: int = MAX_ITERATIONS,
               inlets: tuple[str, ...] = ("inlet",), sample_planes: bool = True,
               mesh_commands: list[str] | None = None, write_format: str = "binary",
               reconstruct: bool = False, planes: list[tuple[str, tuple, tuple]] | None = None) -> dict[str, str]:
    """OpenFOAM case dictionaries and run script, keyed by relative path.

    The defaults build the idealized tube case from a gmsh mesh. Patient
    airways (`nasal.anatomy`) pass their own inlet patches and mesh commands,
    skip the plane sampling, and keep the full fields in ASCII. `planes` replaces
    the tube's x-planes with named cutting planes (name, point, normal) in meters,
    as the smooth patient airways use (`nasal.smooth`).
    """
    if turbulence not in TURBULENCE:
        raise ValueError(f"turbulence must be one of {TURBULENCE}")
    rans = turbulence != "laminar"
    p0 = REFERENCE_PRESSURE / AIR_DENSITY  # simpleFoam pressure is kinematic
    ramp = min(RAMP_ITERATIONS, max_iterations // 3)
    if planes is None:
        planes = [(f"x{i}", (xb, 0, 0), (1, 0, 0)) for i, xb in enumerate(region_boundaries(), start=1)]
    planes = "\n".join(
        f"            {name} {{ type cuttingPlane; point ({pt[0]:.8g} {pt[1]:.8g} {pt[2]:.8g}); "
        f"normal ({nv[0]} {nv[1]} {nv[2]}); interpolate false; }}"
        for name, pt, nv in planes
    )
    inlet_key = inlets[0] if len(inlets) == 1 else '"(' + "|".join(inlets) + ')"'
    field = lambda *args: _field(*args, inlet_key=inlet_key)  # noqa: E731
    plane_function = f"""    planes
    {{
        type surfaces;
        libs (sampling);
        writeControl writeTime;
        surfaceFormat raw;
        formatOptions {{ raw {{ normal yes; }} }}
        interpolationScheme cell;
        fields (p U pMean UMean);
        surfaces
        {{
{planes}
        }}
    }}""" if sample_planes else ""
    flow_objects = "\n".join(
        f"""    {patch}Flow
    {{
        type surfaceFieldValue;
        libs (fieldFunctionObjects);
        regionType patch;
        name {patch};
        operation sum;
        fields (phi);
        writeFields false;
        log false;
    }}"""
        for patch in (*inlets, "outlet")
    )
    files = {
        "system/controlDict": _dict("dictionary", "controlDict", f"""
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {max_iterations};
deltaT 1;
writeControl timeStep;
writeInterval {max_iterations};
purgeWrite 1;
writeFormat {write_format};
runTimeModifiable false;

functions
{{
{flow_objects}
    average
    {{
        type fieldAverage;
        libs (fieldFunctionObjects);
        timeStart {max_iterations - AVERAGE_LAST};
        writeControl writeTime;
        fields ( p {{ mean on; prime2Mean off; base time; }} U {{ mean on; prime2Mean off; base time; }} );
    }}
{plane_function}
}}
"""),
        "system/fvSchemes": _dict("dictionary", "fvSchemes", """
ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
"""),
        "system/fvSolution": _dict("dictionary", "fvSolution", """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    residualControl { p 1e-4; "(U|k|omega)" 1e-5; }
}
relaxationFactors { fields { p 0.3; } equations { ".*" 0.7; } }
"""),
        "system/decomposeParDict": _dict("dictionary", "decomposeParDict", f"""
numberOfSubdomains {processes};
method scotch;
"""),
        "constant/transportProperties": _dict("dictionary", "transportProperties", f"""
transportModel Newtonian;
nu {NU:.6g};
"""),
        "constant/turbulenceProperties": _dict("dictionary", "turbulenceProperties",
            f"\nsimulationType RAS;\nRAS {{ RASModel {turbulence}; turbulence on; printCoeffs off; }}\n"
            if rans else "\nsimulationType laminar;\n"),
        "0/p": field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "0",
                      f"type uniformTotalPressure; p0 table ((0 0) ({ramp} {p0:.6g})); value uniform 0;",
                      "type fixedValue; value uniform 0;", "type zeroGradient;"),
        "0/U": field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "(0 0 0)",
                      "type pressureInletOutletVelocity; value uniform (0 0 0);",
                      "type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);",
                      "type noSlip;"),
    }
    if rans:
        # Fixed inflow turbulence: velocity-scaled inlet conditions give k = 0
        # while the inlet pressure ramps up from rest, and nut = k/omega fails.
        k = 1.5 * (INLET_INTENSITY * INLET_VELOCITY) ** 2
        omega = np.sqrt(k) / (0.09**0.25 * INLET_MIXING_LENGTH)
        free = lambda v: f"type inletOutlet; inletValue uniform {v:.6g}; value uniform {v:.6g};"
        files["0/k"] = field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"{k:.6g}",
                              free(k), free(k), f"type kqRWallFunction; value uniform {k:.6g};")
        files["0/omega"] = field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"{omega:.6g}",
                                  free(omega), free(omega), f"type omegaWallFunction; value uniform {omega:.6g};")
        files["0/nut"] = field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "0",
                                "type calculated; value uniform 0;", "type calculated; value uniform 0;",
                                "type nutkWallFunction; value uniform 0;")
    run = [
        "set -e",
        # The container runs as root; give the files back to the case folder's
        # owner so the host user can delete or rerun the case on Linux.
        "trap 'chown -R --reference=. . 2>/dev/null || true' EXIT",
        *(mesh_commands if mesh_commands is not None else [
            "gmshToFoam side.msh > log.gmshToFoam",
            "foamDictionary constant/polyMesh/boundary -entry entry0/wall/type -set wall > /dev/null",
        ]),
        "checkMesh > log.checkMesh || true",
    ]
    if processes > 1:
        run += [
            "decomposePar -force > log.decomposePar",
            # No core binding: by default every container pins its ranks to
            # cores 0..n-1, so parallel batches would all share the same cores.
            f"mpirun --allow-run-as-root --bind-to none -np {processes} simpleFoam -parallel > log.simpleFoam",
            *(["reconstructPar -latestTime > log.reconstructPar"] if reconstruct else []),
            "rm -rf processor*",
        ]
    else:
        run.append("simpleFoam > log.simpleFoam")
    files["Allrun"] = "\n".join(run) + "\n"
    return files


def write_case(case_dir: Path, geom: SideGeometry, processes: int = 8, turbulence: str = "kOmegaSST",
               cells_across: int = CELLS_ACROSS) -> int:
    """Write a fresh case directory (dictionaries plus mesh); returns the cell count."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for rel, text in case_files(processes, turbulence).items():
        path = case_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return mesh_side(geom, case_dir / "side.msh", cells_across)


def docker_available() -> bool:
    docker = shutil.which("docker")
    return docker is not None and subprocess.run([docker, "info"], capture_output=True).returncode == 0


def run_case(case_dir: Path, image: str = IMAGE) -> None:
    docker = shutil.which("docker") or "docker"
    cmd = [docker, "run", "--rm", "-v", f"{case_dir.resolve()}:/case", image,
           "bash", "-lc", "cd /case && bash Allrun"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
        raise RuntimeError(f"OpenFOAM failed in {case_dir} (see its log.* files):\n{tail}")


# ---- result parsing (pure functions, no OpenFOAM needed) ----


def _latest(directory: Path) -> Path:
    times = [d for d in directory.iterdir() if d.is_dir()]
    return max(times, key=lambda d: float(d.name))


def read_flux_history(path: Path) -> np.ndarray:
    """(iteration, summed phi in m^3/s) rows from a surfaceFieldValue .dat file."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


def read_raw_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Face area vectors (n, 3) and values, (n,) or (n, 3), from a raw surface
    file written with `normal yes`: columns x y z value... area_x area_y area_z."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    values = data[:, 3:-3]
    return data[:, -3:], values[:, 0] if values.shape[1] == 1 else values


def plane_total_pressure(area: np.ndarray, p: np.ndarray, u: np.ndarray) -> float:
    """Flux-weighted total pressure (kinematic) through a plane."""
    flux = np.einsum("ij,ij->i", u, area)
    pt = p + 0.5 * np.einsum("ij,ij->i", u, u)
    return float(np.sum(pt * flux) / np.sum(flux))


def side_from_planes(flow: float, plane_pt: list[float], pressure: float = REFERENCE_PRESSURE) -> dict:
    """Side flow (cm^3/s) and region resistances (Pa*s/cm^3) from the side's
    flow (m^3/s) and the kinematic total pressure on each boundary plane."""
    pts = [pressure] + [AIR_DENSITY * pt for pt in plane_pt] + [0.0]
    q_cm3 = flow / CM3
    region = {r: (pts[i] - pts[i + 1]) / q_cm3 for i, r in enumerate(REGIONS)}
    return {"flow_cm3_s": q_cm3, "resistance": pressure / q_cm3, "region_resistance": region}


def read_side(case_dir: Path, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Parse one finished side case.

    If the residual targets were met, uses the final fields. Otherwise the run
    stopped at max_iterations and the flow and plane fields are averaged over
    the last AVERAGE_LAST iterations; `unsteadiness` (flow std / mean over that
    window) says how much the solution still moves.
    """
    post = case_dir / "postProcessing"
    inlet = read_flux_history(_latest(post / "inletFlow") / "surfaceFieldValue.dat")
    outlet = read_flux_history(_latest(post / "outletFlow") / "surfaceFieldValue.dat")
    iterations = int(inlet[-1, 0])
    converged = iterations < max_iterations
    window = inlet[inlet[:, 0] > iterations - AVERAGE_LAST, 1]
    if converged:
        flow, p_name, u_name = -inlet[-1, 1], "p", "U"
    else:
        flow, p_name, u_name = -float(np.mean(window)), "pMean", "UMean"

    plane_dir = _latest(post / "planes")
    pts = []
    for i in range(1, len(REGIONS)):
        area, p = read_raw_surface(plane_dir / f"{p_name}_x{i}.raw")
        _, u = read_raw_surface(plane_dir / f"{u_name}_x{i}.raw")
        pts.append(plane_total_pressure(area, p, u))
    result = side_from_planes(flow, pts)
    result.update(
        iterations=iterations,
        converged=converged,
        unsteadiness=float(np.std(window) / abs(np.mean(window))),
        mass_imbalance=float(abs(inlet[-1, 1] + outlet[-1, 1]) / abs(inlet[-1, 1])),
    )
    return result


def to_record(sides: dict[str, dict], pressure: float = REFERENCE_PRESSURE) -> dict:
    """Flat record matching `nasal.resistance.evaluate`, plus solver checks."""
    total_flow = sum(s["flow_cm3_s"] for s in sides.values())
    record = {"total_resistance": pressure / total_flow, "total_flow_cm3_s": total_flow}
    for side in SIDES:
        s = sides[side]
        record[f"{side}_flow_cm3_s"] = s["flow_cm3_s"]
        record[f"{side}_resistance"] = s["resistance"]
        for region in REGIONS:
            record[f"{side}_{region}_resistance"] = s["region_resistance"][region]
    for side in SIDES:
        for key in ("cells", "iterations", "converged", "unsteadiness", "seconds"):
            record[f"{side}_{key}"] = sides[side].get(key)
    return record


def simulate(params: CaseParams, work_dir: Path, processes: int = 8, turbulence: str = "kOmegaSST",
             cells_across: int = CELLS_ACROSS, image: str = IMAGE) -> dict:
    """Mesh and run both sides of one case; return the flat result record."""
    sides = {}
    for side, geom in airway(params).items():
        case_dir = work_dir / side
        start = time.perf_counter()
        cells = write_case(case_dir, geom, processes, turbulence, cells_across)
        run_case(case_dir, image)
        seconds = time.perf_counter() - start  # meshing plus solving, wall clock
        sides[side] = {**read_side(case_dir), "cells": cells, "seconds": seconds}
    return to_record(sides)


def compare_to_baseline(simulated: "pd.DataFrame") -> "pd.DataFrame":
    """Simulated / formula-baseline ratio per case for total, side and region
    resistance. The baseline is recomputed from each row's parameters."""
    import pandas as pd

    from .params import PARAM_SPECS
    from .resistance import evaluate

    columns = ["total_resistance"] + [f"{s}_resistance" for s in SIDES] + [
        f"{s}_{r}_resistance" for s in SIDES for r in REGIONS
    ]
    rows = []
    for row in simulated.itertuples(index=False):
        base = evaluate(CaseParams(**{k: getattr(row, k) for k in PARAM_SPECS}))
        rows.append({"case_id": row.case_id, **{c: getattr(row, c) / base[c] for c in columns}})
    return pd.DataFrame(rows)

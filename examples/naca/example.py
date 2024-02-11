# Shape Optimization of Airfoil
# Multiple Objective
# Multiple Constrained

import SU2
from FADO import *
import subprocess
import os, sys

### VARIABLES ###
# MISC
base_config = "config.cfg"
nameObjs = ["DRAG"]
nameCons = ["LIFT", "MOMENT_Z"]
n_cons = len(nameCons)
restart = "NO"
restartAdj = "NO"
path_restart = "DIRECT/solution.dat"
path_restart_adj = "RESTART_ADJOINT/of_grad.dat"
config_tmpl_filename = "config_tmp.cfg"

# Extract from SU2 config ---------------------------------------------------- #
config = SU2.io.Config(base_config)

mesh_filename = config['MESH_FILENAME']   
mesh_out_filename = config['MESH_OUT_FILENAME']
restart_filename = config["RESTART_FILENAME"]
solution_filename = config["SOLUTION_FILENAME"]
restart_adj_filename = config["RESTART_ADJ_FILENAME"]
solution_adj_filename = config["SOLUTION_ADJ_FILENAME"]
restart_lift_adj_filename = f"{restart_adj_filename[:-4]}_cl{restart_adj_filename[-4:]}"
restart_drag_adj_filename = f"{restart_adj_filename[:-4]}_cd{restart_adj_filename[-4:]}"
solution_lift_adj_filename = f"{solution_adj_filename[:-4]}_cl{solution_adj_filename[-4:]}"
solution_drag_adj_filename = f"{solution_adj_filename[:-4]}_cd{solution_adj_filename[-4:]}"

designparams = copy.deepcopy(config['DV_VALUE_OLD'])
designparams = np.array(designparams)
lb = float(copy.deepcopy(config['OPT_BOUND_LOWER']))
ub = float(copy.deepcopy(config['OPT_BOUND_UPPER']))

# Input variables ----------------------------------------------------- #
x = InputVariable(designparams, ArrayLabelReplacer("__X__"),0,1.0,lb,ub)

###########################################
#
#   PARAMETERS
#   - __MATH_PROBLEM__
#   - __MESH_FILENAME__
#   - __RES__
#   - __FUNCTION__
#   - __X__
#
###########################################  
pType_geometry = Parameter(["0.001"+",0.001"*(len(designparams)-1)],LabelReplacer("__X__"))

# switch from direct to adjoint mode and adapt settings
pType_direct = Parameter(["DIRECT"], LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"], LabelReplacer("__MATH_PROBLEM__"))
# switch default and deformed mesh
pType_mesh_filename_original = Parameter([mesh_filename], LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter([mesh_out_filename], LabelReplacer("__MESH_FILENAME__"))
# switch restart on and off
pType_restart_on = Parameter(["YES"],LabelReplacer("__RES__"))
pType_restart_off = Parameter(["NO"],LabelReplacer("__RES__"))
# switch between objectives and constraints
pType_obj = Parameter(["DRAG"], LabelReplacer("__FUNCTION__"))
pType_objs = []
for nameObj in nameObjs:
    pType_obj = Parameter([nameObj], LabelReplacer("__FUNCTION__"))
    pType_objs.append(pType_obj)

pType_constraints = []
for nameCon in nameCons:
    pType_constraint = Parameter([nameCon], LabelReplacer("__FUNCTION__"))
    pType_constraints.append(pType_constraint)

# Evaluations ---------------------------------------------------------- #
def_command = "SU2_DEF config_tmp.cfg"
geo_command = "SU2_GEO config_tmp.cfg"
dir_command = "mpirun -n 12 SU2_CFD config_tmp.cfg"
adj_command = "mpirun -n 12 SU2_CFD_AD config_tmp.cfg"
dot_command = "mpirun -n 12 SU2_DOT_AD config_tmp.cfg"
copyCommandDirect = f"cp ../DIRECT/{restart_filename} {solution_filename}"
copyCommandAdjointDrag = f"cp ../ADJOINT_DRAG/{restart_drag_adj_filename} {solution_drag_adj_filename}"
copyCommandAdjointLift = f"cp ../ADJOINT_LIFT/{restart_lift_adj_filename} {solution_lift_adj_filename}"
max_tries   = 1

# DEFORMATION RUN
meshDeformationRun = ExternalRun("DEFORM", def_command, True)
meshDeformationRun.addConfig(config_tmpl_filename)
meshDeformationRun.addData(mesh_filename)
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_restart_off)
meshDeformationRun.addParameter(pType_obj)
meshDeformationRun.addExpected(f"{mesh_out_filename}")

# GEOMETRY RUN
geometryRun = ExternalRun("GEOMETRY", geo_command, True)
geometryRun.addConfig(config_tmpl_filename)
geometryRun.addData(f"DEFORM/{mesh_out_filename}")
geometryRun.addParameter(pType_geometry)
geometryRun.addParameter(pType_adjoint)
geometryRun.addParameter(pType_mesh_filename_deformed)
geometryRun.addParameter(pType_restart_off)
geometryRun.addParameter(pType_obj)
geometryRun.addExpected("of_func.csv")
geometryRun.addExpected("of_grad.csv")

# DIRECT RUN
directRun = ExternalRun("DIRECT", dir_command, True)
directRun.addConfig(config_tmpl_filename)
directRun.addData(f"DEFORM/{mesh_out_filename}")
directRun.addParameter(pType_direct)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_obj)
if (restart=="YES"):
    directRun.addParameter(pType_restart_on)
    directRun.addData(path_restart)
else:
    directRun.addParameter(pType_restart_off)
directRun.addExpected("history.csv")

# COPY POST DIRECT RUN
copyRun = ExternalRun("RESTART", copyCommandDirect, True)
copyRun.addData(f"DIRECT/{restart_filename}")        

# ADJOINT DRAG
adjointRun= ExternalRun("ADJOINT_DRAG", adj_command, True)
adjointRun.addConfig(config_tmpl_filename)
adjointRun.addData(f"DEFORM/{mesh_out_filename}")
adjointRun.addData(f"RESTART/{solution_filename}")
adjointRun.addParameter(pType_adjoint)
adjointRun.addParameter(pType_mesh_filename_deformed)
adjointRun.addParameter(pType_obj)
if (restartAdj=="YES"):
    adjointRun.addParameter(pType_restart_on)
    adjointRun.addData(f"ADJOINT_DRAG/{restart_adj_filename}")
else:
    adjointRun.addParameter(pType_restart_off)
adjointRun.addExpected(f"{restart_drag_adj_filename}")
                            
# COPY POST ADJOINT RUN
copyAdjRun = ExternalRun("RESTART_ADJOINT_DRAG", copyCommandAdjointDrag, True)
copyAdjRun.addData(f"ADJOINT_DRAG/{restart_drag_adj_filename}")

# DOT RUN
dotProductDragRun= ExternalRun("DOT_DRAG",dot_command, True)
dotProductDragRun.addConfig(config_tmpl_filename)
dotProductDragRun.addData(f"DEFORM/{mesh_out_filename}")
dotProductDragRun.addData(f"RESTART_ADJOINT_DRAG/{solution_drag_adj_filename}")
dotProductDragRun.addParameter(pType_adjoint)
dotProductDragRun.addParameter(pType_mesh_filename_deformed)
dotProductDragRun.addParameter(pType_obj)
dotProductDragRun.addParameter(pType_restart_off)
dotProductDragRun.addExpected("of_grad.dat")

# ADJOINT LIFT
adjointRunCons = ExternalRun("ADJOINT_LIFT", adj_command, True)
adjointRunCons.addConfig(config_tmpl_filename)
adjointRunCons.addData(f"DEFORM/{mesh_out_filename}")
adjointRunCons.addData(f"RESTART/{solution_filename}") 
adjointRunCons.addParameter(pType_adjoint)
adjointRunCons.addParameter(pType_mesh_filename_deformed)
adjointRunCons.addParameter(pType_constraints[0])
if (restartAdj=="YES"):
    adjointRunCons.addParameter(pType_restart_on)
    adjointRunCons.addData(f"ADJOINT_LIFT/{restart_adj_filename}")
else:
    adjointRunCons.addParameter(pType_restart_off)
adjointRunCons.addExpected(f"{restart_lift_adj_filename}")

# POST LIFT RUN
copyAdjConsRun=ExternalRun("RESTART_ADJOINT_LIFT", copyCommandAdjointLift, True)
copyAdjConsRun.addData(f"ADJOINT_LIFT/{restart_lift_adj_filename}")

# DOT LIFT
dotProductLiftRun = ExternalRun("DOT_LIFT", dot_command, True)
dotProductLiftRun.addConfig(config_tmpl_filename)
dotProductLiftRun.addData(f"DEFORM/{mesh_out_filename}")
dotProductLiftRun.addData(f"RESTART_ADJOINT_LIFT/{solution_lift_adj_filename}")
dotProductLiftRun.addParameter(pType_adjoint)
dotProductLiftRun.addParameter(pType_mesh_filename_deformed)
dotProductLiftRun.addParameter(pType_constraints[0])
dotProductLiftRun.addParameter(pType_restart_off)
dotProductLiftRun.addExpected("of_grad.dat")

###########################################
#
#   Objectives
#   - DRAG
#   Constraints
#   - AIRFOIL_THICKNESS 
#   - LIFT
#   
###########################################        

# Drag objective function
fun = Function("DRAG", "DIRECT/history.csv", LabeledTableReader(f"\"CD\""))
fun.addInputVariable(x, "DOT_DRAG/of_grad.dat", TableReader(None,0,start=(1,0),end=(None,None)))
fun.addValueEvalStep(meshDeformationRun) 
fun.addValueEvalStep(directRun)
fun.addGradientEvalStep(copyRun)
fun.addGradientEvalStep(adjointRun)
fun.addGradientEvalStep(copyAdjRun)
fun.addGradientEvalStep(dotProductDragRun)

# Lift constraint function
cons_fun = Function("LIFT", "DIRECT/history.csv", LabeledTableReader(f"\"CL\""))
cons_fun.addInputVariable(x,"DOT_LIFT/of_grad.dat", TableReader(None,0,start=(1,0),end=(None,None)))
cons_fun.addValueEvalStep(meshDeformationRun) 
cons_fun.addValueEvalStep(directRun)
cons_fun.addGradientEvalStep(copyRun)
cons_fun.addGradientEvalStep(adjointRunCons)
cons_fun.addGradientEvalStep(copyAdjConsRun)
cons_fun.addGradientEvalStep(dotProductLiftRun)

# Thickness constraint function
nameConsThickness = "AIRFOIL_THICKNESS"
cons_fun_geo = Function("AIRFOIL_THICKNESS","GEOMETRY/of_func.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\""))
cons_fun_geo.addInputVariable(x, "GEOMETRY/of_grad.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\"", rang=(1, None)))
cons_fun_geo.addValueEvalStep(meshDeformationRun)
cons_fun_geo.addValueEvalStep(geometryRun)        

# Setup driver
driver = ScipyDriver()
driver.addObjective("min", fun, 1.0) # objective
driver.addLowerBound(cons_fun_geo, 0.12, 0.001) # thickness constraint
driver.addLowerBound(cons_fun, 0.28, 0.001) # lift constraint

driver.setWorkingDirectory("OPTIM")
driver.setEvaluationMode(True, 0.1) # Sets parallel evaluation mode
driver.setStorageMode(True, "DESIGN/DSN_")
driver.setFailureMode("SOFT")

his = open("optim.his", "w", 1)
driver.setHistorian(his)

driver.preprocess()
x = driver.getInitial()

# Optimization, SciPy -------------------------------------------------- #
import scipy.optimize

driver.preprocess()
x = driver.getInitial()

options = {'disp': True, 'ftol': 1e-7, 'maxiter': 100}

optimum = scipy.optimize.minimize(driver.fun, x, method="SLSQP", jac=driver.grad,\
          constraints=driver.getConstraints(), bounds=driver.getBounds(), options=options)

his.close()
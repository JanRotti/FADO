# Shape Optimization of Airfoil using SU2

import SU2
from FADO import *
from PYRDO import *
import subprocess
import os, sys

# DRO PARAMETERS
par = {"name":"MACH", "data":[0.73, 0.76, 0.8, 0.83, 0.85], "type":"data"}
droPar = DROParameter(par, LabelReplacer("__MACH_NUMBER__"))
default = Parameter(["0.8"], LabelReplacer("__MACH_NUMBER__"))
numTrainingSamples = 5

### VARIABLES ###
# MISC
base_config = "config.cfg"
nameObjs = ["DRAG"]
nameCons = ["LIFT", "MOMENT_Z"]
n_cons = len(nameCons)
config_tmpl_filename = "config_template.cfg"

# Extract from SU2 config ---------------------------------------------------- #
config = SU2.io.Config(base_config)

mesh_filename = config['MESH_FILENAME']   
mesh_out_filename = config['MESH_OUT_FILENAME']
solution_filename = config["SOLUTION_FILENAME"]
restart_adj_filename = config["RESTART_ADJ_FILENAME"]
solution_adj_filename = config["SOLUTION_ADJ_FILENAME"]
restart_lift_adj_filename = f"{restart_adj_filename[:-4]}_cl{restart_adj_filename[-4:]}"
restart_drag_adj_filename = f"{restart_adj_filename[:-4]}_cd{restart_adj_filename[-4:]}"
solution_lift_adj_filename = f"{solution_adj_filename[:-4]}_cl{solution_adj_filename[-4:]}"
solution_drag_adj_filename = f"{solution_adj_filename[:-4]}_cd{solution_adj_filename[-4:]}"

designparams = np.array(config['DV_VALUE_OLD'])

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
# pType_geometry = Parameter(["0.001"+",0.001"*(len(designparams)-1)], LabelReplacer("__X__"))

# switch from direct to adjoint mode and adapt settings
pType_direct = Parameter(["DIRECT"], LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"], LabelReplacer("__MATH_PROBLEM__"))
# switch default and deformed mesh
pType_mesh_filename_original = Parameter([mesh_filename], LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter([mesh_out_filename], LabelReplacer("__MESH_FILENAME__"))
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
def_command = "SU2_DEF config_template.cfg"
geo_command = "SU2_GEO config_template.cfg"
dir_command = "mpirun -n 4 SU2_CFD config_template.cfg"
adj_command = "mpirun -n 4 SU2_CFD_AD config_template.cfg"
dot_command = "mpirun -n 4 SU2_DOT_AD config_template.cfg"
max_tries   = 1

# DEFORMATION RUN
meshDeformationRun = ExternalRun("DEFORM", def_command, True)
meshDeformationRun.addConfig(config_tmpl_filename)
meshDeformationRun.addData(mesh_filename)
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_obj)
meshDeformationRun.addParameter(default)
meshDeformationRun.addExpected(f"{mesh_out_filename}")

# GEOMETRY RUN
geometryRun = ExternalRun("GEOMETRY", geo_command, True)
geometryRun.addConfig(config_tmpl_filename)
geometryRun.addData(f"DEFORM/{mesh_out_filename}")
geometryRun.addParameter(pType_adjoint)
geometryRun.addParameter(pType_mesh_filename_deformed)
geometryRun.addParameter(pType_obj)
geometryRun.addParameter(default)
geometryRun.addExpected("of_func.csv")
geometryRun.addExpected("of_grad.csv")

# DIRECT RUN
directRun = ExternalRun("DIRECT", dir_command, True)
directRun.addConfig(config_tmpl_filename)
directRun.addData(f"DEFORM/{mesh_out_filename}")
directRun.addParameter(pType_direct)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_obj)
directRun.addParameter(droPar)
directRun.addExpected("history.csv")
directRun.addExpected(f"{solution_filename}")

# ADJOINT DRAG
adjointRun= ExternalRun("ADJOINT_DRAG", adj_command, True)
adjointRun.addConfig(config_tmpl_filename)
adjointRun.addData(f"DEFORM/{mesh_out_filename}")
adjointRun.addData(f"DIRECT/{solution_filename}")
adjointRun.addParameter(pType_adjoint)
adjointRun.addParameter(pType_mesh_filename_deformed)
adjointRun.addParameter(pType_obj)
adjointRun.addParameter(droPar)
adjointRun.addExpected(f"{restart_drag_adj_filename}")

# DOT RUN
dotProductDragRun= ExternalRun("DOT_DRAG",dot_command, True)
dotProductDragRun.addConfig(config_tmpl_filename)
dotProductDragRun.addData(f"DEFORM/{mesh_out_filename}")
dotProductDragRun.addData(f"ADJOINT_DRAG/{solution_drag_adj_filename}")
dotProductDragRun.addParameter(pType_adjoint)
dotProductDragRun.addParameter(pType_mesh_filename_deformed)
dotProductDragRun.addParameter(pType_obj)
dotProductDragRun.addParameter(droPar)
dotProductDragRun.addExpected("of_grad.dat")

# ADJOINT LIFT
adjointRunCons = ExternalRun("ADJOINT_LIFT", adj_command, True)
adjointRunCons.addConfig(config_tmpl_filename)
adjointRunCons.addData(f"DEFORM/{mesh_out_filename}")
adjointRunCons.addData(f"DIRECT/{solution_filename}") 
adjointRunCons.addParameter(pType_adjoint)
adjointRunCons.addParameter(pType_mesh_filename_deformed)
adjointRunCons.addParameter(pType_constraints[0])
adjointRunCons.addParameter(droPar)
adjointRunCons.addExpected(f"{restart_lift_adj_filename}")

# DOT LIFT
dotProductLiftRun = ExternalRun("DOT_LIFT", dot_command, True)
dotProductLiftRun.addConfig(config_tmpl_filename)
dotProductLiftRun.addData(f"DEFORM/{mesh_out_filename}")
dotProductLiftRun.addData(f"ADJOINT_LIFT/{solution_lift_adj_filename}")
dotProductLiftRun.addParameter(pType_adjoint)
dotProductLiftRun.addParameter(pType_mesh_filename_deformed)
dotProductLiftRun.addParameter(pType_constraints[0])
dotProductLiftRun.addParameter(droPar)
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
fun.addGradientEvalStep(adjointRun)
fun.addGradientEvalStep(dotProductDragRun)

# Lift constraint function
cons_fun = Function("LIFT", "DIRECT/history.csv", LabeledTableReader(f"\"CL\""))
cons_fun.addInputVariable(x,"DOT_LIFT/of_grad.dat", TableReader(None,0,start=(1,0),end=(None,None)))
cons_fun.addValueEvalStep(meshDeformationRun) 
cons_fun.addValueEvalStep(directRun)
cons_fun.addGradientEvalStep(adjointRunCons)
cons_fun.addGradientEvalStep(dotProductLiftRun)

# Thickness constraint function
nameConsThickness = "AIRFOIL_THICKNESS"
cons_fun_geo = Function("AIRFOIL_THICKNESS","GEOMETRY/of_func.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\""))
cons_fun_geo.addInputVariable(x, "GEOMETRY/of_grad.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\"", rang=(1, None)))
cons_fun_geo.addValueEvalStep(meshDeformationRun)
cons_fun_geo.addValueEvalStep(geometryRun)        

# Setup driver
driver = DROScipyDriver()
driver.addObjective("min", fun, 1.0) # objective
driver.addLowerBound(cons_fun_geo, 0.12, 0.001) # thickness constraint
driver.addLowerBound(cons_fun, 0.28, 0.001) # lift constraint

driver.setWorkingDirectory("OPTIM")
driver.setEvaluationMode(True, 0.1) # Sets parallel evaluation mode
driver.setStorageMode(True, "DESIGN/DSN_")
driver.setFailureMode("SOFT")
driver.setNumTrainingSamples(numTrainingSamples)

his = open("optim.his", "w", 1)
driver.setHistorian(his)

driver.preprocess()
x = driver.getInitial()

# Optimization, SciPy -------------------------------------------------- #
import scipy.optimize

x = driver.getInitial()

options = {'disp': True, 'ftol': 1e-7, 'maxiter': 10}

optimum = scipy.optimize.minimize(driver.fun, x, method="SLSQP", jac=driver.grad,\
          constraints=driver.getConstraints(), bounds=driver.getBounds(), options=options)

his.close()

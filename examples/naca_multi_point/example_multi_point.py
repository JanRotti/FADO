# Shape optimization of airfoil with multipoint MACH
import SU2
from FADO import *
import subprocess
import os 

# PARALLELIZATION PARAMETERS
distr_num_procs = 3
distr_variable_marker = "__MACH_NUMBER__"
distr_variable_values = [0.8, 0.88, 0.76] #[0.8, 0.83, 0.86, 0.79, 0.75]

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
pType_objs = []
for nameObj in nameObjs:
    pType_obj = Parameter([nameObj], LabelReplacer("__FUNCTION__"))
    pType_objs.append(pType_obj)
pType_constraints = []
for nameCon in nameCons:
    pType_constraint = Parameter([nameCon], LabelReplacer("__FUNCTION__"))
    pType_constraints.append(pType_constraint)

distr_pType_variable = [
    Parameter([str(val)], LabelReplacer(distr_variable_marker)) for val in distr_variable_values
]

# Evaluations ---------------------------------------------------------- #
def_command = "SU2_DEF config_tmp.cfg"
geo_command = "SU2_GEO config_tmp.cfg"
dir_command = "mpirun -n 4 SU2_CFD config_tmp.cfg"
adj_command = "mpirun -n 4 SU2_CFD_AD config_tmp.cfg"
dot_command = "mpirun -n 4 SU2_DOT_AD config_tmp.cfg"
copyCommandDirect = f"cp ../DIRECT_%i/{restart_filename} {solution_filename}"
copyCommandAdjointDrag = f"cp ../ADJOINT_DRAG_%i/{restart_drag_adj_filename} {solution_drag_adj_filename}"
copyCommandAdjointLift = f"cp ../ADJOINT_LIFT_%i/{restart_lift_adj_filename} {solution_lift_adj_filename}"
max_tries = 1

# DEFORMATION RUN
meshDeformationRun = ExternalRun(f"DEFORM", def_command, True)
meshDeformationRun.addConfig(config_tmpl_filename)
meshDeformationRun.addData(mesh_filename)
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_restart_off)
meshDeformationRun.addParameter(pType_obj)
meshDeformationRun.addParameter(distr_pType_variable[0])

# GEOMETRY RUN
geometryRun = ExternalRun("GEOMETRY", geo_command, True)
geometryRun.addConfig(config_tmpl_filename)
geometryRun.addData(f"DEFORM/{mesh_out_filename}")
geometryRun.addParameter(pType_geometry)
geometryRun.addParameter(pType_adjoint)
geometryRun.addParameter(pType_mesh_filename_deformed)
geometryRun.addParameter(pType_restart_off)
geometryRun.addParameter(pType_obj)
geometryRun.addParameter(distr_pType_variable[0])

# DIRECT RUN
directRuns = []
for i in range(distr_num_procs):
    directRun = ExternalRun(f"DIRECT_{i}", dir_command, True)
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
    directRun.addParameter(distr_pType_variable[i])
    directRuns.append(directRun)
    
# COPY POST DIRECT RUN
copyRuns = []
for i in range(distr_num_procs):
    copyRun=ExternalRun(f"RESTART_{i}", copyCommandDirect % i, True)
    copyRun.addData(f"DIRECT_{i}/{restart_filename}")
    copyRuns.append(copyRun)  

# ADJOINT RUN
adjointRuns = []
for i in range(distr_num_procs):
    adjointRun= ExternalRun(f"ADJOINT_DRAG_{i}", adj_command, True)
    adjointRun.addConfig(config_tmpl_filename)
    adjointRun.addData(f"DEFORM/{mesh_out_filename}")
    adjointRun.addData(f"RESTART_{i}/{solution_filename}")  
    adjointRun.addParameter(pType_adjoint)
    adjointRun.addParameter(pType_mesh_filename_deformed)
    adjointRun.addParameter(pType_obj)
    adjointRun.addParameter(distr_pType_variable[i])
    if (restartAdj=="YES"):
        adjointRun.addParameter(pType_restart_on)
        adjointRun.addData(f"ADJOINT_DRAG_{i}/{restart_adj_filename}")
    else:
        adjointRun.addParameter(pType_restart_off)
    adjointRuns.append(adjointRun)

# COPY POST ADJOINT RUN
copyAdjRuns = []
for i in range(distr_num_procs):
    copyAdjRun=ExternalRun(f"RESTART_ADJOINT_DRAG_{i}", copyCommandAdjointDrag % i,True)
    copyAdjRun.addData(f"ADJOINT_DRAG_{i}/{restart_drag_adj_filename}")
    copyAdjRuns.append(copyAdjRun)

# DOT RUN
dotProductDragRuns = []
for i in range(distr_num_procs):
    dotProductDragRun= ExternalRun(f"DOT_DRAG_{i}",dot_command, True)
    dotProductDragRun.addConfig(config_tmpl_filename)
    dotProductDragRun.addData(f"DEFORM/{mesh_out_filename}")
    dotProductDragRun.addData(f"RESTART_ADJOINT_DRAG_{i}/{solution_drag_adj_filename}")
    dotProductDragRun.addParameter(pType_adjoint)
    dotProductDragRun.addParameter(pType_mesh_filename_deformed)
    dotProductDragRun.addParameter(pType_obj)
    dotProductDragRun.addParameter(pType_restart_off)
    dotProductDragRun.addParameter(distr_pType_variable[i])
    dotProductDragRuns.append(dotProductDragRun)

# ADJOINT LIFT
adjointRunConss = []
for i in range(distr_num_procs):
    adjointRunCons = ExternalRun(f"ADJOINT_LIFT_{i}", adj_command, True)
    adjointRunCons.addConfig(config_tmpl_filename)
    adjointRunCons.addData(f"DEFORM/{mesh_out_filename}")
    adjointRunCons.addData(f"RESTART_{i}/{solution_filename}") 
    adjointRunCons.addParameter(pType_adjoint)
    adjointRunCons.addParameter(pType_mesh_filename_deformed)
    adjointRunCons.addParameter(pType_constraints[0])
    if (restartAdj=="YES"):
        adjointRunCons.addParameter(pType_restart_on)
        adjointRunCons.addData(f"ADJOINT_LIFT_{i}/{restart_adj_filename}")
    else:
        adjointRunCons.addParameter(pType_restart_off)
    adjointRunCons.addParameter(distr_pType_variable[i])
    adjointRunConss.append(adjointRunCons)

# POST LIFT RUN
copyAdjConsCommands = []
for i in range(distr_num_procs):
    copyAdjConsRun=ExternalRun(f"RESTART_ADJOINT_LIFT_{i}", copyCommandAdjointLift % i,True)
    copyAdjConsRun.addData(f"ADJOINT_LIFT_{i}/{restart_lift_adj_filename}")
    copyAdjConsCommands.append(copyAdjConsRun)  

# DOT LIFT
dotProductConsRuns = []
for i in range(distr_num_procs):
    dotProductConsRun= ExternalRun(f"DOT_LIFT_{i}",dot_command, True)
    dotProductConsRun.addConfig(config_tmpl_filename)
    dotProductConsRun.addData(f"DEFORM/{mesh_out_filename}")
    dotProductConsRun.addData(f"RESTART_ADJOINT_LIFT_{i}/{solution_lift_adj_filename}")
    dotProductConsRun.addParameter(pType_adjoint)
    dotProductConsRun.addParameter(pType_mesh_filename_deformed)
    dotProductConsRun.addParameter(pType_constraints[0])
    dotProductConsRun.addParameter(distr_pType_variable[i])
    dotProductConsRun.addParameter(pType_restart_off)
    dotProductConsRuns.append(dotProductConsRun)

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
funs = []
for i in range(distr_num_procs):
    fun = Function(f"DRAG_{i}", f"DIRECT_{i}/history.csv", LabeledTableReader(f"\"CD\""))
    fun.addInputVariable(x, f"DOT_DRAG_{i}/of_grad.dat", TableReader(None,0,start=(1,0),end=(None,None)))
    fun.addValueEvalStep(meshDeformationRun) 
    fun.addValueEvalStep(directRuns[i])
    fun.addGradientEvalStep(copyRuns[i])
    fun.addGradientEvalStep(adjointRuns[i])
    fun.addGradientEvalStep(copyAdjRuns[i])
    fun.addGradientEvalStep(dotProductDragRuns[i])
    funs.append(fun)

# CONSTRAINT FUNCTION
cons_funs = []
for i in range(distr_num_procs):
    cons_fun = Function(f"LIFT_{i}", f"DIRECT_{i}/history.csv", LabeledTableReader(f"\"CL\""))
    cons_fun.addInputVariable(x, f"DOT_LIFT_{i}/of_grad.dat", TableReader(None,0,start=(1,0),end=(None,None)))
    cons_fun.addValueEvalStep(meshDeformationRun) 
    cons_fun.addValueEvalStep(directRuns[i])
    cons_fun.addGradientEvalStep(copyRuns[i])
    cons_fun.addGradientEvalStep(adjointRunConss[i])
    cons_fun.addGradientEvalStep(copyAdjConsCommands[i])
    cons_fun.addGradientEvalStep(dotProductConsRuns[i])
    cons_funs.append(cons_fun)

# Thickness constraint function
cons_fun_geo = Function("AIRFOIL_THICKNESS", "GEOMETRY/of_func.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\""))
cons_fun_geo.addInputVariable(x, "GEOMETRY/of_grad.csv", LabeledTableReader("\"AIRFOIL_THICKNESS\"", rang=(1, None)))
cons_fun_geo.addValueEvalStep(meshDeformationRun)
cons_fun_geo.addValueEvalStep(geometryRun)        

# Setup driver
driver = ScipyDriver()
for fun in funs:
    driver.addObjective("min", fun, 1.0, 1.0 / distr_num_procs)
driver.addLowerBound(cons_fun_geo, 0.12, 0.001)
for cons_fun in cons_funs:
    driver.addLowerBound(cons_fun, 0.28, 0.001)

driver.setWorkingDirectory("OPTIM")
driver.setEvaluationMode(True, 0.1)
driver.setStorageMode(True,"DESIGN/DSN_")
driver.setFailureMode("SOFT")

his = open("optim.his","w",1)
driver.setHistorian(his)

# Optimization, SciPy -------------------------------------------------- #
import scipy.optimize

driver.preprocess()
x = driver.getInitial()

options = {'disp': True, 'ftol': 1e-7, 'maxiter': 100}

optimum = scipy.optimize.minimize(driver.fun, x, method="SLSQP", jac=driver.grad,\
          constraints=driver.getConstraints(), bounds=driver.getBounds(), options=options)

his.close()
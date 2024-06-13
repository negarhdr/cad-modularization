import pathlib

from occwl.compound import Compound
from occwl.viewer import Viewer

# step_file_path = "/home/nh/CAD/data/fabwave/CAD16-24/Pipes/STEP/eedfd59d-92e0-488d-be6e-4c1c07b73ff3.stp"
# step_file_path = "/home/nh/CAD/data/assembly_joint/joint/45594_30b85d4c_0021_2.step"
# step_file_path = "/mnt/archive/common/datasets/CAD_datasets/Fusion360/Assembly/downloaded/assembly/99842_e4ee42a9/assembly.step"
step_file_path = "/mnt/archive/common/datasets/CAD_datasets/Fusion360/Assembly/downloaded/assembly/16550_e88d6986/assembly.step"
# "/home/nh/CAD/data/Runi/exported_CADAssistant/5500000040.step"
# Load everything from a STEP file as a single Compound, Returns a list of bodies from the step file
compound = Compound.load_from_step(pathlib.Path(__file__).resolve().parent.joinpath(step_file_path))
solids = list(compound.solids())
print('Number of Solids', len(solids))

for idx, solid in enumerate(solids):
    print('Solid {}:'.format(idx+1))
    print('{} Face'.format(len(list(solid.faces()))))
    print('{} Edge'.format(len(list(solid.edges()))))
    print('{} Shell'.format(len(list(solid.shells()))))
    print('----------------------------------------')

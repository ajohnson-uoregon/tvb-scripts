from tvb.simulator.plot.tools import *
from mpl_toolkits.mplot3d import Axes3D
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.simulator import monitors
from tvb.simulator.models import WilsonCowan, Generic2dOscillator, ReducedWongWang, JansenRit, Linear
from tvb.simulator.integrators import VODE, VODEStochastic
from tvb.simulator import simulator
import time

# zipped directory that contains connectivity/tractography info
conn_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/connectivity/DH_20120806_Connectivity/DHconn.zip"
conn = Connectivity(load_file=conn_fname)

conn.configure()
print("connectivity loaded")
# print(conn.summary_info)

plot_connectivity(connectivity=conn)
# pyplot.show()

# triangulated cortical surface mesh
ctx_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/cortex/DH_20120806_Surface_Cortex.zip"
# maps nodes from cortex mesh to brain regions
region_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_RegionMapping.txt"
rm = RegionMapping(load_file=region_fname)
# matlab matrix that describes how waves propagate to the surface
eeg_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_ProjectionMatrix.mat"
ctx = Cortex(load_file=ctx_fname,
    region_mapping_data=rm)
ctx.configure()
print("cortex loaded")

pyplot.figure()
ax = pyplot.subplot(111, projection='3d')
x,y,z = ctx.vertices.T
ax.plot_trisurf(x, y, z, triangles=ctx.triangles, alpha=0.1, edgecolor='k')
# pyplot.show()
print("cortex plot ready")

# unit vectors that describe the location of eeg sensors
sensoreeg_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_EEGLocations.txt"

sensorsEEG = SensorsEEG(load_file=sensoreeg_fname)
prEEG = ProjectionSurfaceEEG(load_file=eeg_fname)

fsamp = 1e3/1024.0 # 1024 Hz
mon = monitors.EEG(sensors=sensorsEEG, projection=prEEG,
    region_mapping=rm, period=fsamp)

sim = simulator.Simulator(
    connectivity=conn,
    # conduction speed: 3 mm/ms
    # coupling: linear - rescales activity propagated
    # stimulus: None - can be a spatiotemporal function

    # model: Generic 2D Oscillator - neural mass has two state variables that
    # represent a neuron's membrane potential and recovery; see the
    # mathematics paper for default values and equations; runtime 8016 s
    model=Linear(),
    # model: Wilson & Cowan - two neural masses have an excitatory/inhibitory
    # relationship; see paper for details; runtime 16031 s
    # model: Wong & Wang - reduced system of two non-linear coupled differential
    # equations based on the attractor network model; see paper; runtime 8177 s
    # model: Jansen & Rit - biologically inspired mathematical framework
    # originally conceived to simulate the spontaneous electrical activity of
    # neuronal assemblies, with a particular focus on alpha activity; had
    # some weird numpy overflow errors, only generated ~200 ms data; runtime 5929 s


    # integrator=VODEStochastic(),
    # initial conditions: None
    monitors=mon,
    surface=ctx,
    simulation_length=5.0 # ms
).configure()
print("sim configured")

print("sim running")
start = time.time()
result = sim.run()[0]
end = time.time()
print("sim done, took " + str(end-start) + " seconds")

print(result)
pyplot.figure()
time, data = result
pyplot.plot(time, data[:, 0, :, 0], 'k', alpha=0.1)
pyplot.show()

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import *
from mpl_toolkits.mplot3d import Axes3D
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
import time


conn_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/connectivity/DH_20120806_Connectivity/DHconn.zip"
conn = connectivity.Connectivity.from_file(conn_fname)

conn.configure()
print("connectivity loaded")
# print(conn.summary_info)

plot_connectivity(connectivity=conn)
# pyplot.show()


ctx_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/cortex/DH_20120806_Surface_Cortex.zip"
region_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_RegionMapping.txt"
eeg_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_ProjectionMatrix.mat"
ctx = cortex.Cortex.from_file(source_file=ctx_fname,
    region_mapping_file=region_fname,
    eeg_projection_file=eeg_fname)
ctx.configure()
print("cortex loaded")

pyplot.figure()
ax = pyplot.subplot(111, projection='3d')
x,y,z = ctx.vertices.T
ax.plot_trisurf(x, y, z, triangles=ctx.triangles, alpha=0.1, edgecolor='k')
# pyplot.show()
print("cortex plot ready")

sensoreeg_fname = "/home/annajo/git/tvb/tvb-data/tvb_data/berlinSubjects/DH_20120806/DH_20120806_EEGLocations.txt"

rm = RegionMapping.from_file(region_fname)
sensorsEEG = SensorsEEG.from_file(sensoreeg_fname)
prEEG = ProjectionSurfaceEEG.from_file(eeg_fname)

fsamp = 1e3/1024.0 # 1024 Hz
mon = monitors.EEG(sensors=sensorsEEG, projection=prEEG,
    region_mapping=rm, period=fsamp)

sim = simulator.Simulator(
    connectivity=conn,
    monitors=mon,
    surface=ctx,
    simulation_length=1000.0
).configure()
print("sim configured")

print("sim running")
start = time.time()
result = sim.run()[0]
end = time.time()
print("sim done, took " + str(end-start) + " seconds")
pyplot.figure()
time, data = result
pyplot.plot(time, data[:, 0, :, 0], 'k', alpha=0.1)
pyplot.show()

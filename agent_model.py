from images2gif import writeGif
import numpy as np
from numpy.random import *
from matplotlib.pyplot import *
from PIL import Image,ImageSequence
import sys,os,csv

#Default parameters:
#run_number = 0
#passive_signal_production = 0.5
#active_signal_production = 5.0
#factor_production = 0.1
#Dz=0.02;
#Rz=0.001;
#signal_thresh = 10.0
#factor_thresh = 10.0
#Dz_factor = 0.02
#Rz_factor = 0.001

# Which of the indexed sets of parameters to try
run_number = float(sys.argv[1])
# How much signal do bacteria produce when they're /below/ their activation threshold
passive_signal_production = float(sys.argv[2])
# How much signal do bacteria produce when they're /above/ their activation threshold
active_signal_production = float(sys.argv[3])
# How much factor (end result) do they produce when they're activated? (they never produce factor when inactive)
factor_production = float(sys.argv[4])
# Diffusion coefficient for signal (how fast the factor spreads through medium)
Dz=float(sys.argv[5])
# Extinction coefficient for signal (how fast it "disappears")
Rz=float(sys.argv[6])
# Activation threshold for bacteria's signal production
signal_thresh = float(sys.argv[7])
# Activation threshold for bacteria's factor production
factor_thresh = float(sys.argv[8])
# Diffusion & extinction coefficients for factor
Dz_factor = float(sys.argv[9])
Rz_factor = float(sys.argv[10])

# Make a string for the directory containing the parameters (in csv) and results (as gif)
directory = "./img/agent"+str(run_number)
# If the directory doesn't already exist, make it
if not os.path.exists(directory):
  os.makedirs(directory)
# Make a string for the filename of the results gif
filename = "./img/agent"+str(run_number)+"/agent"+str(run_number)+".gif"

# Write parameters to a csv in the directory for future reference
with open("./img/agent"+str(run_number)+"/agent"+str(run_number)+".csv","wb") as paramsFile:
  paramsWriter = csv.writer(paramsFile,delimiter=",")
  paramsWriter.writerow([sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10]])

# Steps in x and y directions, and time, used in diffusion equation approximation
dt=0.002;
dx=1;
dy=1;

# The map.csv is a matrix of either " " or "b"
# "b" denotes locations of bacteria
# Here, read csv, and put contents into a list of lists (called positions)
positions = []
with open("./map.csv","rb") as mapfile:
  mapreader = csv.reader(mapfile,delimiter=",")
  for row in mapreader:
    positions.append(row)
    
# Determine size of map (for diffusion) from the dimensions of the list of lists
nx=len(positions[0]);
ny=len(positions);
# Set the number of steps for the model to go through
nt=1000001;
# Generate two matrices, same dimensions as positions
# signals will contain concentration of signal at every location
# factors will contain concentration of factor at every location
# Set them to 0 everywhere for the beginning of the model.
signals = np.zeros([nx,ny,2]);
factors = np.zeros([nx,ny,2]);

# Also make a third matrix, and give it values of 0.001 (arbitrary)
# Will use this later as a threshold for factor being "present"
compare = np.zeros([nx,ny,2]);
compare[:,:,1] = 0.001

frame_names = []
# For each of nt steps, the model carries out diffusion, then increases the concentration of signals and factors at the locations of bacteria
for m in range(1,nt):
  # Diffusion equation approximation
  # Can talk about math if you want but it's not really necessary to know everythign that's going on
  # Pretty much, these two lines handle the diffusion of signal from one step to the next
  signals[:,:,0] = signals[:,:,1]
  signals[1:nx-1,1:ny-1,1]=signals[1:nx-1,1:ny-1,0]-dt*Rz*signals[1:nx-1,1:ny-1,0]+dt*Dz*((signals[2:nx,1:ny-1,0]-2*signals[1:nx-1,1:ny-1,0]+signals[0:nx-2,1:ny-1,0])/np.power(dx,2)+(signals[1:nx-1,2:ny,0]-2*signals[1:nx-1,1:ny-1,0]+signals[1:nx-1,0:ny-2,0])/np.power(dy,2));
  
  # Go thorugh each column of each row of positions
  for row in range(len(positions)):
    for col in range(len(positions[row])):
      # If the location in the list of lists is "b", there is an inactive bacterium there
      if positions[row][col]=="b":
        # Produce passive_signal_production level of signal
        signals[col,row,1] += passive_signal_production
      # If the location is "ba", there is an active bacterium there
      elif positions[row][col] == "ba":
        # Produce active_signal_production level of signal
        signals[col,row,1] += active_signal_production
      # If there is a "b" and the signal concentration there reaches the threshold
      if positions[row][col] == "b" and signals[col,row,1] >= signal_thresh:
        # Change to activated bacterium, "ba"
        positions[row][col] = "ba"
      # If there is a "ba" abd the signal concentration does not reach the threshold
      elif positions[row][col] == "ba" and signals[col,row,1] < signal_thresh:
        # Change to inactive bacterium "b"
        positions[row][col] = "b" 

  # Factor diffusion: same as signal diffusion above
  factors[:,:,0] = factors[:,:,1]
  factors[1:nx-1,1:ny-1,1]=factors[1:nx-1,1:ny-1,0]-dt*Rz_factor*factors[1:nx-1,1:ny-1,0]+dt*Dz_factor*((factors[2:nx,1:ny-1,0]-2*factors[1:nx-1,1:ny-1,0]+factors[0:nx-2,1:ny-1,0])/np.power(dx,2)+(factors[1:nx-1,2:ny,0]-2*factors[1:nx-1,1:ny-1,0]+factors[1:nx-1,0:ny-2,0])/np.power(dy,2));
  # Factor concentration doesn't activate or inactivate bacteria
  # It is only triggered when bacterium is in active state
  for row in range(len(positions)):
    for col in range(len(positions[row])):
      if positions[row][col] == "ba":
        factors[col,row,1] += factor_production
        
  # Save images for gif on multiples of 500
  if m % 1000 == 0:
    print(m)    
    fig = figure(figsize=(2.84,2),dpi=120)
    axis([0,ny,0,nx])
    pcolor(factors[:,:,1],vmin=-0.001,vmax=0.001,cmap='jet')
    axis('off')
    savefig("./img/agent"+str(run_number)+"/step"+str(m)+".png",bbox_inches="tight",pad_inches=-0.2)
    frame_names.append("./img/agent"+str(run_number)+"/step"+str(m)+".png")
    close()

frame_images = []
for i in frame_names:
  frame_images.append(Image.open(i))

# Check to see if right side of image has no factor by comparing with 0.001 matrix
# Note it in notable_runs.txt if that is the case.
#if np.amax(factors[:,250:,1]) < compare[:,250:,1].all():
#  with open("notable_runs.txt","a") as runs_file:
#    runs_file.write("\nagent"+str(run_number))
#  print("\nagent"+str(run_number)+" is an interesting run!! (^-^)")
  
# Make gif
writeGif(filename,frame_images,duration=0.01,repeat=True)

# Clean up by deleting pngs
for i in frame_names:
  os.remove(i)
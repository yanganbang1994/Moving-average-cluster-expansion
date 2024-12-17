import numpy as np
import random
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import BallTree
from qutip import *

# in version1, we consider the 2 body among the all atoms inside the cluster.



B_vector = [1,1,0]

def generate_cubic_lattice(Lx, Ly, Lz):
    """
    Generate a simple cubic lattice.
    Args:
        Lx, Ly, Lz: Dimensions of the lattice in the x, y, z directions.
    Returns:
        Array of lattice points.
    """
    lattice_points = []
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                lattice_points.append([x, y, z])
    return lattice_points

def gaussian_distribution(lattice_points, center, sigma,shell_radius):
    """
    Compute a Gaussian distribution over the lattice points.
    Args:
        lattice_points: List of lattice points (x, y, z).
        center: Center of the Gaussian (cx, cy, cz).
        sigma: Standard deviation of the Gaussian.
    Returns:
        List of probabilities corresponding to each lattice point.
    """
    R_c = shell_radius
    probabilities = []
    for point in lattice_points:
        distance = np.sqrt((np.array(point) - np.array(center))[0]**2+(np.array(point) - np.array(center))[1]**2+7*(np.array(point) - np.array(center))[2]**2)
        prob = np.exp(-(distance-R_c)**2 / (2 * sigma**2))
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    return probabilities / probabilities.sum()  # Normalize to sum to 1

def sample_points(lattice_points, probabilities, num_samples,repeat):
    """
    Sample points from the lattice based on given probabilities.
    Args:
        lattice_points: List of lattice points.
        probabilities: List of probabilities for each lattice point.
        num_samples: Number of points to sample.
    Returns:
        List of sampled lattice points.
    """
    indices = np.random.choice(len(lattice_points), replace = repeat, size=num_samples,  p=probabilities)
    return [lattice_points[i] for i in indices]
    
# Define dipole-dipole interactions (absolute)
def DD_interactions(u,v):
    r = np.linalg.norm(u-v)
    dot_product = np.dot(u-v,B_vector)
    magnitude_v1 = np.linalg.norm(u-v)
    magnitude_v2 = np.linalg.norm(B_vector)
    v1_safe = np.where(magnitude_v1 == 0, np.inf, magnitude_v1)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (v1_safe * magnitude_v2)
        
    # Avoid numerical issues that may push cos_theta slightly out of bounds
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    if r > 0:
        DDinteraction = (1-3*cos_theta**2)/r**3
        DD_safe = np.where(DDinteraction==0, np.inf, DDinteraction)
        return 1/abs(DDinteraction)
    else:
        return np.inf
 

# Define lattice size
Lx, Ly, Lz = 60, 60, 22
lattice_points = generate_cubic_lattice(Lx, Ly, Lz)

# Define Gaussian distribution parameters
center = (Lx/2, Ly/2, Lz/2)  # Center of the Gaussian
sigma = 30  # Standard deviation
shell_radius = 35

# Compute probabilities
probabilities = gaussian_distribution(lattice_points, center, sigma,shell_radius)

# Sample points
num_samples = 6000
sampled_atoms = sample_points(lattice_points, probabilities, num_samples, False)
sampled_atoms = np.array(sampled_atoms)

   
# Calculate sampled atom interactions strengths
tree = BallTree(sampled_atoms,metric = DD_interactions)

cluster_dim = 6
# initial state
initial_state = (basis(2,0)+basis(2,1)).unit()
state_list = [initial_state]*cluster_dim
psi0 = tensor(state_list)

# Interaction coefficients
Jp = 2*np.pi*104* np.ones(cluster_dim)
Jz = 0.0 * np.pi * np.ones(cluster_dim)

# Construct spin operators for 2-level system
# They follow the commutation relations required in PRL 113,195302. page 2
S_up = (sigmax()+1j*sigmay())/2
S_down = (sigmax()-1j*sigmay())/2
S_z = sigmaz()/2
S_x = sigmax()/2
S_y = sigmay()/2

# Construct spin operations for individual atoms in the cluster
sup_list, sdown_list, sz_list = [], [], []
for i in range(cluster_dim):
    op_list = [qeye(2)] * cluster_dim
    op_list[i] = S_up
    sup_list.append(tensor(op_list))
    op_list[i] = S_down
    sdown_list.append(tensor(op_list))
    op_list[i] = S_z
    sz_list.append(tensor(op_list))
    
sx_list, sy_list = [],[]
for i in range(cluster_dim):
    op_list = [qeye(2)] * cluster_dim
    op_list[i] = S_y
    sy_list.append(tensor(op_list))
    op_list[i] = S_x
    sx_list.append(tensor(op_list))
# Hamiltonian - Energy splitting terms

#for i in range(N):
#    H -= 0.5 * h[i] * sz_list[i]

def dd_interactions(n,m,cluster):
    u = cluster[n]
    v = cluster[m]
    r = np.linalg.norm(u-v)
    dot_product = np.dot(u-v,B_vector)
    magnitude_v1 = np.linalg.norm(u-v)
    magnitude_v2 = np.linalg.norm(B_vector)
    v1_safe = np.where(magnitude_v1 == 0, np.inf, magnitude_v1)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (v1_safe * magnitude_v2)

    # Avoid numerical issues that may push cos_theta slightly out of bounds
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    if r > 0:
        DDinteraction = (1-3*cos_theta**2)/r**3
        return DDinteraction
    else:
        return np.inf
result_Sx, result_Sy,result_C = np.zeros(100),np.zeros(100),np.zeros(100)
#Loop through all the sampeld atoms, solve the master equation for each cluster associated to the atoms
for n in range(num_samples):
    query_atom = sampled_atoms[n]

    # Abtain the N atoms with strongest interactions
    distances, indices = tree.query([query_atom], k=5)
    # Get the indices for the N connected atoms
    nearest_neighbours = sampled_atoms[indices]
    cluster = np.append(nearest_neighbours[0],[query_atom],axis=0)

    # Interaction terms
    H = 0
    l = cluster_dim - 1
    for n in range(cluster_dim):
        for m in range (n+1,cluster_dim):
            H += 0.5 * dd_interactions(n,m,cluster) * (0.5 * Jp[n] * (sup_list[n] * sdown_list[m]+ sup_list[m]*sdown_list[n]) +  Jz[n] * sz_list[n] * sz_list[m])

    times = np.linspace(0, 0.04 , 100)
    result = mesolve(H, psi0, times, [], [])
    states = [s * s.dag() for s in result.states]
    # Expectation value
    exp_sz = np.array(expect(states, sz_list))
    exp_sx = np.array(expect(states, sx_list))
    exp_sy = np.array(expect(states, sy_list))

    result_Sx+=exp_sx[:, -1]
    result_Sy+=exp_sy[:, -1]
    result_C += 2*np.sqrt(exp_sx[:, -1]**2+exp_sy[:, -1]**2)/num_samples
    
# Plot the expecation value
plt.plot(times, 2*np.sqrt(result_Sx**2+result_Sy**2)/num_samples, label=r"$\langle S_x^{2}+ \rangle+\langle S_y^{2}+ \rangle$")
plt.legend(loc="lower right")
plt.plot(times, result_C, label=r"$Contrast$")
plt.title("Dynamics of Spin Contrast")
plt.savefig('SpinDynamics_6000_40ms_12core_v1.pdf')

data = np.column_stack((times,result_C))
name = "outputv1"
repeat_num = 1
filestyle = ".csv"
filename = f"{name}_{repeat_num}_{filestyle}"
np.savetxt(filename, data, fmt="%f", delimiter=",", header="times(ms), Contrast", comments="")
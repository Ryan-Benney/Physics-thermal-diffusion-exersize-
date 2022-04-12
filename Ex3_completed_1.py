import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy import linalg
import time



start = time.time()

###############################################################################################################################################

#The alpha function is just defining the Alpha variable in the equation

def Alpha(n,N):

    h = 0.5/N
    k= (59/(450*7900))
    alpha = k*n/(h*h)
    #n is timestep
    #h is seperation
    #N is the number of nodes

    return alpha


#The below function constructs a Matrix with Dirichlet boundary conditions

def Matrix_Dirichlet (n,N):

    A = Alpha(n,N)
    n = 0.5/N

    centre_M = [(1+2*A)]
    side_M = [-A]
    Centre_M = np.repeat(centre_M, N-2)
    Side_M = np.repeat(side_M, N-2)
    diagonals = [Centre_M, Side_M, Side_M]
    Matrix = diags(diagonals, [0, 1, -1]).toarray()

    return Matrix

#This function constructs the vector that defines the Dirichlet boundary

def Boundary_Dirichlet (n,N):

    A = Alpha(n,N)

    B = np.zeros((N-2,1))
    B[0] = -A*(1274.15)
    B[N-3] = -A*(274.15)

    return B

#The initial temeprature across all nodes of the rods, which is at room temeprature

def U (N):

    U = np.repeat(294.15,N-2)
    U1 = np.resize(U,(N-2,1))

    return U1

#Matrix calculation 1 involves using the inverse matrix function from the scipy library to equate the future Node temperatures
#as a multiple of the inverse matrix, requiring an extra line of code

def Matrix_calculation (n,N,U):

    B = U - Boundary_Dirichlet(n, N)
    Matrix_calc = linalg.inv(Matrix_Dirichlet(n,N))
    temp_n_1 = Matrix_calc @ B

    return temp_n_1

#Matrix calculation 2 involves using the linear matrix function from the scipy library to calculate the future Node temperatures
#using the library function, saving an extra line of code compared to Matrix-calculation

def Matrix_calculation2 (n,N,U):

    B = U - Boundary_Dirichlet(n, N)
    temp_n_1 = linalg.solve(Matrix_Dirichlet(n,N), B)

    return temp_n_1

#This iteration function, takes the initial temperature and uses the Matrix calculation function to iterate through a specified
#number of times. It is written to stop iterating if the previous temperature vector is equal to the new one, as to not iterate
#forever

def Iteration (n,N,U,I):

    temp = []
    temp_n_1 = U
    temp_n_2 = U

    for i in range (I):

        if np.all(temp_n_1) != np.all(temp_n_2):

            break
        temp.append(temp_n_1)
        temp_n_2 = temp_n_1
        temp_n_1 = Matrix_calculation2 (n,N,temp_n_1)


    return np.array(temp)

#Graphing function reshapes the new list of temperature vectors made by the Iteration function, in order to turn the 3D numpy
#array into a 2D numpy array. Otherwise a contour plot wont be able to be made, since it requires a 2D array

def Graphing (n,N,U,I):

    A = Iteration(n, N, U, I)
    newA = A.reshape(I,(N-2)*1)

    return newA

#The plot function plots a contour graph, showing heat distribution as a change in time

def Plot (n,N,I):
    rod_length = np.linspace(0, 0.5,(N-2))
    Time_total = np.linspace(0, n*I, I)

    X,Y = np.meshgrid(rod_length, Time_total)
    Z = Graphing(n, N, U(N),I)

    plt.contourf(X, Y, Z, 20, cmap='coolwarm')
    plt.xlabel('Rod length/m')
    plt.ylabel('Total Time elapsed/s')
    plt.title("Heat distrubtion along a Rod as time evolves")
    cbar = plt.colorbar()
    cbar.set_label('Temperature/K', rotation=270);


    return

Plot(1, 100, 8000)

#The below function constructs a Matrix with Neumann boundary conditions
def Matrix_Neumann (n,N):

    A = Alpha(n,N)
    n = 0.5/N

    centre_M = [(1+2*A)]
    side_M = [-A]
    Centre_M = np.repeat(centre_M, N-2)
    Side_M = np.repeat(side_M, N-2)
    diagonals = [Centre_M, Side_M, Side_M]
    Matrix = diags(diagonals, [0, 1, -1]).toarray()
    Matrix[(N-3),(N-3)] = (1+A)

    return Matrix

#This function constructs the Boundary vector that defines the Dirichlet Boundary with the Neumann condition accounted

def Boundary_Neumann (n,N):

    A = Alpha(n,N)

    B = np.zeros((N-2,1))
    B[0] = -A*(1274.15)

    return B

#Matrix calculation 1 involves using the inverse matrix function from the scipy library to equate the future Node temperatures
#as a multiple of the inverse matrix, requiring an extra line of code. The neumann version uses the Neumann Matrix

def Matrix_calculation_Neumann (n,N,U):

    B = U - Boundary_Neumann(n, N)
    Matrix_calc = linalg.inv(Matrix_Neumann(n,N))
    temp_n_1 = Matrix_calc @ B

    return temp_n_1

#Matrix calculation 2 involves using the linear matrix function from the scipy library to calculate the future Node temperatures
#using the library function, saving an extra line of code compared to Matrix-calculation. The neumann version uses the Neumann Matrix

def Matrix_calculation2_Neumann (n,N,U):

    B = U - Boundary_Neumann(n, N)
    temp_n_1 = linalg.solve(Matrix_Neumann(n,N), B)

    return temp_n_1

#This iteration function, takes the initial temperature and uses the Matrix calculation function to iterate through a specified
#number of times. It is written to stop iterating if the previous temperature vector is equal to the new one, as to not iterate
#forever. The neumann iteration uses the Nuemann matrix calculation function.

def Iteration_Neumann (n,N,U,I):

    temp = []
    temp_n_1 = U
    temp_n_2 = U

    for i in range (I):

        if np.all(temp_n_1) != np.all(temp_n_2):

            break
        temp.append(temp_n_1)
        temp_n_2 = temp_n_1
        temp_n_1 = Matrix_calculation2_Neumann (n,N,temp_n_1)


    return np.array(temp)

#Graphing function reshapes the new list of temperature vectors made by the Iteration function, in order to turn the 3D numpy
#array into a 2D numpy array. Otherwise a contour plot wont be able to be made, since it requires a 2D array.

def Graphing_Neumann (n,N,U,I):

    A = Iteration_Neumann(n, N, U, I)
    newA = A.reshape(I,(N-2)*1)

    return newA

#Plots a contour graph using the Graphing Neumann function showing showing heat distribution as a change in time.

def Plot_Neumann (n,N,I):
    rod_length = np.linspace(0, 0.5,(N-2))
    Time_total = np.linspace(0, n*I, I)

    X,Y = np.meshgrid(rod_length, Time_total)
    Z = Graphing_Neumann(n, N, U(N),I)

    plt.contourf(X, Y, Z, 20, cmap='coolwarm')
    plt.xlabel('Rod length/m')
    plt.ylabel('Total Time elapsed/s')
    plt.title("Heat distrubtion along a Rod as time evolves")
    cbar = plt.colorbar()
    cbar.set_label('Temperature/K', rotation=270);

    return

Plot_Neumann(1, 1000, 10000)

################################################################################################################################################




end = time.time()

print("--- %s seconds ---" % (start-end))

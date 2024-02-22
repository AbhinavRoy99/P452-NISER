import math
import numpy as np
import scipy
from scipy import sparse


#Fixed Point Method
def fixed_point_method(func, guess, tolerance=1e-5, max_iterations=1000):
	iterations = []
	values = []
	guesses = []
	for i in range(max_iterations):
		next_guess = func(guess)
		if abs(next_guess - guess) < tolerance:
			return next_guess, iterations, values, guesses
		guess = next_guess
		iterations.append(i)
		guesses.append(guess)
		values.append(func(guess))
	return None, iterations, values, guesses


#Simpson's rule
def simpsons_rule(function, lower_limit, upper_limit, num_intervals):
	# Calculate the width of each interval
	interval_width = (upper_limit - lower_limit) / num_intervals

	# Generate the x-values for each interval
	x_values = np.linspace(lower_limit, upper_limit, num_intervals+1)

	# Calculate the y-values for each x-value
	y_values = function(x_values)

	# Apply Simpson's rule formula
	result = (interval_width / 3) * (y_values[0] 
									+ 4 * np.sum(y_values[1:-1:2]) 
									+ 2 * np.sum(y_values[2:-1:2]) 
									+ y_values[-1])
	return result


# Gaussian quadrature
def gaussian_quadrature(f, a, b, n):
	# Compute the sample points and weights from legendre polynomials
	x, w = np.polynomial.legendre.leggauss(n)
	# Change of variables
	t = 0.5 * (x + 1) * (b - a) + a
	return np.sum(w * f(t)) * 0.5 * (b - a)


#RK4 method for dy/dx
def RK4_dydx(f, x0, y0, x1, h):
	#take dy/dx as f(x,y)
	n = int((x1 - x0) / h)
	x = [x0]
	y = [y0]
	for i in range(n):
		xi = x[i]
		yi = y[i]

		x.append(xi + h)

		k1 = h * f(xi, yi)
		k2 = h * f(xi + 0.5*h, yi + 0.5*k1)
		k3 = h * f(xi + 0.5*h, yi + 0.5*k2)
		k4 = h * f(xi + h, yi + k3)

		y.append(yi + ((k1 + 2*k2 + 2*k3 + k4)/6))
		
	return y, x


#Crank Nicolson method for 1D heat equation
def crank_nicolson_1d(M, N, alpha, u_initial, T, L):
	x0, xL = 0, L
	dx = (xL - x0)/(M-1)
	t0, tF = 0, T 
	dt = (tF - t0)/(N-1)

	a0 = 1 + 2*alpha
	c0 = 1 - 2*alpha

	xspan = np.linspace(x0, xL, M)
	tspan = np.linspace(t0, tF, N)

	# Create the main diagonal for the left-hand side matrix with all elements as a0
	maindiag_a0 = a0*np.ones((1,M))

	# Create the off-diagonal for the left-hand side matrix with all elements as -alpha
	offdiag_a0 = (-alpha)*np.ones((1, M-1))

	# Create the main diagonal for the right-hand side matrix with all elements as c0
	maindiag_c0 = c0*np.ones((1,M))

	# Create the off-diagonal for the right-hand side matrix with all elements as alpha
	offdiag_c0 = alpha*np.ones((1, M-1))

	# Create the left-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	a = maindiag_a0.shape[1]

	# Create a list of the diagonals
	diagonalsA = [maindiag_a0, offdiag_a0, offdiag_a0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	A = sparse.diags(diagonalsA, [0,-1,1], shape=(a,a)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	A[0,1] = (-2)*alpha
	A[M-1,M-2] = (-2)*alpha

	# Create the right-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	c = maindiag_c0.shape[1]

	# Create a list of the diagonals
	diagonalsC = [maindiag_c0, offdiag_c0, offdiag_c0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	Arhs = sparse.diags(diagonalsC, [0,-1,1], shape=(c,c)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	Arhs[0,1] = 2*alpha
	Arhs[M-1,M-2] = 2*alpha

	#nitializes matrix U
	U = np.zeros((M, N))

	#Initial conditions
	U[:,0] = u_initial(xspan)

	#Boundary conditions
	f = np.arange(1, N+1)
	U[0,:] = 0
	f = U[0,:]
	
	g = np.arange(1, N+1)
	U[-1,:] = 0
	g = U[-1,:]
	
	#k = 1
	for k in range(1, N):
		ins = np.zeros((M-2,1)).ravel()
		b1 = np.asarray([4*alpha*dx*f[k], 4*alpha*dx*g[k]])
		b1 = np.insert(b1, 1, ins)
		b2 = np.matmul(Arhs, np.array(U[0:M, k-1]))
		b = b1 + b2  # Right hand side
		U[0:M, k] = np.linalg.solve(A,b)  # Solving x=A\b
	
	return (U, tspan, xspan)


	######

def printhash():
	print('\n ######################################################################################################## \n')

def crossmat(a,b):
	p_ab = [[0]*len(b[0]) for j in range(len(a))]
	
	for i in range(0,len(a)):
		for j in range(0,len(b[0])):
			for k in range(0,len(b)):
				p_ab[i][j] = p_ab[i][j] + a[i][k]*b[k][j]
	return p_ab

def pivotmat(M):
	m = len(M)                                                                                                                                                                                           
	id_mat = [[float(i ==j) for i in range(m)] for j in range(m)]
	
	for j in range(m):
		row = max(range(j, m), key=lambda i: abs(M[i][j]))
		if j != row:                                                                                                                                                                          
			id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

	return id_mat

def LUdecomposition(A,P):
	n = len(A)                                                                                                                                                                                                                 
	L = [[0] * n for i in range(n)]
	U = [[0] * n for i in range(n)]                                                                                                                                                     
	PA = crossmat(P,A)                                                                                                                                                                                                                     
	for j in range(n):                                                                                                                                                                                                  
		L[j][j] = 1                                                                                                                                                                                      
		for i in range(j+1):
			s1 = sum(U[k][j] * L[i][k] for k in range(i))
			U[i][j] = PA[i][j] - s1
		
		for i in range(j, n):
			s2 = sum(U[k][j] * L[i][k] for k in range(j))
			L[i][j] = (PA[i][j] - s2) / U[j][j]
	
	print('\n The lower triangle matrix is: \n')
	for row in L:
		print(row)
	
	print('\n The upper triangle matrix is: \n')
	for row in U:
		print(row)
	return L, U

def printeq(l,k,x,key):
	print('\n',key,'system of linear equations is: \n')
	for i in range(0,len(l)):
		print(l[i],'(',x,i+1,')',k[i])

def forwardsub(l,k):
	k1 = []
	i=0
	for i in range(0,len(l)):
		k1.append(k[i][0])
		for j in range(0,i):
			k1[i]= k1[i]-(l[i][j]*k1[j])
		k1[i]= k1[i]/l[i][i]
	
	m, n = len(l), 1
	k2 = [[0] * n for i in range(m)]
	
	for i in range(0,len(l)):
		k2[i][0]=k1[i]
	
	return k2

def backwardsub(l,k):
	m, n = len(l), 1
	k1 = [[0] * n for i in range(m)]
	k2 = [[0] * n for i in range(m)]
	
	for i in range(0,len(l)):
		k2[i][0]=k[i][0]
	
	for i in range(len(l)-1,-1,-1):
		for j in range(i+1,len(l)):
			k2[i][0]= k2[i][0]-(l[i][j]*k1[j][0])
		
		k1[i][0]= k2[i][0]/l[i][i]
	
	return k1

# Function to calculate the RK4 step
def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5*h, y + 0.5*k1)
    k3 = h * f(x + 0.5*h, y + 0.5*k2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


#Regula Falsi Method
def regula_falsi(f,a,b,tol):
	print('Regula Falsi Method')
	print('Given Interval:')
	print (a, f(a))
	print (b, f(b))
	root_estimates = []
	f_estimate = []

	if f(a) * f(b) >= 0:
		print("Wrong interval!")
		print("You have not assumed right a and b!")
		print("Enter new values of a and b: \n")
		return regula_falsi(f, float(input('Enter x0: ')), float(input('Enter x1: ')), tol)

	c = a
	root_estimates.append(c)
	f_estimate.append(f(c))
	print("Interval Accepted:")
	print (a, f(a))
	print (b, f(b))

	while abs(f(c)) > tol:
		#point touching x axis
		c = (b - ((b - a) * f(b))/ (f(b) - f(a)))
		root_estimates.append(c)
		f_estimate.append(f(c))
		# Find root
		if f(c) * f(a) < 0:
			b = c
		else:
			a = c
	return c, root_estimates, f_estimate



# Newton-Raphson method
def newton_raphson(f, df, x, eps):
	root_estimates = [x]
	f_estimate = [f(x)]
	while abs(f(x)) > eps:
		x = x - f(x)/df(x)
		root_estimates.append(x)
		f_estimate.append(f(x))
	return x, root_estimates, f_estimate

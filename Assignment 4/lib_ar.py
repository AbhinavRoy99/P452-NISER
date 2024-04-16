import math
import numpy as np
import scipy
from scipy import sparse
import lib_ar as toolsar
import scipy.stats as stats


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
	x, w = np.polynomiaar.legendre.leggauss(n)
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


def gaussjordan(l,k):
	x=len(l)
	y=len(l[0])
	
	
	for ok in range(0,x):
		

		
		for i in range(ok+1, x):
			if l[i][ok]>l[ok][ok]:
				for j in range(ok,x):
					l[ok][j],l[i][j]=l[i][j],l[ok][j]
				k[ok],k[i]=k[i],k[ok]
		
		t = l[ok][ok]
		
		if t != 0:
			k[ok][0] = k[ok][0]/t
			for a in range(ok,y):
				l[ok][a] = l[ok][a]/t
				
		for a in [ x for x in range(0,x) if x!=ok ]:
			lok=l[a][ok]
			k[a][0] = k[a][0] - (lok*k[ok][0])
			for b in range(0,y):
				l[a][b] = l[a][b] - (lok*l[ok][b])
	
	for a in range(0,x):
		for b in range(0,y):
			l[a][b] = round((l[a][b]),3)

	for a in range(0,x):
		k[a][0] = round((k[a][0]),3)
		
	print('\n System of linear equations after Gauss Jordan Elimination is: \n')
	for i in range(0,len(l)):
		print(l[i],'(','X',i+1,')',k[i])
	
	print('\n The answer matrix X is: \n',k)



def printeq(l,k,x,key):
	print('\n',key,'system of linear equations is: \n')
	for i in range(0,len(l)):
		print(l[i],'(',x,i+1,')',k[i])

def matrixtranspose(l):
	m,n = len(l),len(l)
	lt=[[0] * n for i in range(m)]
	
	for i in range(0,len(l)):
		for j in range(0,len(l)):
			lt[i][j]=l[j][i]
	
	return lt

def checksym(l):
	def check(l):
		lt=toolsar.matrixtranspose(l)
		for i in range(0,len(l)):
			for j in range(0,len(l)):
				if lt[i][j] != l[j][i]:
					return False
		return True
	if (check(l)):
		print('\n Yes the given matrix is symmetric.')
	else:
		print('\n No the given matrix is not symmetric.')

def CholeskyD(content):
	
	print('\n The matrix to be decomposed is: \n')
	for row in content:
		print(row)
	
	n=len(content)
		
	
	for j in range(0,n):
		for i in range(0,n):
			
			s=0
			
			if j == i:
				
				for k in range(0,j):
					s = s + (content[i][k]**2)
					
				content[i][i] = math.sqrt(content[i][i] - s)
				
				
				
			s=0
			
			if j < i:
				
				for k in range(0,j):
					s = s + (content[i][k] * content[j][k])
				
				if(content[j][j] != 0):
					content[i][j] = int(content[i][j] - s) / content[j][j]
					
				
		
			
			
			if j > i:
				
				content[i][j] = 0.0
			 
				
				
	for a in range(0,n):
		for b in range(0,n):
			content[a][b] = round((content[a][b]),1)
	
	print('\n The decomposed matrix is: \n')
	for row in content:
		print(row)   

	
	contenttranspose = toolsar.matrixtranspose(content)
	print('\n The decomposed matrix transpose is: \n')
	for row in contenttranspose:
		print(row)
	return content,contenttranspose
	
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


def diagonaldom(l,k):
	lnew=[[0] * len(l) for i in range(len(l))]
	knew=[0] * len(l)
	for i in range(len(l)):
		for j in range(len(l)):
			lnew[i][j]=l[i][j]
			knew[i]=k[i]
	swapc=0
	for i in range(len(l)):
		summ=0
		for t in range(len(l)):
			summ = summ + abs(lnew[i][t])
		for j in range(len(l)):
			if abs(lnew[i][j])> (summ-l[i][j]):
				lnew[i],lnew[j] = lnew[j], lnew[i]
				knew[i], knew[j] = knew[j], knew[i]
	
	toolsar.printeq(lnew,knew,'X','The diaonally dominant')
	return lnew,knew



def arrclose(x,y,tol):
	count=0
	if len(x)==len(y):
		for i in range(0,len(x)):
			if (abs(x[i]-y[i]))/abs(y[i]) < tol:
				count =count+1
			else:
				return False
	if count==len(x):
		return True



def JacobiM(a,b,it,tol):
	print("\n Jacobi Calculations: \n")
	
	x=[1]*len(a)
	newx=[1]*len(a)
	
	
	for i in range(0,it):
		t=0
		for t in range(0,len(x)):
			x[t] = newx[t]
		
		newx = toolsar.jac(a,b,newx)
	
		print(i+1,".",newx)
	
		z = toolsar.arrclose(newx,x,tol)
		if z == True:
			print('\n The answer is: \n',newx,"\n after",i+1,"iterations.")
			break
		else:
			continue

def jac(a,b,x):
	for i in range(0, len(a)):
		d=b[i]
		for j in range(0, len(a)):  
			if (j != i):
				d = d - (a[i][j]*x[j])
		
		x[i] = d / a[i][i]
		
	return x

def gsm(a,x,b):
						

	for j in range(0, len(a)):               
		  
		d=b[j]
		
		for i in range(0, len(a)):     
			if(j != i):
				d = d - (a[j][i]*x[i])
		
		x[j] = d / a[j][j]
		
	return x

def GaussSeidel(a,x,b,it,tol):
	print("\n Gauss Seidel Calculations: \n")
	
	y=[]
	
	for i in range(0,len(x)):
		y.append(x[i])
	
	for i in range(0,it):
		t=0
		for t in range(0,len(x)):
			y[t] = x[t]
	
		x = toolsar.gsm(a,x,b)
	
		print(i+1,".",x)
	
		z = toolsar.arrclose(x,y,tol)
		if z == True:
			print('\n The answer is: \n',x,"\n after",i+1,"iterations.")
			break
		else:
			continue
			



def Bracket(a,b,func):
	
	while (func(a)*func(b)>=0):
		if (abs(func(a)) < abs(func(b))):
			a = a - 1.5*(b-a)
			print(a,b)
			
			
		if (abs(func(a)) > abs(func(b))):
			b = b + 1.5*(b-a)
			print(a,b)
	
	return (a,b)

			

def Bisect(a,b,e,d,func):
	c=0
	t=0
	print(t,">>",a,b,c)
	
		
	while abs(a-b)>e:
		while abs(func(a))>d:
			c=(a+b)/2
			
			
			if abs(func(c))<=d:
				print(t+1,">>",a,b,c)
				print("the solution is",'%10.4E' %c,"in",t+1,"iterations.")
				return c
			else:  
				if (func(a)*func(c))<0:
					b=c
					t=t+1
					print(t,">>",a,b,c)
					
				else:
					a=c
					t=t+1
					print(t,">>",a,b,c)


def RegulaFalsi(a,b,e,d,fn):
	c0=0
	c1=1
	t=0
	print(t,">>",a,b,c0,c1)
	
	while abs(c1-c0)>e:
		while abs(fn(a))>d:
			c1 = b-((b-a)*fn(b))/(fn(b)-fn(a))
			
			if abs(fn(c1))<=d:
				print(t+1,">>",a,b,c0,c1)
				print("the solution is",c1,"in",t+1,"iterations.")
				return c1
			else:
				if (fn(a)*fn(c1))<0:
					b=c1
					c0=c1
					t=t+1
					print(t,">>",a,b,c0,c1)
					
				else:
					a=c1
					c0=c1
					t=t+1
					print(t,">>",a,b,c0,c1)


	
	

def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=100):
	x = np.array(x0, dtype=float)  # Initial guess
	r = b - np.dot(A, x)  # Initial residual
	p = r.copy()  # Initial search direction
	r_r = np.inner(r, r)
   
	for _ in range(max_iter):
		Ap = np.dot(A, p)
		alpha = r_r / np.inner(p, Ap)
		x += alpha * p
		r -= alpha * Ap

		r_r_next = np.inner(r, r)

		if np.linalg.norm(r) < tol: 
			break

		p = r + (r_r_next / r_r) * p
		r_r = r_r_next

	return x


def calculate_norm(r):
	"""
	Calculates the Euclidean norm (L2 norm) of a vector r and returns the Euclidean norm of the vector r.
	"""
	norm_squared = sum(element ** 2 for element in r)
	norm = norm_squared ** 0.5
	return norm 
	

def matrix_A_ij(x):
	"""
	Calculate the matrix-vector product Ax for the given vector x.
	Returns The result of the matrix-vector product Ax.
	"""
	m = 0.2
	N = len(x)
	delta = 1.0 / N
	result = np.zeros_like(x)
	
	for i in range(N):
		result[i] += (delta + m) * x[i]
		result[i] -= 2 * delta * x[i]
		result[i] += delta * x[(i + 1) % N]  # Periodic boundary condition
		result[i] += delta * x[(i - 1) % N]  # Periodic boundary condition
		result[i] += m ** 2 * delta * x[i] 
	#print(result)
	return result

def conjugate_fly(matrix_A_ij, b, x0, tol=10**(-6), max_iter=100):
	"""
	Conjugate Gradient method for solving linear systems Ax = b.
	Returns: The approximate solution vector x and the List of residue norms at each iteration step.
	"""
	it = 0
	x = x0.copy()  # Initial guess
	r = b - matrix_A_ij(x)  # Initial residual
	p = r.copy()  # Initial search direction
	residue_norms = [np.linalg.norm(r)]  # List to store residue norms

	for k in range(max_iter):
		Ap = matrix_A_ij(p)
		alpha = np.dot(r, r) / np.dot(p, Ap)
		x += alpha * p
		r -= alpha * Ap
		
		beta = np.dot(r, r) / np.dot(r - alpha * Ap, r - alpha * Ap)
		p = r - alpha * Ap + beta * p

		residue_norm = np.linalg.norm(r)
		residue_norms.append(residue_norm)
		it= it+1
		if residue_norm < tol:
			break

	return x, residue_norms, it


def conjugate_inv(matrix_A_ij, b, x0, tol=10**(-6), max_iter=100):
	N = len(b)
	inverse_columns = []
	
	for i in range(N):
		# Create the right-hand side vector for solving Ax = e_i
		ei = np.zeros(N)
		ei[i] = 1
		
		# Solve the equation Ax = e_i using Conjugate Gradient method
		x, _, _ = conjugate_fly(matrix_A_ij, ei, x0, tol, max_iter)
		
		# Append the solution (column of the inverse matrix) to the list
		inverse_columns.append(x)
	
	# Stack the columns of the inverse matrix horizontally to form the complete inverse matrix
	A_inv = np.column_stack(np.round(inverse_columns,4))
	return A_inv


def power_method(A, num_simulations: int):
	n = A.shape[0]
	
	# Step 1: Initialize a random vector
	v = np.random.rand(n)
	
	# Step 2: Power method iterations
	for _ in range(num_simulations):
		# Multiply v by the matrix
		Av = np.dot(A, v)
		
		# Normalize Av
		v = Av / np.linalg.norm(Av)
		
	# Step 3: Calculate the eigenvalue
	eigenvalue = np.dot(v, np.dot(A, v)) / np.dot(v, v)
	
	return eigenvalue


def gram_schmidt(A):
	Q = np.zeros_like(A)
	R = np.zeros((A.shape[1], A.shape[1]))

	for k in range(A.shape[1]):
		R[k, k] = np.linalg.norm(A[:, k])
		Q[:, k] = A[:, k] / R[k, k]
		for j in range(k + 1, A.shape[1]):
			R[k, j] = np.dot(Q[:, k], A[:, j])
			A[:, j] = A[:, j] - R[k, j] * Q[:, k]
	return Q, R

def qr_factorization(A, num_simulations: int):
	for _ in range(num_simulations):
		Q, R = gram_schmidt(A)
		A = R @ Q
	return np.diag(A)

def mlcg(a, m, seed=1):
	while True:
		seed = (a * seed) % m
		yield seed / m

def monte_carlo_integration(f, a, b, n, gen):
	sum = 0.0
	for _ in range(n):
		x = a + (b - a) * next(gen)  # Scale the random number to the interval [a, b]
		sum += f(x)
	return (b - a) * sum / n


# Monte Carlo integration with importance sampling
def monte_carlo_importance_sampling(f, p, inverse_cdf_p, n):
    samples = inverse_cdf_p(np.random.uniform(0, 1, n))
    weights = f(samples) / p(samples)
    return np.mean(weights), np.var(weights)

# Perform F-test
def F_test(A, B, var_A, var_B):
	F = var_A / var_B
	df1 = len(A) - 1
	df2 = len(B) - 1
	p_value_F = 1 - stats.f.cdf(F, df1, df2)
	return F, p_value_F

# Perform t-test
def t_test(mean_A, mean_B, std_A, std_B, A, B):
	t = (mean_A - mean_B) / np.sqrt(std_A**2/len(A) + std_B**2/len(B))
	p_value_t = 2 * (1 - stats.t.cdf(np.abs(t), len(A) + len(B) - 2))
	return t, p_value_t
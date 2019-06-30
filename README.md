# emarkjs
Javascript library with various mathematical functions and also including a simple multilayer perceptron

# Functions 
## Linear Algebra
- init_vector(number_of_elements) -> return vector with specified number of elements
- "class" vector:
  - n : number of elements
  - data : array 
- "class" matrix:
  - rows : number of rows
  - cols : number of columns
  - data : array
- init_matrix(number_of_rows,number_of_columns) -> return a matrix with specified number of rows and columns
- ones_vector(number_of_elements) -> return a vector with specified number of elements with all elements initialized in 1
- ones_matrix(number_of_rows,number_of_cols) -> return a matrix with specified number of rows and columns with all elements initialized in 1
- unit_vector(number_of_elements,position) -> return a unitary vector with the specified number of elements and all elements are initialized in 0 except for the position specified which is initialized in 1
- identity_matrix(dimension) -> return an identity matrix of dimension specified
- random_vector(number_of_elements) -> return a vector with the specified number of elements and all elements are initialized randomly
- random_matrix(number_of_rows,number_of_columns) -> return a matrix with the specified dimensions and all elements are initialized randomly
- str_2_vector(string,number_of_elements) -> return a vector with the specified number of elements and all elements are specified in the input string (eg. "1 2 3") 
- str_2_matrix(string,rows,columns) -> return a matrix with the specified dimensions with all elements specified in the input string (eg. "1 2;3 4")
- print_vector(vector) -> print the input vector
- print_matrix(matrix) -> print the input matrix
- sum_vector(vector_1,vector_2) -> return the sum of input vectors
- diff_vector(vector_1,vector_2) -> return the difference of input vectors (in the order they were input)
- dot_vector(vector_1,vector_2) -> return the dot product of the input vectors
- prod_vector(number,vector) -> multiplies each element of the vector times the input number and return the resulting vector
- norm_vector(vector) -> return the norm of the input vector
- sum_matrix(matrix_1,matrix_2) -> return the sum of input matrices
- diff_matrix(matrix_1,matrix_2) -> return the difference of input matrices
- dot_matrix(matrix_1,matrix_2) -> return the matricial product of input matrices
- prod_matrix(matrix_1,matrix_2) -> multiplies element by element of both matrices and return the resulting matrix
- prod_num_matrix(number,matrix) -> multiplies each element of the input matrix times the input number and return the resulting matrix
- trans_matrix(matrix) -> return the transpose of the input matrix
- lu_fact_matrix(matrix) -> factorize the input matrix into an upper triangular matrix (U) and a lower triangular matrix(L) and return an array containing both matrices (eg. [L U])
- det_matrix(matrix) -> return the determinant of the input matrix
- inverse_matrix(matrix) -> return the inverse matrix of the input matrix
- qr_fact_matrix(matrix) -> factorize the input matrix into an orthogonal matrix (Q) and an upper triangular matrix (R) using the Schmidt orthogonalization method
- eigen_matrix(matrix,N) -> Uses the QR factorization method to iteratively calculate the eigenvectors of the input matrix. N is the number of iterations
- solve_linear_system(matrix,vector) -> return the vector solution of the linear system conformed by the input matrix and the input vector
## Complex Numbers
- init_complex(real,imaginary)
- sum_complex(complex_1,complex_2)
- diff_complex(complex_1,complex_2)
- prod_complex(complex_1,complex_2)
- conjugate_complex(complex)
- norm_complex(complex)
- arg_complex(complex)
- exp_complex(complex)
- fft(data) -> return the Fast Fourier Transform of data which is an array of complex numbers
## Neural Network
- sigmoid_matrix(matrix)
- init_layer(inputs,outputs)
- "class" layer:
  - n_inputs : number of inputs
  - n_outputs : number of outputs
  - w : Weight matrix
  - forward(input)
  - print()
  - backward(error)
  - update(alpha)
-init_mlp(structure)
- "class" mlp:
  - struct : structure of the network, number of neuron in each layer (eg. [2 3 5])
  - n_layers : number of layers
  - layers : array containing all layers of the network
  - forward(input)
  - print()
  - backward(error)
  - update(alpha)
  - train(input_set,output_set,alpha,N)
## Non-Linear Equation
- newton_method(function,derivate_of_function,x0,N)
- bisection_method(function,a,b,N)
## Calculus
- derivate_function(function,x0,h)
- integral_function(function,a,b,N)
## Statistics
- average(vector)
- std_deviation(vector)
- combinatory(n,k)
- factorial(n)
- linear_regression(vector_1,vector_2)
- gaussian_distribution(x,average,standard_deviation)
- binomial_distribution(x,p,N)
## Graph Tool
- init_graph(context,WIDTH,HEIGHT,xlim,ylim)
- "class" graph:
  - ctx : context
  - h : HEIGHT
  - w : WIDTH
  - xlim : limits in the x axis
  - ylim : limits in the y axis
  - hx : conversion factor for the x axis
  - hy : conversion factor for the y axis
  - draw_bacground(Nx,Ny)
  - draw_arrow(initial,final)
  - plot(data_x,data_y)
  - clear()

import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.animation as animation
from matplotlib.patches import Circle

def newton(f, f_prime_x, f_prime_y, dx, dy, x0, y0, tol = 1.e-6):
    """ Uses Newton's method to find a root (x, y) of a function of two variables f
    Parameters:
    -----------
    - f: function f(x, y)
    - f_prime_x: function f_x(x, y)
    - f_prime_y: function f_y(x, y)
    - x0: initial x guess 
    - y0: initial y guess
    - tolerance: Returns when |f(x, y)| < tol
        
    Returns:
    --------
    - (x, y): final iterate
        
    Raises:
    -------
    Warning: if number of iterations exceed MAX_STEPS
    """
    MAX_STEPS = 200
    
    x = x0
    y = y0
    
    """
    Initial loop for Newton's method
    --------------------------------
    for k in range(1, MAX_STEPS + 1):
        x = x - f(x, y) / f_prime_x(x)
        y = y - f(x, y) / f_prime_y(y)
        x_array.append((x, y))
        if np.abs(f(x, y)) < tol:
            break
    """
    for k in range(1, MAX_STEPS + 1):
        matrix = np.array([[f_prime_x(x), f_prime_y(y)], [dx, dy]])
        matrix_inverse = np.linalg.inv(matrix)
        
        (dwx, dwy) = np.dot(matrix_inverse, np.array([- f(x, y), 0]))
        
        x += dwx
        y += dwy

        if np.abs(f(x, y)) < tol:
            break
    
    if k == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    

    return (x, y)   

#set up length of the bar
l = 4 

#set up objective function and constraints
f0 = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2
f1 = lambda x, y: x**2 + y**2 - 1
f2 = lambda x, y: (x - l)**2 + y**2 - 1

# calculate the gradient of the objective function and constraints
f0_prime = lambda x1, y1, x2, y2: np.array([2*(x1 - x2), 2*(y1 - y2), -2*(x1 - x2), -2*(y1 - y2)])
f1_x = lambda x: 2*x
f1_y = lambda y: 2*y
f2_x = lambda x: 2*(x - l)
f2_y = lambda y: 2*y

hessian = lambda x1, y1, x2, y2: np.array([[2, 0, -2, 0], [0, 2, 0, -2], [-2, 0, 2, 0], [0, -2, 0, 2]])
jacobian = lambda x1, y1, x2, y2: np.array([[2 * x1, 2 * y1, 0, 0], [0, 0, 2 * (x2 -l), 2 * y2]])
jacobian_transpose = lambda x1, y1, x2, y2: np.linalg.inv(jacobian(x1, y1, x2, y2))

# M = (H, J^T), (J, 0)
system_matrix = lambda x1, y1, x2, y2: np.array([[2, 0, -2, 0, 2 * x1, 0], [0, 2, 0, -2, 2 * y1, 0], [-2, 0, 2, 0, 0, 2 * (x2 - l)], 
                                          [0, -2, 0, 2, 0, 2 * y2], [2 * x1, 2 * y1, 0, 0, 0, 0], [0, 0, 2 * (x2 -l), 2 * y2, 0, 0]])

# v = (- grad f, constraints)
system_vector = lambda step, x1, y1, x2, y2: - np.array([step * 2*(x1 - x2), step * 2*(y1 - y2), step * (-2)*(x1 - x2), 
                                                         step * (-2)*(y1 - y2), x1**2 + y1**2 - 1, (x2 - l)**2 + y2**2 - 1]).reshape(6, 1)


def gradient_descent(obj, obj_prime, constraint1, constraint1_x, constraint1_y, constraint2, constraint2_x, constraint2_y, 
                     x0, step, tol, max_iterations):
    """ Uses Gradient descent to optimize the objective function combined with Newton's method to satisfy the two constraints
    Parameters:
    -----------
    - obj: the objective function to minimize
    - obj_prime: the gradient of the objective function
    - constraint1, constraint2: the constraints
    - constraint1_x, constraint2_x, constraint1_y, constraint2_y: the gradients of the constraints
    - x0: initial guess for the minimum
    - step: step size for updating x
    - tol: tolerance under which we stop the iteration
    - max_iterations: maximum number of iterations

    Returns:
    --------
    - x_min: the minimum value of x found by the algorithm
    - x_array: array containing these minimum values after each iteration
    """

    x = x0
    i = 0
    x_array = [x0]
    obj_val = [obj(x[0], x[1], x[2], x[3])] # array containing the objective fct value at after each step
    min_val = obj_val[0]

    while i < max_iterations:
        # Do Gradient descent step: x_new =(x1, y1, x2, y2)
        (dx1, dy1, dx2, dy2) = obj_prime(x[0], x[1], x[2], x[3])
        x_new = x - step * obj_prime(x[0], x[1], x[2], x[3])

        # Use Newton to get back to the constraints:
        x_new[0], x_new[1] = newton(constraint1, constraint1_x, constraint1_y, dx1, dy1, x_new[0], x_new[1])
        x_new[2], x_new[3] = newton(constraint2, constraint2_x, constraint2_y, dx2, dy2, x_new[2], x_new[3])
        
        # Append the new objective fct value and compute new minimum objective fct value
        """
        obj_val.append(obj(x_new[0], x_new[1], x_new[2], x_new[3]))
        min_val = np.min([min_val, obj_val[-1]]) # minimum value of the objective function over all iterations
        """
        delta = np.sum(np.abs(x_new - x))

        # This condition is crucial! Only take the G.D. and Newton step if the objective fct is actually minimized
        """
        Old condition
        if i >= 2000:
            if (obj_val[-1] <= 1.05 * min_val):
                x = x_new
            else:
                break
        else:
            x = x_new
        """
        x = x_new
        i += 1
        x_array.append(x)

        if delta < tol:
            break

    
    x_min = x_array[-1]
    x_array = np.array(x_array)
    return x_min, x_array


def gradient_descent2(matrix, vector, obj, constraint1, constraint2, x0, step, tol, max_iterations):
    """
    Parameters:
    -----------
    - matrix: the matrix with form [(H, J^T), (J, 0)], H - hessian of f, J - jacobian of the constraints
    - vector: the RHS of the linear system -(grad f, constraints)
    - obj: the objective function to minimize
    - constraint1, constraint2: the constraints
    - x0: initial guess for the minimum
    - step: step size for updating x
    - tol: tolerance under which we stop the iteration
    - max_iterations: maximum number of iterations
    Returns:
    --------
    - x_min: the minimum value of x found by the algorithm
    - x_array: array containing these minimum values after each iteration
    """
    x = x0
    i = 0
    x_array = [[x0[0], x0[1], x0[2], x0[3]]]

    while i < max_iterations:
        M = matrix(x[0], x[1], x[2], x[3])
        v = vector(step, x[0], x[1], x[2], x[3])

        result = np.dot(np.linalg.inv(M), v)
        for j in [0, 1, 2, 3]:
            x[j] += result[j][0]
        
        x_array.append(x[:])

        if (constraint1(x[0], x[1]) > tol):
            break
        if (constraint2(x[2], x[3]) > tol):
            break

    x_min = x_array[-1]
    x_array = np.array(x_array)
    return x_min, x_array


# Plots the data on xy plane along with two circles of radius 1 centered at (0, 0) and (l, 0)
# Not actually used in the main function, but useful as a plot
def plot_graph(data):
    fig = plt.figure(figsize=(12,6))
    axes = fig.add_subplot(1, 1, 1)

    for i in range(len(data)):
        x1 = data[i][0]
        y1 = data[i][1]
        x2 = data[i][2]
        y2 = data[i][3]
        if (i == 0):
            axes.plot(x1, y1, marker='*', color='orange', ls='none', ms=20, label='initial')
            axes.plot(x2, y2, marker='*', color='orange', ls='none', ms=20, label='initial')
        elif(i % 50 == 0):
            axes.plot(x1, y1, 'ro')
            axes.plot(x2, y2, 'ko')
        elif (i == len(data) - 2):
            axes.plot(x1, y1, marker='*', color='g', ls='none', ms=20, label='final')
            axes.plot(x2, y2, marker='*', color='g', ls='none', ms=20, label='final')

    theta = np.linspace( 0 , 2 * np.pi , 150 )
    
    radius = 1
    
    a1 = radius * np.cos( theta )
    b1 = radius * np.sin( theta )
    
    axes.plot(a1, b1, 'b', linestyle='dashed')
    axes.plot( l + a1, b1, 'b', linestyle='dashed')
    axes.plot(0, 0, 'kx')
    axes.plot(l, 0, 'kx')
    axes.plot(x1, y1, 'ro', label='($x_1, y_1$)')
    axes.plot(x2, y2, 'ko', label='($x_2, y_2$)')
    axes.set_title("xy plane")
    axes.set_xlabel('$x$',fontsize=16)
    axes.set_ylabel('$y$',fontsize=16)
    axes.legend()
    axes.grid()
    plt.axis("equal")
    plt.show()


def make_animation(x1, y1, x2, y2, initial_guess):
    """
    Makes an animation showing the points (x1, y1), (x2, y2) over time, the distance between these two points,
    as well as two circles of radius 1 centered at (0, 0) and (l, 0)
    """
    fig , axes = plt.subplots()

    # Function to update the plot for each frame of the animation
    def update(frame):
        axes.cla()  # Clear the current plot
        axes.plot(x1[:frame], y1[:frame], color='g')  # Plot data up to the current frame
        axes.scatter(x1[frame], y1[frame], color='r')  # Highlight the current point
        axes.plot(x2[:frame], y2[:frame], color='g')  
        axes.scatter(x2[frame], y2[frame], color='r')  
        axes.set_title("Distance animation for initial guess ({}, {}), ({}, {})".format(initial_guess[0], initial_guess[1], 
                                                                                        initial_guess[2], initial_guess[3]))
        axes.set_xlabel('$x$',fontsize=16)
        axes.set_ylabel('$y$',fontsize=16)
        
        # Add circles
        circle1 = Circle((0, 0), 1, fill=False, color='b', linestyle='--')  # Circle border at (0, 0) with radius 1
        circle2 = Circle((l, 0), 1, fill=False, color='b', linestyle='--')  # Circle border at (l, 0) with radius 1
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)

        # Line segment between (x1, y1) and (x2, y2)
        plt.plot([x1[frame], x2[frame]], [y1[frame], y2[frame]], color='black', linestyle='-')  

        # Add a grid and make axes equal
        plt.grid(True)
        plt.axis('equal')

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(x1), repeat=False)
    anim.save("myanimation3.mp4")
    plt.show()

if __name__ == "__main__":
    """ Makes a pretty animation using Gradient descent and Newton's method.
    Pick a value for the starting conditions a, b, c, d
    Parameters:
    -----------
    - a, b, c, d: initial starting conditions (must satisfy the constraints)
     Returns:
    --------
    Saves the animation of the movement of (x1, y1) and (x2, y2) as an mp4 file
    """
    #a, b, c, d = 0.8, 0.6, 3.2, 0.6
    #x_min, x_array = gradient_descent2(system_matrix, system_vector, f0, f1, f2, [a, b, c, d], 0.002, 2 * 1.e-3, 500)
    #x_min, x_array = gradient_descent(f0, f0_prime, f1, f1_x, f1_y, f2, f2_x, f2_y, [a, b, c, d], 0.002, 1e-5, 20000)

    #a, b, c, d = 0.1, 0.9949, 4.2, 0.97979
    #x_min, x_array = gradient_descent2(system_matrix, system_vector, f0, f1, f2, [a, b, c, d], 0.002, 100 * 1.e-3, 500)
    #x_min, x_array = gradient_descent(f0, f0_prime, f1, f1_x, f1_y, f2, f2_x, f2_y, [a, b, c, d], 0.002, 1e-5, 20000)

    a, b, c, d = -0.99, 0.1411, 4.99, 0.1411
    x_min, x_array = gradient_descent2(system_matrix, system_vector, f0, f1, f2, [a, b, c, d], 0.002, 2 * 1.e-3, 500)
    #x_min, x_array = gradient_descent(f0, f0_prime, f1, f1_x, f1_y, f2, f2_x, f2_y, [a, b, c, d], 0.002, 1e-5, 38000)

    #a, b, c, d = -0.6, 0.8, 4, -1
    #x_min, x_array = gradient_descent2(system_matrix, system_vector, f0, f1, f2, [a, b, c, d], 0.002, 5 * 1.e-3, 2000)
    #x_min, x_array = gradient_descent(f0, f0_prime, f1, f1_x, f1_y, f2, f2_x, f2_y, [a, b, c, d], 0.002, 1e-5, 20000)

    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(x_array)):
            if (i% 10 == 0 or i > 0.95 * len(x_array) ):
                x1.append(x_array[i][0])
                y1.append(x_array[i][1])
                x2.append(x_array[i][2])
                y2.append(x_array[i][3])
    
    make_animation(x1, y1, x2, y2, (a, b, c, d))

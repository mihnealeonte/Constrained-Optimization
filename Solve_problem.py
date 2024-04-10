import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.animation as animation
from matplotlib.patches import Circle

def newton(constraint1, grad1, constraint2, grad2, x, dx, tol = 1.e-6):
    """ Uses Newton's method to find a root x (in R^4) of two constraint functions f1, f2: R^4 -> R

    Parameters:
    -----------
    - constraint1, constraint2: the constraints
    - grad1, grad2: the gradients of the constraints
    - x: (numpy array) initial value in R^4
    - dx: (numpy array) gradient descent step taken in R^4
    - tol: Returns when |constraint(x + dx + dw)| < tol
        
    Returns:
    --------
    - alphas: array of coefficients that solve constraint(x + dx + alpha1 * grad1 + alpha2 * grad2) = 0
        
    Raises:
    -------
    Warning: 
        if number of iterations exceed MAX_STEPS
    """
    MAX_STEPS = 150
    
    # Initialize alpha1 = 0, alpha2 = 0
    alphas = np.array([0.0, 0.0])
    def F(alpha1, alpha2, constraint1, grad1, constraint2, grad2, x, dx):
        """Sets up a function F(alpha1, alpha2) : R^2 -> R^2"""

        evaluation_x = x + dx + alpha1 * grad1(x + dx) + alpha2 * grad2(x + dx)
        return np.array([constraint1(evaluation_x), constraint2(evaluation_x)])
    
    def gradF(alpha1, alpha2, grad1, grad2, x, dx):
            """Computes the gradient of F(alpha1, alpha2) by Chain rule and matrix multiplication"""

            evaluation_x = x + dx + alpha1 * grad1(x + dx) + alpha2 * grad2(x + dx)
            # First gradient in the chain rule (2 x 4 matrix)
            first_gradient = np.array([grad1(evaluation_x), grad2(evaluation_x)])
            # Second gradient in the chain rule (4 x 2 matrix)
            second_gradient = np.transpose(np.array([grad1(x + dx), grad2(x + dx)]))
            return np.dot(first_gradient, second_gradient)
        
    for k in range(1, MAX_STEPS + 1):
        # Evaluate the gradient of F at alpha1, alpha2
        gradF_evaluated = gradF(alpha1=alphas[0], alpha2=alphas[1], grad1= grad1, grad2=grad2, x=x, dx=dx)

        # Update the alphas
        delta_alphas = np.linalg.solve(gradF_evaluated, - F(alphas[0], alphas[1], constraint1, grad1, constraint2, grad2, x, dx))
        alphas = alphas + delta_alphas

        # Evaluate the constraints and check if they are satisfied
        evaluation_x = x + dx + alphas[0] * grad1(x + dx) + alphas[1] * grad2(x + dx)
        if np.abs(constraint1(evaluation_x)) < tol and np.abs(constraint2(evaluation_x)) < tol:
            break
        
    if k == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    
    return alphas 

#set up length of the bar
l = 4 

#set up objective function and constraints
# Improved f0 formula
f0 = lambda x: (x[0] - x[2])**2 + (x[1] - x[3])**2
# Improved f1 formula
f1 = lambda x: x[0]**2 + x[1]**2 - 1
# Improved f2 formula
f2 = lambda x: (x[2] - l)**2 + x[3]**2 - 1

# calculate the gradient of the objective function and constraints
#Improved gradients
f0_grad = lambda x: np.array([2*(x[0] - x[2]), 2*(x[1] - x[3]), -2*(x[0] - x[2]), -2*(x[1] - x[3])])
f1_grad = lambda x: np.array([2*x[0], 2*x[1], 0, 0])
f2_grad = lambda x: np.array([0, 0, 2*(x[2] - l), 2*x[3]])


def gradient_descent(obj, obj_grad, constraint1, grad1, constraint2, grad2, x0, step, tol, max_iterations):
    """ Uses Gradient descent to optimize the objective function combined with Newton's method 
        to satisfy the two constraints
    
    Parameters:
    -----------
    - obj: the objective function to minimize
    - obj_prime: the gradient of the objective function
    - constraint1, constraint2: the constraints
    - grad1, grad2: the gradients of the constraints
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

    while i < max_iterations:
        # Do  Gradient Descent step
        # dx = [x1, y1, x2, y2]
        dx = - step * obj_grad(x)

        # Use Newton to get back to the constraints:
        alphas = newton(constraint1=constraint1, grad1=grad1, constraint2=constraint2, grad2=grad2, x=x, dx=dx)

        # Update x by adding both the GD step and the correction
        alpha1, alpha2 = alphas[0], alphas[1]
        x_new = x + dx + alpha1 * grad1(x + dx) + alpha2 * grad2(x + dx)
        
        delta = np.sum(np.abs(x_new - x))

        x = x_new
        i += 1

        x_array.append(x)

        if delta < tol:
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
    anim.save("myanimation1.mp4")
    plt.show()

if __name__ == "__main__":
    """ Makes a pretty animation using Gradient descent and Newton's method.
    Pick a value for the starting conditions a, b, c, d
    Parameters:
    -----------
    - a, b, c, d: initial starting conditions (must satisfy the constraints)
     Returns:
    --------
    saves a the animation of the movement of (x1, y1) and (x2, y2) as a mp4 file
    """
    a, b, c, d = 0.8, 0.6, 3.2, 0.6
    x_min, x_array = gradient_descent(f0, f0_grad, f1, f1_grad, f2, f2_grad, [a, b, c, d], 0.01, 1e-5, 200)

    #a, b, c, d = 0.1, 0.99, 4.2, 0.98
    #x_min, x_array = gradient_descent(f0, f0_grad, f1, f1_grad, f2, f2_grad, [a, b, c, d], 0.01, 1e-5, 200)

    #a, b, c, d = -0.99, 0.1411, 4.99, 0.1411
    #x_min, x_array = gradient_descent(f0, f0_grad, f1, f1_grad, f2, f2_grad, [a, b, c, d], 0.01, 1e-5, 200)

    #a, b, c, d = -0.6, 0.8, 4, -1
    #x_min, x_array = gradient_descent(f0, f0_grad, f1, f1_grad, f2, f2_grad, [a, b, c, d], 0.01, 1e-5, 200)

    x1, y1, x2, y2 = [], [], [], []
    for i in range(len(x_array)):
            if (i% 5 == 0 or i > 0.95 * len(x_array) ):
                x1.append(x_array[i][0])
                y1.append(x_array[i][1])
                x2.append(x_array[i][2])
                y2.append(x_array[i][3])
    
    make_animation(x1, y1, x2, y2, (a, b, c, d))
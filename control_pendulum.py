import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

def control_pendulum(theta0, phi0, alpha):

    #theta0 = 0
    #phi0 = 0
    cost = 0

    J = np.matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
    #del_H = np.matrix([[4*theta0 + 1*costate2],[costate1 - costate2/10],[phi0],[(-1/10)*phi0] + theta0 - costate2])
    del_H = np.matrix([[1,0,0,1], [0,0,1,-alpha], [0,1,0,0], [0,-alpha,0,-1]])
    #del_H = np.matrix([[0, 1, 0, 0], [1, -alpha, 0, -1], [-4, 0, 0, -1], [0, 0, -1, alpha]])
    A = np.matmul(J,del_H)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    D = np.matmul(np.matmul(np.linalg.inv(eigenvectors),A),eigenvectors)
    savei = np.zeros([2,1])
    count = 0
    for i in range(0,len(D)-2):
        if D[i, i] < 0:
            savei[count,0] = i + 1
            count = count + 1

    #for i in range(0,len(D)):
    #    for j in range(0,len(D)):
            #if(abs(D[i,j]) < .00000000001):
            #    D[i,j] = 0

    Vs = np.zeros([4,2],dtype = complex)
    Vs2 = np.zeros([2,2],dtype = complex)
    Vs1 = np.zeros([2,2], dtype=complex)
    count = 0
    for i in [2,3]:
        for j in range(0,len(D)):
            Vs[j,count] = eigenvectors[j,i]
        count = count + 1
    Vs2 = np.array([[Vs[0,0], Vs[0,1]],[Vs[1,0],Vs[1,1]]],dtype = complex)
    Vs1 = np.array([[Vs[2,0], Vs[2,1]],[Vs[3,0],Vs[3,1]]],dtype = complex)
    P = np.matmul(Vs2,np.linalg.inv(Vs1))

    #P[0,0] = 5.793037833617297
    #P[0,1] = 3.236067977499793
    #P[1,0] = 3.236067977499793
    #P[1,1] = 2.446003918889283
    r = .001
    N = 360

    time_values_p = np.zeros([1,1])
    state_values_p = np.zeros([1,9])

    costs = 0

    for i in range(1,N):

        theta1 = r*math.cos(2*math.pi*i/N)
        phi1 =  r*math.sin(2*math.pi*i/N)

        costate1 = P[0,0]*theta1 + P[0,1]*phi1
        costate2 = P[1,0]*theta1 + P[1,1]*phi1

        cs = np.zeros([2,1])
        S = np.matrix([[theta1],[phi1]])

        t0 = 0
        cs = P*S
        t_bound = .2
        #y0 = np.array([theta0,phi0,P[0,0],P[0,1],P[1,0],P[1,1],alpha,cost])
        y0 = np.array([theta1,phi1,costate1,costate2,P[1,0],P[1,1],alpha,cost,i])
        sol = sp.integrate.RK45(Dynamics, t0, y0, t_bound, max_step=1, rtol=0.001, atol=1e-06)

        time_values = [sol.t]
        state_values = [sol.y.copy()]

        while sol.status == 'running':
            sol.step()
            time_values.append(sol.t)
            state_values.append(sol.y.copy())

        time_values = np.array(time_values)
        state_values = np.array(state_values)

        time_values_n = np.zeros([len(time_values[:])+len(time_values_p[:])])
        state_values_n = np.zeros([len(state_values[:, 0]) + len(state_values_p[:, 0]), 9])

        for j in range(0,len(time_values_n[:])):
            if j < len(time_values_p[:]):
                time_values_n[j] = time_values_p[j]
            else:
                time_values_n[j] = time_values[j-len(time_values_p[:])]

        for j in range(0,len(time_values_n[:])):
            if j < len(time_values_p[:]):
                state_values_n[j,:] = state_values_p[j,:]
            else:
                state_values_n[j,:] = state_values[j-len(time_values_p[:]),:]

        state_values_p = state_values_n
        time_values_p = time_values_n

        #lambda1 = np.zeros([len(state_values[:,0]),1])
        #lambda2 = np.zeros([len(state_values[:,0]),1])

        #for i in range(0,len(time_values[:])):
            #lambda1[i,0] = state_values[i,0]*P[0,0] + state_values[i,1]*P[0,1]
            #lambda2[i,0] = state_values[i,0]*P[1,0] + state_values[i,1]*P[1,1]
        #    if(i <= len(time_values[:])-2):
                #cost = cost + (time_values[i+1]-time_values[i])*(lambda1*state_values[i,1] + lambda2*(-alpha*state_values[i,1] + math.sin(state_values[i,0]) - lambda2*math.cos(state_values[i,0]))) + (math.sin(2*state_values[i,0])*math.sin(2*state_values[i,0])/2) + lambda2*lambda2/2
        #       cost = cost + (time_values[i + 1] - time_values[i]) * (lambda1 * state_values[i, 1] + lambda2 * (-alpha * state_values[i, 1] + math.sin(state_values[i, 0]) - lambda2 * math.cos(state_values[i, 0]))) + (1-math.cos(state_values[i,0])) + lambda2 * lambda2 / 2

        # Plotting
    '''
    plt.plot(state_values_n[:, len(state_values[0, :]) - 8], state_values_n[:, len(state_values[0, :]) - 7])

        # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simple Line Plot')
    plt.show()
    '''

    lambda1 = 1
    lambda2 = 2

    lambda1 = lambda1.real
    lambda2 = lambda2.real

    cstate1 = float(cs[0,0])
    cstate2 = float(cs[1,0])

    # Display the plot

    error_m = 1
    k_m = 1000000
    for k in range(0, len(state_values_n)):
        error = abs(state_values_n[k, 0].real - theta0) + abs(state_values_n[k, 1].real - phi0)
        if (error < error_m):
            error_m = error
            k_m = k
    if (k_m != 1000000):
        #lambda1 = state_values_n[k_m, 2]
        #lambda2 = state_values_n[k_m, 3]
        cost = time_values_n[k_m]
        cost1 = time_values_n[k_m]+.000001
        J1 = .0001*np.identity(2)
        k = int(state_values_n[k_m, 8]) + 1
        psi_n1 = 2 * math.pi * k / N
        r = r
        psi_n = 2 * math.pi * (k-1)/ N
        if(psi_n1 > 2*math.pi):
            psi_n1 = psi_n1 - 2*math.pi
        if(psi_n > 2*math.pi):
            psi_n = psi_n - 2*math.pi
        if(psi_n1 < 0):
            psi_n1 = psi_n1 + 2*math.pi
        if(psi_n < 0):
            psi_n = psi_n + 2*math.pi
        theta1 = r * math.cos(psi_n)
        phi1 = r * math.sin(psi_n)
        costate1_0 = P[0, 0] * theta1 + P[0, 1] * phi1
        costate2_0 = P[1, 0] * theta1 + P[1, 1] * phi1
        error_theta, error_phi, lambda1, lambda2, cost = Integrate(theta1, phi1, costate1_0.real, costate2_0.real, alpha, theta0, phi0,error.real, lambda1.real, lambda2.real, cost.real)
        #theta1 = r * math.cos(psi_n1)
        #phi1 = r * math.sin(psi_n1)
        #costate1_0 = P[0, 0] * theta1 + P[0, 1] * phi1
        #costate2_0 = P[1, 0] * theta1 + P[1, 1] * phi1
        #error1_theta, error1_phi, lambda1, lambda2, cost1 = Integrate(theta1, phi1, costate1_0.real, costate2_0.real, alpha, theta0, phi0, error.real, lambda1.real, lambda2.real, cost1.real)
        on = 1
        count = 0

        error1_theta = 1
        error1_phi = 1
        while ((abs(error1_theta) > .000001 or abs(error1_phi) > .000001)):
            #if(count != 0):
            #    error1_theta, error1_phi,lambda1,lambda2,cost1 = Integrate(theta1, phi1, costate1_0.real, costate2_0.real, alpha, theta0, phi0, error.real,lambda1.real,lambda2.real,cost1.real)

            #count = count + 1
            #if (error1_theta != error_theta and error1_phi != error_phi):
                #del_xJ = np.matmul([psi_n,cost],J1)

            xp1 = np.array([[psi_n], [cost]] - np.matmul(J1, [[error_theta], [error_phi]]))
            if(xp1[1,0] < 0):
                xp1[1,0] = .05
            if (xp1[0,0] > 2 * math.pi):
                xp1[0,0] = xp1[0,0] - 2 * math.pi
            if (psi_n > 2 * math.pi):
                xp1[0,0] = xp1[0,0] - 2 * math.pi
            if (psi_n1 < 0):
                xp1[0,0] = xp1[0,0] + 2 * math.pi
            if (psi_n < 0):
                xp1[0,0] = xp1[0,0] + 2 * math.pi
            psi_n1 = xp1[0,0]
            cost1 = xp1[1,0]
            theta1 = r * math.cos(xp1[0,0])
            phi1 = r * math.sin(xp1[0,0])
            costate1_0 = P[0, 0] * theta1 + P[0, 1] * phi1
            costate2_0 = P[1, 0] * theta1 + P[1, 1] * phi1
            error1_theta, error1_phi, lambda1, lambda2, cost1 = Integrate(theta1, phi1, costate1_0.real,costate2_0.real, alpha, theta0, phi0,error.real, lambda1.real, lambda2.real,xp1[1,0])
            Jdelx = np.matmul(J1,[error1_theta-error_theta,error1_phi-error_phi])
            delx = np.array([psi_n1, cost1])-np.array([psi_n, cost])
            xJ = np.matmul(delx, J1)
            xJF = np.matmul(xJ,[[error1_theta-error_theta],[error1_phi-error_phi]])
            xp = np.array([[psi_n1],[cost1]])
            J1 = J1 + np.matmul((delx - Jdelx) / xJF, xJ)
            if(cost1 < 0):
                cost1 = .05
            if (psi_n1 > 2 * math.pi):
                psi_n1 = psi_n1 - 2 * math.pi
            if (psi_n > 2 * math.pi):
                psi_n = psi_n - 2 * math.pi
            if (psi_n1 < 0):
                psi_n1 = psi_n1 + 2 * math.pi
            if (psi_n < 0):
                psi_n = psi_n + 2 * math.pi
            psi_n = psi_n1
            cost = cost1
            error_theta = error1_theta
            error_phi = error1_phi
            #else:
            #    on = 2

        costs = cost

    return(lambda1,lambda2),costs

def Integrate(theta1,phi1,costate1_0,costate2_0,alpha,theta0,phi0,error,lambda1,lambda2,cost):

    t_bound = cost
    t0 = 0
    # y0 = np.array([theta0,phi0,P[0,0],P[0,1],P[1,0],P[1,1],alpha,cost])
    y0 = np.array([theta1, phi1, costate1_0, costate2_0, 0, 0, alpha, 0,0])
    if(np.isnan(y0[0])):
        return error,lambda1,lambda2,cost
    sol = sp.integrate.RK45(Dynamics, t0, y0, t_bound, max_step=1, rtol=0.001, atol=1e-06)

    time_values = [sol.t]
    state_values = [sol.y.copy()]

    while sol.status == 'running':
        sol.step()
        time_values.append(sol.t)
        state_values.append(sol.y.copy())

    time_values = np.array(time_values)
    state_values = np.array(state_values)
    '''
    error_m = 1
    error_theta_m = 1
    error_phi_m = 1
    k_m = 1000000
    for k in range(0, len(state_values)):
        error_theta = (state_values[k, 0] - theta0) * (state_values[k, 0] - theta0)
        error_phi = (state_values[k, 1] - phi0) * (state_values[k, 1] - phi0)
        if(error_theta + error_phi < error_m):
            error_m = error_theta_m + error_phi_m
            k_m = k
            error_theta_m = error_theta
            error_phi_m = error_phi
    '''
    error_theta_m = (state_values[len(state_values[:,0])-1,0]-theta0)
    error_phi_m = (state_values[len(state_values[:,0])-1,1]-phi0)

    return error_theta_m, error_phi_m,state_values[len(state_values)-1,2],state_values[len(state_values)-1,3],time_values[len(state_values)-1]

def Dynamics(t,y):
    theta = y[0]
    phi = y[1]
    #costate1 = y[2]*theta + y[3]*phi
    #costate2 = y[4]*theta + y[5]*phi
    costate1 = y[2]
    costate2 = y[3]
    alpha = y[6]

    theta_dot = -phi
    phi_dot = -(-phi*alpha + math.sin(theta) - costate2*math.cos(theta))
    costate1_dot = -(costate2 * (math.cos(theta) + costate2 * math.sin(theta)) + math.sin(theta))
    costate2_dot = -(costate1 + costate2 * (-alpha ))
    J_dot =  (costate1 * phi + costate2 * (-alpha * phi + math.sin(theta) - costate2 * math.cos(theta))) + (1-math.cos(theta)) + costate2 * costate2 / 2

    dtheta_J_dot = theta_dot/J_dot
    dphi_J_dot = phi_dot/J_dot
    costate1_dot_J_dot = costate1_dot/J_dot
    costate2_dot_J_dot = costate2_dot/J_dot

    yn = np.array([dtheta_J_dot,dphi_J_dot,costate1_dot_J_dot,costate2_dot_J_dot,0,0,0,0,0])

    return yn
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#from kneed import KneeLocator


def build_cov_giuseppe(geometry, TS, sigma_noise, sigma_model, L, L0):
    ### BUILD THE VARIANCE MATRICES
    Cd = sigma_noise * np.eye(len(TS[0, :, 0]), len(TS[0, :, 0]))

    # COMPUTE INTER-PATCH DISTANCE
    s = np.zeros((geometry.shape[0], geometry.shape[0]))
    for i in range(geometry.shape[0]):
        s[i, :] = np.sqrt((geometry[:, 9] - geometry[i, 9]) ** 2 + (geometry[:, 10] - geometry[i, 10]) ** 2)

    # COMPUTE THE MODEL COVARIANCE MATRIX
    Cm = (sigma_model * L0 / L) ** 2 * np.exp(-s / L)
    Cm = np.concatenate((np.concatenate((Cm, np.zeros(np.shape(Cm)))), np.concatenate((np.zeros(np.shape(Cm)), Cm))),
                        axis=1)

    return Cd, Cm



def build_cov_rake(geometry, model, TS, sigma_noise, sigma_model, L, L0):

    ### BUILD THE VARIANCE MATRICES
    Cd = sigma_noise * np.eye(len(TS[0,:,0]),len(TS[0,:,0]))

    # COMPUTE INTER-PATCH DISTANCE
    s = np.zeros((len(model[0,:]),len(model[0,:])))
    for i in range(len(model[0,:])):
        s[i,:] = np.sqrt( (geometry[:,9]-geometry[i,9])**2 + (geometry[:,10]-geometry[i,10])**2 )

    # COMPUTE THE MODEL COVARIANCE MATRIX
    Cm = (sigma_model*L0/L)**2 * np.exp(-s/L)

    return Cd, Cm

def build_cov(geometry, model, TS, sigma_noise, sigma_model, L, L0):

    ### BUILD THE VARIANCE MATRICES
    Cd = sigma_noise * np.eye(len(TS[0,:,0]),len(TS[0,:,0]))

    # COMPUTE INTER-PATCH DISTANCE
    s = np.zeros((len(model[0,:]),len(model[0,:])))
    for i in range(len(model[0,:])):
        s[i,:] = np.sqrt( (geometry[:,9]-geometry[i,9])**2 + (geometry[:,10]-geometry[i,10])**2 )

    # COMPUTE THE MODEL COVARIANCE MATRIX
    Cm = (sigma_model*L0/L)**2 * np.exp(-s/L)
    Cm = np.concatenate(( np.concatenate((Cm, np.zeros(np.shape(Cm)))), np.concatenate((np.zeros(np.shape(Cm)), Cm)) ), axis = 1)

    return Cd, Cm

def inverse_LS_fixed_rake(n_points, n_time_steps, geometry, GREEN, TS, Cd, Cm):

    ### INITIATE THE MODEL
    m0 = np.zeros(2*len(geometry[:,9]))

    models = np.zeros((n_time_steps, len(geometry[:,9])))

    for i in range(n_points,n_time_steps-n_points,1): # Loop over the time steps

        ## Initialize the model
        m = m0

        ##### Loop over the components E,N,U
        for comp in range(3):

            ##### MANAGE THE TIME SERIES AVERAGING ######
            s, e = i-n_points, i+1+n_points
            d = np.sum(TS[s:e, :, comp], axis = 0)/(1+2*n_points) # Suppose complete time series, compute the offsets

            ##### MANAGE THE GREEN FUNCTIONS FOR DIP DIRECTION
            G = GREEN[:, :, comp, 0].T

            ##### LEAST-SQUARES INVERSION (DIP-SLIP COMPONENT) #####
            X = np.linalg.inv(np.linalg.multi_dot([G, Cm, G.T]) + Cd) ## STDs / G
            XX = (d - np.dot(G, m)) ## Data / model
            m += np.linalg.multi_dot([Cm, G.T, X, XX]) ## Dot product all

        models[:,i] = m

    return models

def inverse_LS(n_points, n_time_steps, geometry, GREEN, TS, Cd, Cm):

    ### INITIATE THE MODEL
    m0 = np.zeros(2*len(geometry[:,9]))

    models = np.zeros((n_time_steps, 2*len(geometry[:,9])))

    for i in range(n_points,n_time_steps-n_points,1): # Loop over the time steps
        if i % 100 == 0:
            print(i)
        ## Initialize the model
        m = m0

        ##### Loop over the components E,N,U
        for comp in range(3):

            ##### MANAGE THE TIME SERIES AVERAGING ######
            s, e = i-n_points, i+1+n_points
            d = np.sum(TS[s:e, :, comp], axis = 0)/(1+2*n_points) # Suppose complete time series, compute the offsets

            ##### MANAGE THE GREEN FUNCTIONS FOR DIP AND STRIKE SLIP #####
            G = np.concatenate((GREEN[:, :, comp, 0].T, GREEN[:, :, comp, 1].T), axis = 1)

            ##### LEAST-SQUARES INVERSION (DIP-SLIP COMPONENT) #####
            X = np.linalg.inv(np.linalg.multi_dot([G, Cm, G.T]) + Cd) ## STDs / G
            XX = (d - np.dot(G, m)) ## Data / model
            m += np.linalg.multi_dot([Cm, G.T, X, XX]) ## Dot product all

        models[i,:] = m

    return models

def _inverse_LS_parallel_step(i, n_points, geometry, G_matrices, TS, Cd_3D, Cm, fixed_rake=False):
    if fixed_rake:
        m0 = np.zeros(len(geometry[:, 9]))
    else:
        m0 = np.zeros(2 * len(geometry[:, 9]))
    m = m0
    for comp in range(3):
        ##### MANAGE THE TIME SERIES AVERAGING ######
        if n_points > 0:
            s, e = i - n_points, i + 1 + n_points
            d = np.sum(TS[s:e, :, comp], axis=0) / (
                    1 + 2 * n_points)  # Suppose complete time series, compute the offsets
        else:
            d = TS[i, :, comp]

        ##### MANAGE THE GREEN FUNCTIONS FOR DIP AND STRIKE SLIP #####
        # G = np.concatenate((GREEN[:, :, comp, 0].T, GREEN[:, :, comp, 1].T), axis=1)
        G = G_matrices[comp]

        Cd = Cd_3D[..., comp]

        ##### LEAST-SQUARES INVERSION (DIP-SLIP COMPONENT) #####
        X = np.linalg.inv(np.linalg.multi_dot([G, Cm, G.T]) + Cd)  ## STDs / G
        XX = (d - np.dot(G, m))  ## Data / model
        m += np.linalg.multi_dot([Cm, G.T, X, XX])  ## Dot product all
    return m


def inverse_LS_parallel(n_points, n_time_steps, geometry, GREEN, TS, Cd, Cm, fixed_rake=False):
    from joblib import Parallel, delayed
    if fixed_rake:
        G_matrices = [GREEN[:, :, comp, 0].T for comp in range(3)]
    else:
        G_matrices = [np.concatenate((GREEN[:, :, comp, 0].T, GREEN[:, :, comp, 1].T), axis=1) for comp in range(3)]
    models = Parallel(n_jobs=-1, verbose=True)(
        # delayed(_inverse_LS_parallel_step)(i, n_points, geometry, GREEN, TS, Cd, Cm) for i in
        delayed(_inverse_LS_parallel_step)(i, n_points, geometry, G_matrices, TS, Cd, Cm, fixed_rake=fixed_rake) for i
        in range(n_points, n_time_steps - n_points, 1))
    return np.array(models)

def _inverse_LS_parallel_step_optimized(i, n_points, geometry, G_matrices, TS, sigma_noise, sigma_model, L0):
    misfit_array = []
    max_slip_array = []
    models_array = []
    L_array = [5, 7, 10]
    for L in L_array:
        m0 = np.zeros(2 * len(geometry[:, 9]))
        m = m0
        Cd, Cm = build_cov_giuseppe(geometry, TS, sigma_noise, sigma_model, L, L0)
        misfit = 0
        for comp in range(3):
            ##### MANAGE THE TIME SERIES AVERAGING ######
            if n_points > 0:
                s, e = i - n_points, i + 1 + n_points
                d = np.sum(TS[s:e, :, comp], axis=0) / (1 + 2 * n_points)  # Suppose complete time series, compute the offsets
            else:
                d = TS[i, :, comp]

            ##### MANAGE THE GREEN FUNCTIONS FOR DIP AND STRIKE SLIP #####
            # G = np.concatenate((GREEN[:, :, comp, 0].T, GREEN[:, :, comp, 1].T), axis=1)
            G = G_matrices[comp]

            ##### LEAST-SQUARES INVERSION (DIP-SLIP COMPONENT) #####
            X = np.linalg.inv(np.linalg.multi_dot([G, Cm, G.T]) + Cd)  ## STDs / G
            XX = (d - np.dot(G, m))  ## Data / model
            m += np.linalg.multi_dot([Cm, G.T, X, XX])  ## Dot product all
            misfit += 0.5 * (np.linalg.multi_dot(
                [(np.dot(G, m) - d).T, np.linalg.inv(Cd), (np.dot(G, m) - d)]) + np.linalg.multi_dot(
                [(m - m0).T, np.linalg.inv(Cm), (m - m0)]))
        max_slip_array.append(np.max(m))
        misfit_array.append(misfit)
        models_array.append(m)
    optimal_L = KneeLocator(L_array, misfit_array, curve='convex', direction='decreasing').knee
    idx_optimal_L = L_array.index(optimal_L)
    optimal_model = models_array[idx_optimal_L]
    return optimal_model


def inverse_LS_parallel_optimized(n_points, n_time_steps, geometry, GREEN, TS, sigma_noise, sigma_model, L0):
    from joblib import Parallel, delayed
    G_matrices = [np.concatenate((GREEN[:, :, comp, 0].T, GREEN[:, :, comp, 1].T), axis=1) for comp in range(3)]
    models = Parallel(n_jobs=-2, verbose=True)(
        # delayed(_inverse_LS_parallel_step)(i, n_points, geometry, GREEN, TS, Cd, Cm) for i in
        delayed(_inverse_LS_parallel_step_optimized)(i, n_points, geometry, G_matrices, TS, sigma_noise, sigma_model, L0) for i in
        range(n_points, n_time_steps - n_points, 1))
    return np.array(models)


def compute_forward_model_rake():

    TS = np.zeros((n_time_steps,len(GREEN[0,:,0,0]),3))

    ### MAKE TIME SERIES
    TS_N = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))
    TS_E = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))
    TS_U = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))

    for i in range(n_points,n_time_steps-n_points,1): # Loop over the time steps

        m = models[:,i]

        #### COMPUTE THE NEW DIRECT MODEL
        dd_x = np.dot( GREEN[:, :, 0, 0].T, m)
        dd_y = np.dot( GREEN[:, :, 1, 0].T, m)
        dd_z = np.dot( GREEN[:, :, 2, 0].T, m)

        TS_N[i,:] += d_y_0
        TS_E[i,:] += d_x_0
        TS_U[i,:] += d_z_0

    TS[:,:,0], TS[:,:,1], TS[:,:,2] = TS_E, TS_N, TS_U

    return TS

def compute_forward_model(slip_model, GREEN, n_points, n_time_steps):

    TS = np.zeros((n_time_steps,len(GREEN[0,:,0,0]),3))

    ### MAKE TIME SERIES
    TS_N = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))
    TS_E = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))
    TS_U = np.zeros((len(slip_model[:,0]),len(GREEN[0,:,0,0])))

    for i in range(n_points,n_time_steps-n_points,1): # Loop over the time steps

        m = slip_model[i,:]

        #### COMPUTE THE NEW DIRECT MODEL
        dd_x = np.dot( np.concatenate((GREEN[:, :, 0, 0].T, GREEN[:, :, 0, 1].T), axis = 1), m)
        dd_y = np.dot( np.concatenate((GREEN[:, :, 1, 0].T, GREEN[:, :, 1, 1].T), axis = 1), m)
        dd_z = np.dot( np.concatenate((GREEN[:, :, 2, 0].T, GREEN[:, :, 2, 1].T), axis = 1), m)

        TS_N[i,:] += dd_y
        TS_E[i,:] += dd_x
        TS_U[i,:] += dd_z

    TS[:,:,0], TS[:,:,1], TS[:,:,2] = TS_E, TS_N, TS_U

    return TS

def compute_forward_model_vectorized(slip_model, GREEN, n_points, n_time_steps, fixed_rake=False, positivity=False):
    # Make time series
    TS_N = np.zeros((n_time_steps, len(GREEN[0, :, 0, 0])))
    TS_E = np.zeros((n_time_steps, len(GREEN[0, :, 0, 0])))
    TS_U = np.zeros((n_time_steps, len(GREEN[0, :, 0, 0])))

    # Compute the new direct model for all time steps at once
    if fixed_rake or positivity:
        dd_x = np.dot(GREEN[:, :, 0, 0].T, slip_model.T)
        dd_y = np.dot(GREEN[:, :, 1, 0].T, slip_model.T)
        dd_z = np.dot(GREEN[:, :, 2, 0].T, slip_model.T)
    else:
        dd_x = np.dot(np.concatenate((GREEN[:, :, 0, 0].T, GREEN[:, :, 0, 1].T), axis=1), slip_model.T)
        dd_y = np.dot(np.concatenate((GREEN[:, :, 1, 0].T, GREEN[:, :, 1, 1].T), axis=1), slip_model.T)
        dd_z = np.dot(np.concatenate((GREEN[:, :, 2, 0].T, GREEN[:, :, 2, 1].T), axis=1), slip_model.T)

    # Populate the time series arrays
    TS_E[n_points:n_time_steps-n_points, :] = dd_x.T
    TS_N[n_points:n_time_steps-n_points, :] = dd_y.T
    TS_U[n_points:n_time_steps-n_points, :] = dd_z.T

    TS = np.stack((TS_E, TS_N, TS_U), axis=-1)

    return TS
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pyproj import Transformer
#import jigsawpy
import scipy
import matplotlib.pyplot as plt


""""""""" PROJECTIONS """""""""
def GEO_UTM(longitude,latitude):
    
    # CONVERSION THE GEOGRAFICAL COORDINATES EPSG:4326 TO MERCANTOUR COORDINATES EPSG:3857
    longitude, latitude = np.array(longitude), np.array(latitude)
    TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    (x,y) = TRAN_4326_TO_3857.transform(longitude, latitude)
    
    return x/1.E3, y/1.E3


def UTM_GEO(x, y):
    
    # CONVERSION THE MERCANTOUR COORDINATES EPSG:3857 TO GEOGRAFICAL COORDINATES EPSG:4326
    x, y = np.array(x), np.array(y) # CONVERT BACK TO METERS
    TRAN_3857_TO_4326 = Transformer.from_crs("EPSG:3857","EPSG:4326", always_xy=True) # DEFINE THE TRANSFORMATION
    
    return TRAN_3857_TO_4326.transform(x*1.E3, y*1.E3) # RETURN THE COORDIANTES IN DEGREES


def mat_rot_local_to_general(lam,phi,unit='radians'):
    
    # R IS AN ORTHOGONAL CONVERSION MATRIX, FROM THE GEOCENTRIC REFERENCE (X,Y,Z) FRAME TO THE LOCAL FRAME (N,E,U)
    
    if unit not in ['radians','dec_deg']:
        raise OptionError("unit option must be in [radians,dec_deg];unit=",unit)
    
    return(mat_rot_general_to_local(lam,phi,unit=unit).T)


def mat_rot_general_to_local(lam,phi,unit='radians'):
    
    # R IS AN ORTHOGONAL CONVERSION MATRIX, FROM THE LOCAL FRAME (N,E,U) TO THE GEOCENTRIC REFERENCE (X,Y,Z)
    """ MATRIX
    R = [[-sin(lambda),      cos(lambda),     0      ],
    
    [-sin(phi)*cos(lambda) , -sin(phi)*sin(lamda) , cos(phi)],
    
    [ cos(phi)*cos(lambda) , cos(phi)*sin(lamda) , sin(phi)]]
    """

    if unit not in ['radians','dec_deg']:
        raise OptionError("unit option must be in [radians,dec_deg];unit=",unit)

    if unit == 'dec_deg':
        lam = np.radians(lam)
        phi = np.radians(phi)
    
    R = np.zeros([3,3], float)
    R[0,0] = -np.sin(lam)
    R[0,1] = np.cos(lam)

    R[1,0] = -np.sin(phi)*np.cos(lam)
    R[1,1] = -np.sin(phi)*np.sin(lam)
    R[1,2] = np.cos(phi)
    
    R[2,0] = np.cos(phi)*np.cos(lam)
    R[2,1] = np.cos(phi)*np.sin(lam)
    R[2,2] = np.sin(phi)

    return R

""""""""" """"""""" """"""""" """""""""


""""""""" SLAB RELATED FUNCTIONS """""""""
def cut_slab_rectangle(model_slab,x1,x2,y1,y2):
    
    # THIS FUNCTION SIMPLY CUTS THE SLAB MODEL TO THE FOCUS AREA
    # ACCORDING TO THE DEFINED x1x2,y1,y2
    idxs = np.where((model_slab[:,0] >= x1) & (model_slab[:,0] <= x2) & (model_slab[:,1] >= y1) & (model_slab[:,1] <= y2))[0]
    model_cut = model_slab[idxs]

    return model_cut

def sort_contours(model_slab_contours):

    contours_x, contours_y, contours_z = [], [], []

    for i in range(len(model_slab_contours[:,0])):

        if model_slab_contours[i,0] == '>':
            pass

        else:
            contours_x.append(float(model_slab_contours[i,0]))
            contours_y.append(float(model_slab_contours[i,1]))
            contours_z.append(float(model_slab_contours[i,2]))

    model = np.c_[contours_x,contours_y,contours_z]
    return model


def cut_slab_convergence(slab_contours,convergence_direction):

    # THIS FUNCTION ALLOWS TO CUT THE SLAB IN A DIRECTION _| TO THE CONVERGENCE DIRECTION

    # COMPUTE THE TANGENTS FOR THE DIRECTION OF THE CONVERGENCE
    tan_calc_max = (np.max(slab_contours[:,1]) - np.min(slab_contours[:,1])) * np.tan(convergence_direction*np.pi/180)
    
    # DEFINE THE LIMITS OVER LAT AND LON 
    lim_lon = [np.min(slab_contours[:,0])-3, np.max(slab_contours[:,0])+2]
    lim_lat = [np.min(slab_contours[:,1]), np.max(slab_contours[:,1])]

    # DEFINE THE EDGES OF THE PARALLELOGRAM THAT FOLOOWS THE CONVERGENCE
    threshold_min_L = lim_lon[0]
    threshold_max_L = lim_lon[0] + tan_calc_max
    threshold_min_R = lim_lon[1] - tan_calc_max
    threshold_max_R = lim_lon[1]

    # DEFINE A BOUNDARY
    a1 = np.polyfit([threshold_min_R ,threshold_max_R], lim_lat, 1) # East
    b1 = np.polyfit([threshold_min_L ,threshold_max_L], lim_lat, 1) # West
    
    x = np.arange(threshold_min_L, threshold_max_L, 0.1)
    y_lim_right = a1[0]*x+a1[1]
    y_lim_left = b1[0]*x+b1[1]

    # GET THE UNIQUE CONTOURS
    u, indices = np.unique(slab_contours[:,2], return_index=True)

    # GET THE BOUNDED CONTOURS FOR EACH DEPTH LIMIT
    xc_lim, yc_lim = [], []

    # LOOP OVER THE UNIQUE CONTOURS
    for i in range(len(u)):

        idxs_depth = np.where( slab_contours[:,2] == u[i])[0]

        lons = slab_contours[idxs_depth,0]
        lats = slab_contours[idxs_depth,1]

        # COMPUTE THE INTERSECTION OF THE CONTOUR AND THE LEFT BOUNDARY LIMIT

        b = np.polyfit(lons[-2:],lats[-2:],1) # West part

        x = np.arange(threshold_min_L,threshold_max_L,0.1)
        y_slab = b[0]*x+b[1]

        intersection_left = line_intersection([x[0],y_slab[0]],[x[-1],y_slab[-1]],[x[0],y_lim_right[0]],[x[-1],y_lim_right[-1]])

        # COMPUTE THE INTERSECTION OF THE CONTOUR AND THE RIGHT BOUNDARY LIMIT

        a = np.polyfit(lons[:2],lats[:2],1) # East part
        y_slab = a[0]*x+a[1]

        print(y_lim_left, y_lim_right)

        intersection_right = line_intersection([x[0],y_slab[0]],[x[-1],y_slab[-1]],[x[0],y_lim_left[0]],[x[-1],y_lim_left[-1]])

        for ii in range(len(lons)):

            if intersection_left[0] <= lons[ii] and lons[ii] <= intersection_right[0] and intersection_left[1] <= lats[ii] and lats[ii] <= intersection_right[1] :
                
                xc_lim.append(lon[ii])
                yc_lim.append(lat[ii])

    # MANAGE THE OUTPUT MATRIX
    contours = np.hstack((np.array(xc_lim), np.array(yx_lim)))

    return contours

def line_intersection(a, b, c, d):

    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / ((a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))
    u = ((a[0] - c[0]) * (a[1] - b[1]) - (a[1] - c[1]) * (a[0] - b[0])) / ((a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))

    # check if line actually intersect
    if (0 <= t and t <= 1 and 0 <= u and u <= 1):
        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
    else: 
        return False

def compute_contours(slab_grid, trench_depth, downdip_depth, contours_interval):

    #### THIS FUNCTION USED THE SLAB2 GRID FILE TO EXTRACT CONTOURS

    # 1) DEFINE THE LEVELS OF ISODEPTH TO EXTRACT
    levels = np.append(np.arange(downdip_depth, 0, contours_interval), trench_depth)

    # 2) EXTRACT THE CONTOUR LEVELS FROM THE SLAB GRID
    c = plt.tricontour(slab_grid[:,0], slab_grid[:,1], slab_grid[:,2],  levels = levels, alpha=0.0 )
    plt.close()

    # 3) SORT THE X,Y AND Z COORDINATES 
    contours_slab_x, contours_slab_y, contours_slab_z = [], [], []
    for i in range(len(levels)):
        contour = c.allsegs[i][:][0]  # for all element :, in level i
        for ii in range(len(contour[:,0])):
            contours_slab_x.append(contour[ii,0])
            contours_slab_y.append(contour[ii,1])
            contours_slab_z.append(levels[i])
    contours_slab = np.vstack((np.array(contours_slab_x), np.array(contours_slab_y), np.array(contours_slab_z))).T

    return contours_slab

def add_trench(model_contours,trench_model,trench_model_depth):

    # 2) RESAMPLE THE TRENCH (DONE ALONG THE X AXIS)
    n_samples = 1000
    trench_model = resample_trench(trench_model,n_samples)

    # 1) MAKE A DEPTH ARRAY 
    depth_array = trench_model_depth * np.ones((len(trench_model[:,0]),1))

    # 2) MANAGE THE OUTPUT MATRIX
    trench_matrix = np.hstack((trench_model, depth_array))
    contours = np.vstack((model_contours, trench_matrix))

    return contours

def resample_trench(trench_model,n_samples):

    ## THE INTERPOLATION IS DONE HERE ALONG THE X AXIS
    x_t = np.linspace(np.min(trench_model[:,0]),np.max(trench_model[:,0]),n_samples)
    f = scipy.interpolate.interp1d(trench_model[:,0],trench_model[:,1])
    y_t = f(x_t)

    # MANAGE THE OUTPUT FILE
    trench_res = np.hstack((np.reshape(x_t,(n_samples,1)), np.reshape(y_t,(n_samples,1))))

    return trench_res

def get_contours_polygon(slab): # slab as x, y, z for each contour

    #### THIS FUNCTION EXTRACTS THE SHALLOWEST AND DEEPEST CONTOURS FROM THE CONTOUR FILE OF THE SLAB2 MODEL

    # 1) FIND INDEXES THE SHALLOWEST AND DEEPEST CONTOURS
    shallow = np.where(slab[:,2] == np.max(slab[:,2]))[0] # Trench
    deep = np.where(slab[:,2] == np.min(slab[:,2]))[0]    # Defined downdip limit
    contours = np.zeros((len(shallow)+len(deep),2)) # INITIALIZE CONTOURS MATRIX

    # 2) GET THE POSITIONS OF THE CONTOURS (X,Y)
    for i in range(len(shallow)):
        contours[i,:] = slab[shallow[i],0:2]
    for i in range(len(deep)):
        contours[len(shallow)+len(deep)-i-1,:] = slab[deep[i],0:2]

    return contours # Return the shallowest and deepest contours to build a polygon

""""""""" """"""""" """"""""" """""""""


""""""""" MESH AND GEOMETRY FUNCTIONS """""""""
def make_mesh_jigsaw(contours_slab, len_edge):

    # 1) PREPARE BY MEASURING THE INPUT CONTOURS
    i_node = []
    for i in np.arange( contours_slab.shape[0] ):
        i_node.append( ((contours_slab[i,0],contours_slab[i,1]),0) )
    i_edge = []
    for i in np.arange( len(i_node) ):
        i_edge.append( ((i,i+1),0) )
    i_edge[-1] = ((i,0),0)

    # 2) INITIALIZE JIGSAWPY
    opts = jigsawpy.jigsaw_jig_t() # OPTIONS
    geom = jigsawpy.jigsaw_msh_t() # GEOMETRY
    hmat = jigsawpy.jigsaw_msh_t() # H MATRIX
    mesh = jigsawpy.jigsaw_msh_t() # MESH

    # 3) DEFINE THE GEOMETRY FROM THE CONTOURS 
    geom.mshID = "euclidean-mesh" # TYPE OF MESH
    geom.ndims = +2               # DIMENTION OF THE MESH
    geom.vert2 = np.array(i_node,dtype=geom.VERT2_t) # VERTICES
    geom.edge2 = np.array(i_edge,dtype=geom.EDGE2_t) # EDGES

    # 4) DEFINE THE OPTIONS OF THE MESH
    opts.hfun_hmax = len_edge /4.     # PUSH HFUN LIMITS
    opts.hfun_hmin = len_edge /4.     # PUSH HFUN LIMITS
    opts.mesh_dims = +2             # DIMENTIONS
    opts.optm_qlim = +.95           # OPTIMIZATION LIMIT

    ##### > CALCULATE THE MESH 
    jigsawpy.lib.jigsaw(opts,geom,mesh)

    return mesh


def extract_mesh(mesh):

    # 1) EXTRACT THE SIZE PROPERTIES OF THE MESH
    n_vertices = mesh.vert2.shape[0]
    n_edges = mesh.edge2.shape[0]
    n_triangles = mesh.tria3.shape[0]

    # 2) VERTICES
    vertices = np.zeros(( n_vertices , 3 ))
    for i in np.arange( n_vertices ):
        vertices[i,:2] = mesh.vert2[i][0]
    # 3) EDGES
    edges = np.zeros(( n_edges , 2 ))
    for i in np.arange( n_edges ):
        edges[i,:2] = mesh.edge2[i][0]
    # 4) FACES
    faces = np.zeros((n_triangles,3),dtype=int)
    for i in np.arange( n_triangles ):
        faces[i,:] = mesh.tria3[i][0]

    return vertices, edges, faces


def project_vertices(vertices,model_xyz):

    # 1) MESHGRID THE SLAB MODEL
    xv, yv = np.meshgrid(model_xyz[:,0], model_xyz[:,1], sparse=False, indexing='ij')

    # 2) INTERPOLATE THE VERTICIES OF THE MESH ON THE SLAB MODEL MESH
    slab_mesh = scipy.interpolate.griddata(np.c_[model_xyz[:,0], model_xyz[:,1]], model_xyz[:,2], (xv, yv), method='cubic')

    # 3) GET THE POINTS OF THE SLAB MESH THAT ARE CLOSEST TO THE VERTICES
    slab_mesh = slab_mesh.flatten()
    depths = []
    for i in range(len(vertices[:,0])):

        # FIND THE CLOSEST POINT
        idx = ( (xv-vertices[i,0])**2 + (yv-vertices[i,1])**2 ).argmin() 

        # MANAGE THE EDGE EFFECTS BY FINDING NaNs
        if np.isnan(slab_mesh[idx]) == True:
            if np.isnan(slab_mesh[idx+1]) == False:
                depths.append(slab_mesh[idx+1])
            else :
                depths.append(slab_mesh[idx-1])
        else:
            depths.append(slab_mesh[idx])
    
    # CHECK NaNs AGAIN
    for i in range(len(depths)):
        if np.isnan(depths[i]) == True:
            print('YOUR GRID HAS NaNs, TRY A DIFFERENT TRIANGLE SIZE OR CHECK YOUR SLAB MODEL')
    vertices[:,2] = np.array(depths) # ATTRIBUTE DEPTHS

    return vertices


def mk_rectangular_dislocation(x,y,z,strike,dip,width,height):

    alpha = np.radians(strike) + np.pi/2

    # TOP PART 
    X1 = np.array([x, y, z])
    x2 = x + np.cos(alpha)*height
    y2 = y + np.sin(alpha)*height
    X2 = np.array([x2,y2,z])

    # BOTTOM PART 
    delta_x = np.cos(np.radians(dip)) * np.cos(alpha - np.pi/2) * width
    delta_y = np.cos(np.radians(dip)) * np.sin(alpha - np.pi/2) * width
    delta_z = np.sin(np.radians(dip)) * width
    d = np.array([delta_x, delta_y, delta_z])
    
    X3 = X2 + d
    X4 = X1 + d

    return (X1,X2,X3,X4)

def geometry_dip_strike(vertices_xyz,mesh):

    # INITIALIZE THE ARRAYS
    areas = []
    dips, strikes = [], []
    xs,ys,zs = [], [], []
    # INITIALIZE THE GEOMETRY MATRIX
    n_triangles = mesh.tria3.shape[0]
    geometry = np.zeros(( n_triangles, 22 ))
    recs = np.zeros((n_triangles,12))

    for i in range(n_triangles): # LOOP OVER THE NUMBER OF TRIANGLES

        # GET THE VERTICES OF THE TRIANGLE
        [idx_v1, idx_v2, idx_v3] = mesh.tria3[i][0]
        AA = np.array([vertices_xyz[idx_v1][0], vertices_xyz[idx_v1][1], vertices_xyz[idx_v1][2]])
        BB = np.array([vertices_xyz[idx_v2][0], vertices_xyz[idx_v2][1], vertices_xyz[idx_v2][2]])
        CC = np.array([vertices_xyz[idx_v3][0], vertices_xyz[idx_v3][1], vertices_xyz[idx_v3][2]])

        # COMPUTE THE NORMAL VECTOR
        N = np.cross((BB-AA),(CC-AA)) # Cross product
        L = np.sqrt( N[0]**2 + N[1]**2 + N[2]**2 ) # LENGTH
        n = N/L # NORMALIZE
        A = 0.5*L # AREA, in km**2, S=0.5 * | B-A x C-A |
        areas.append(A)
        M = (AA+BB+CC)/3 # BARYCENTER OF THE TRIANGLE

        # ROTATE THE LOCAL FRAME TO GEO
        (lam,phi) = UTM_GEO(M[0],M[1])
        R = mat_rot_local_to_general(lam,phi,unit='dec_deg') # Rotate general to local ENU
        ENU = np.dot(R,n) # Apply the rotation matrix
        if ENU[2]<0 : ENU = -ENU # Upward normal vector

        # CALCULATE THE DIP 
        dip = np.degrees(np.arctan(np.sqrt(ENU[0]**2 + ENU[1]**2) / ENU[2])) - 90
        # CALCULATE THE STRIKE
        usv = np.array([-ENU[1], ENU[0], 0.]) / np.sqrt(ENU[0]**2 + ENU[1]**2) # UNIT STRIKE VECTOR
        strike = np.degrees(np.arctan2(usv[0], usv[1]))

        # POLYGONE OF THE QUASI-EQUILATERAL TRIANGLE
        A_poly = 2*A
        Rec_width = np.sqrt(A_poly)
        Rec_height = Rec_width * np.sqrt(3)/2

        #########################
        (X1,X2,X3,X4) = mk_rectangular_dislocation(M[0], M[1], M[2], strike, dip, Rec_width, Rec_height) # for the face barycenter
        Rec_center = (X1+X2+X3+X4)/4
        d = M - Rec_center
        Center_dislocation = M - d
        ####################

        ### RECTANGLES
        recs[i,0:3], recs[i,3:6], recs[i,6:9], recs[i,9:12] = X1, X2, X3, X4

        # GEOMETRY 
        geometry[i,0], geometry[i,1], geometry[i,2] = Center_dislocation # DISLOCATION POSITION OF THE POLYGONE
        geometry[i,3], geometry[i,4], geometry[i,5] = Rec_width, Rec_height, A_poly # POLYGONE WIDTH, HEIGHT, AREA
        geometry[i,6] = A/A_poly # RATIO BETWEEN THE AREA OF THE RECTANGLE AND THE ONE OF THE TRIANGLE 
        geometry[i,7], geometry[i,8] = dip, strike # DIP, STRIKE
        geometry[i,9], geometry[i,10], geometry[i,11] = M[0], M[1], M[2] # CENTROID X, CENTROID Y, CENTROID Z
        geometry[i,12:15], geometry[i,15:18], geometry[i,18:21] = vertices_xyz[idx_v1], vertices_xyz[idx_v2], vertices_xyz[idx_v3] # VERTEXES (IN KM)
        geometry[i,21] = A # AREA

    
    return geometry, recs


def save_geometry(geometry):

    ## >>>>>>>>>> INITIATE THE GEOMETRY FILE OF THE MESH <<<<<<<<< ##
    file_geometry = open('Geom.txt','w')
    file_geometry.write('# Rectangle_X, Rectangle_Y, Rectangle_Dep, \
    Width_km, Height_km, Area_km, LonLatD_to_XYZ, \
    Dip, Strike, \
    Centroid_X, Centroid_Y, Centroid_Z, \
    Vertex1_X, Vertex1_Y, Vertex1_Z,\
    Vertex2_X, Vertex2_Y, Vertex2_Z,\
    Vertex3_X, Vertex3_Y, Vertex3_Z, Area\n')
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>> | <<<<<<<<<<<<<<<<<<<<<<<<<< ##

    # SAVE THE GEOMETRY
    for i in range(len(geometry[:,0])):
        for ii in range(len(geometry[i,:])):
            file_geometry.write('%.5f ' % geometry[i,ii])
        file_geometry.write('\n')
    file_geometry.close()

""""""""" """"""""" """"""""" """""""""
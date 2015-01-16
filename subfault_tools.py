import numpy as np
import os
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
from itertools import islice
import re


rad = np.pi/180.
rr = 6.378e6
lat2meter = rr*rad

####################################################################################################
def read_subfault_model(fname, columns, units=None, \
                    defaults = {'latlong_location': 'top center'}, \
                    skiprows=0, delimiter=None):
    """
    Read a subfault model and return a list of dictionaries specifying the
    Okada parameters for each subfault.
    The dictionary may also contain entries 'rupture_time', 'rise_time' and
    perhaps 'rise_time_ending' if these are specified in the file.

    The file *fname* is read in using loadtxt.  The first *skiprow* rows may be
    comments, after that there should be a row for each subfault.  The
    contents of the columns is specified by the input parameter *columns*,
    each of whose elements is one of the following:
        'latitude', 'longitude', 'length', 'width', 'depth',
        'strike', 'dip', 'rake', 'slip',
        'rupture_time', 'rise_time', 'rise_time_ending', 'ignore'
    Columns labelled 'ignore' will be ignored, others will be used to set
    the corresponding elements of the parameter dictionary for each subfault.

    If some Okada parameters are missing from the columns but have the same
    value for each subfault (e.g. 'latlong_location', 'length' or 'width'),
    set these in the *defaults* dictionary, e.g.
    defaults = {'latlong_location': 'centroid', 'length': 100e3, 'width': 50e3}

    A dictionary *units* can also be provided.  By default the units for
    param = 'length', 'depth', 'width', and 'slip' are assumed to be meters,
    but you can set units[param] = 'm' or 'km' or 'cm'.

    *delimiter* is the delimiter separating columns.  By default it is any
    whitespace but it could be ',' for a csv file, for example.

    """

    valid_labels = """latitude longitude strike dip rake slip length width
                      depth rupture_time rise_time rise_time_ending ignore""".split()

    if units is None:
        units = {}
        units['slip'] = 'm'
        units['length'] = 'm'
        units['width'] = 'm'
        units['depth'] = 'm'

    usecols = []
    for j,label in enumerate(columns):
        if label not in valid_labels:
            raise Exception("Unrecognized label in columns list: %s" % label)
        if label != 'ignore':
            usecols.append(j)

    try:
        data = np.genfromtxt(fname, skiprows=skiprows, \
                             delimiter=delimiter,usecols=usecols)
    except:
        raise Exception("Unable to load file %s" % fname)


    try:
        ncols = data.shape[1]
        nfaults = data.shape[0]
    except:
        nfaults = 1
        data = np.array([data])
        ncols = data.shape[1]

    print("Read %s faultslip datasets from %s" % (nfaults,fname))

    subfaults = []
    total_slip = 0.
    total_area = 0.
    for k in range(nfaults):
        subfault_params = copy.copy(defaults)
        for j in range(ncols):
            jj = usecols[j]
            #print "+++ j, jj, param: ",j,jj,columns[jj]
            if columns[jj] != 'ignore':
                subfault_params[columns[jj]] = data[k,j]
            else:
                raise Exception("This shouldn't happen!")
        #import pdb; pdb.set_trace()
        for param in ['slip','length','width','depth']:
            # convert to meters if necessary:
            if units.get(param, 'm') == 'km':
                subfault_params[param] = subfault_params[param] * 1e3
                #units[param] = 'm'
            elif units.get(param, 'm') == 'cm':
                subfault_params[param] = subfault_params[param] * 1e-2
                #units[param] = 'm'

        subfaults.append(subfault_params)
        subfault_slip = subfault_params['slip']*subfault_params['length'] \
                        * subfault_params['width']
        subfault_area = subfault_params['length']*subfault_params['width']
        total_slip += subfault_slip
        total_area += subfault_area
##        print('Subfault_area = %s [m^2]' %subfault_area)
        #print "Subfault slip*length*width = ",subfault_slip
        #import pdb; pdb.set_trace()
    print("Total area = ", total_area, " m**2")
    print("Total slip*length*width = ",total_slip, " m**3")
    if 1:
        mu = 4.e11  # Rigidity (shear modulus)  might not be right.
        Mo = mu*total_slip*(10**6)  # 0.1 factor to convert to Nm
        Mw = 2./3. * np.log10(Mo) - 10.7
        print("With rigidity mu = %6.1e, moment magnitude is Mw = %5.2f" % (mu,Mw))
    return subfaults, Mw
####################################################################################################

####################################################################################################
def set_fault_xy(faultparams):
    longitude = faultparams['longitude']
    latitude = faultparams['latitude']
    dip = faultparams['dip']
    strike = faultparams['strike']
    rake = faultparams['rake']
    length = faultparams['length']
    width = faultparams['width']
    depth = faultparams['depth']
    location = faultparams['latlong_location']
    ang_strike = strike*rad
    ang_dip = dip*rad
    ang_rake = rake*rad
    x0 = longitude
    y0 = latitude

    if location == "top center":
        depth_top = depth
        depth_centroid = depth +0.5*width*np.sin(ang_dip)
        depth_bottom = depth + width*np.sin(ang_dip)

        #convert fault origin from top of fault plane to bottom:
        del_x = width*np.cos(ang_dip)*np.cos(ang_strike) / (lat2meter*np.cos(y0*rad))
        del_y = -width*np.cos(ang_dip)*np.sin(ang_strike) / lat2meter

        x_top = x0
        y_top = y0
        x_bottom = x0+del_x
        y_bottom = y0+del_y
        x_centroid = x0+0.5*del_x
        y_centroid = y0+0.5*del_y

    else:
        raise ValueError("Unrecognized latlong_location: %s.\nPlease use 'top center'" % location)

    #distance along strike from center of an edge to corner:
    dx2 = 0.5*length*np.sin(ang_strike) / (lat2meter*np.cos(y_bottom*rad))
    dy2 = 0.5*length*np.cos(ang_strike) / lat2meter
    x_corners = [x_bottom-dx2, x_top-dx2, x_top+dx2, x_bottom+dx2, x_bottom-dx2]
    y_corners = [y_bottom-dy2, y_top-dy2, y_top+dy2, y_bottom+dy2, y_bottom-dy2]

    paramlist = """x_top y_top x_bottom y_bottom x_centroid y_centroid depth_top depth_bottom depth_centroid
x_corners y_corners""".split()

    for param in paramlist:
        cmd = "faultparams['%s'] = %s" % (param,eval(param))
        exec(cmd)
####################################################################################################

####################################################################################################
def plot_subfaults_basemap(subfaults, basemap,
                           plot_centerline=False, slip_color=False,
                           cmax_slip=None, cmin_slip=None, plot_rake=False,
                           xylim=None):
    """
    Plot each subfault projected onto the surface.
    Using a basemap projection defined by 'basemap'
    """
    max_slip = 0.
    min_slip = 0.
    for subfault in subfaults:
        slip = subfault['slip']
        max_slip = max(abs(slip), max_slip)
        min_slip = min(abs(slip), min_slip)
    print("Max slip, Min slip: ",max_slip, min_slip)

    if slip_color:
        cmap_slip = plt.cm.jet
        if cmax_slip is None:
            cmax_slip = max_slip+0.1
        if cmin_slip is None:
            cmin_slip = 0.

    y_ave = 0.
    for subfault in subfaults:

        set_fault_xy(subfault)

        # unpack parameters:
##        paramlist = """x_top y_top x_bottom y_bottom x_centroid y_centroid
##            depth_top depth_bottom depth_centroid x_corners y_corners""".split()
##
##        for param in paramlist:
##            cmd = "%s = subfault['%s']" % (param,param)
##            exec(cmd) in globals()
        x_top, ytop =subfault['x_top'], subfault['y_top']
        x_bottom, y_bottom = subfault['x_bottom'], subfault['y_bottom']
        x_centroid, y_centroid = subfault['x_centroid'], subfault['y_centroid']
        depth_top = subfault['depth_top']
        depth_bottom = subfault['depth_bottom']
        depth_centroid = subfault['depth_centroid']
        x_corners, y_corners = subfault['x_corners'],subfault['y_corners']


        y_ave += y_centroid


        # Plot projection of planes to x-y surface:
        if plot_centerline:
            basemap.plot([x_top],[y_top],'bo',label="Top center", latlon=True)
            basemap.plot([x_centroid],[y_centroid],'ro',label="Centroid", latlon=True)
            basemap.plot([x_top,x_centroid],[y_top,y_centroid],'r-', latlon=True)
        if plot_rake:
            if test_random:
                subfault['rake'] = 90. + 30.*(rand()-0.5)  # for testing
            tau = (subfault['rake'] - 90) * np.pi/180.
            basemap.plot([x_centroid],[y_centroid],'go',label="Centroid", latlon=True)
            dxr = x_top - x_centroid
            dyr = y_top - y_centroid
            x_rake = x_centroid + np.cos(tau)*dxr - np.sin(tau)*dyr
            y_rake = y_centroid + np.sin(tau)*dxr + np.cos(tau)*dyr
            basemap.plot([x_rake,x_centroid],[y_rake,y_centroid],'g-',linewidth=2, latlon=True)
        if slip_color:
            slip = subfault['slip']
            #c = cmap_slip(0.5*(cmax_slip + slip)/cmax_slip)
            #c = cmap_slip(slip/cmax_slip)
            s = min(1, max(0, (slip-cmin_slip)/(cmax_slip-cmin_slip)))
            c = cmap_slip(s)
            x_corners_m, y_corners_m = basemap(x_corners,y_corners)
            plt.fill(x_corners_m,y_corners_m,color=c,edgecolor='none')
        else:
            basemap.plot(x_corners, y_corners, 'k-', latlon=True)

    slipax = plt.gca()

    y_ave = y_ave / len(subfaults)
    slipax.set_aspect(1./np.cos(y_ave*np.pi/180.))
    plt.ticklabel_format(format='plain',useOffset=False)
    plt.xticks(rotation=80)
    if xylim is not None:
        plt.axis(xylim)
    plt.title('Fault planes')
####################################################################################################

####################################################################################################
#exec command does not export params...
def plot_subfaults(subfaults,plot_centerline=False, slip_color=False, \
            cmax_slip=None, cmin_slip=None, plot_rake=False, xylim=None):

    """
    Plot each subfault projected onto the surface.
    Describe parameters...
    """

    #plt.figure(202)
    #plt.clf()

    # for testing purposes, make random slips:
    test_random = False


    max_slip = 0.
    min_slip = 0.
    for subfault in subfaults:
        if test_random:
            subfault['slip'] = 10.*rand()  # for testing
            #subfault['slip'] = 8.  # uniform
        slip = subfault['slip']
        max_slip = max(abs(slip), max_slip)
        min_slip = min(abs(slip), min_slip)
    print("Max slip, Min slip: ",max_slip, min_slip)

    if slip_color:
        cmap_slip = plt.cm.jet
        if cmax_slip is None:
            cmax_slip = max_slip+0.1
        if cmin_slip is None:
            cmin_slip = 0.
        if test_random:
                        print("*** test_random == True so slip and rake have been randomized")

    y_ave = 0.
    for subfault in subfaults:

        set_fault_xy(subfault)

        # unpack parameters:
##        paramlist = """x_top y_top x_bottom y_bottom x_centroid y_centroid
##            depth_top depth_bottom depth_centroid x_corners y_corners""".split()
##
##        for param in paramlist:
##            cmd = "%s = subfault['%s']" % (param,param)
##            exec(cmd) in globals()
        x_top, ytop = subfault['x_top'], subfault['y_top']
        x_bottom, y_bottom = subfault['x_bottom'], subfault['y_bottom']
        x_centroid, y_centroid = subfault['x_centroid'], subfault['y_centroid']
        depth_top = subfault['depth_top']
        depth_bottom = subfault['depth_bottom']
        depth_centroid = subfault['depth_centroid']
        x_corners, y_corners = subfault['x_corners'],subfault['y_corners']


        y_ave += y_centroid


        # Plot projection of planes to x-y surface:
        if plot_centerline:
            plt.plot([x_top],[y_top],'bo',label="Top center")
            plt.plot([x_centroid],[y_centroid],'ro',label="Centroid")
            plt.plot([x_top,x_centroid],[y_top,y_centroid],'r-')
        if plot_rake:
            if test_random:
                subfault['rake'] = 90. + 30.*(rand()-0.5)  # for testing
            tau = (subfault['rake'] - 90) * np.pi/180.
            plt.plot([x_centroid],[y_centroid],'go',label="Centroid")
            dxr = x_top - x_centroid
            dyr = y_top - y_centroid
            x_rake = x_centroid + np.cos(tau)*dxr - np.sin(tau)*dyr
            y_rake = y_centroid + np.sin(tau)*dxr + np.cos(tau)*dyr
            plt.plot([x_rake,x_centroid],[y_rake,y_centroid],'g-',linewidth=2)
        if slip_color:
            slip = subfault['slip']
            #c = cmap_slip(0.5*(cmax_slip + slip)/cmax_slip)
            #c = cmap_slip(slip/cmax_slip)
            s = min(1, max(0, (slip-cmin_slip)/(cmax_slip-cmin_slip)))
            c = cmap_slip(s)
            plt.fill(x_corners,y_corners,color=c,edgecolor='none')
        else:
            plt.plot(x_corners, y_corners, 'k-')

    slipax = plt.gca()

    y_ave = y_ave / len(subfaults)
    slipax.set_aspect(1./np.cos(y_ave*np.pi/180.))
    plt.ticklabel_format(format='plain',useOffset=False)
    plt.xticks(rotation=80)
    if xylim is not None:
        plt.axis(xylim)
    plt.title('Fault planes')
    if slip_color:
        cax,kw = mpl.colorbar.make_axes(slipax)
        norm = mpl.colors.Normalize(vmin=cmin_slip,vmax=cmax_slip)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap_slip, norm=norm)
        #import pdb; pdb.set_trace()
    plt.sca(slipax) # reset the current axis to the main figure
####################################################################################################

####################################################################################################
# function to determine great circle distance between two coordinates
def greatcircle_distance(xa,ya,xb,yb):
    # where xa, ya are coordinates of start of great circle (point A)
    #       xb, yb are end coordinates of great circle  (Point B)

    rad = np.pi/180

    # convert coordinates to radians
    xa = xa*rad
    ya = ya*rad
    xb = xb*rad
    yb = yb*rad

    #cal. great circle distance A-B
    dist_AB = np.arccos(np.sin(ya)*np.sin(yb) + np.cos(ya)*np.cos(yb)*np.cos(xa-xb)) # distance in radians

    return dist_AB
####################################################################################################

####################################################################################################
# function to determine shortest distance from a point to a given line
def crosstrack_error(xa,ya,xb,yb,xd,yd):
    # where xa, ya are coordinates of start of great circle (point A)
    #       xb, yb are end coordinates of great circle  (Point B)
    #       xd, yd are coordinates of point off of great circle (point D)
    #   crosstrack error gives the shortest distance from point D to the great circle between A and B

    rad = np.pi/180.

    # convert coordinates to radians
    xa = xa*rad
    ya = ya*rad
    xb = xb*rad
    yb = yb*rad
    xd = xd*rad
    yd = yd*rad

    # calc. great circle A-D
    dist_AD = np.arccos(np.sin(ya)*np.sin(yd) + np.cos(ya)*np.cos(yd)*np.cos(xa-xd))

    # calc. great circl A-B
    dist_AB = np.arccos(np.sin(ya)*np.sin(yb) + np.cos(ya)*np.cos(yb)*np.cos(xa-xb))

    # calc. initial bearing from A-D
    if np.sin(xd-xa) < 0:
        bearing_AD = np.arccos((np.sin(yd) - np.sin(ya)*np.cos(dist_AD)) / (np.sin(dist_AD)*np.cos(ya)))
    else:
        bearing_AD = 2*np.pi - np.arccos((np.sin(yd) - np.sin(ya)*np.cos(dist_AD)) / (np.sin(dist_AD)*np.cos(ya)))

    # calc. inital bearing from A-B
    if np.sin(xb-xa) < 0:
        bearing_AB = np.arccos((np.sin(yb) - np.sin(ya)*np.cos(dist_AB)) / (np.sin(dist_AB)*np.cos(ya)))
    else:
        bearing_AB = 2*np.pi - np.arccos((np.sin(yb) - np.sin(ya)*np.cos(dist_AB)) / (np.sin(dist_AB)*np.cos(ya)))

    CTE = np.arcsin(np.sin(dist_AD)*np.sin(bearing_AD - bearing_AB)) #CTE in radians

    return CTE
####################################################################################################

####################################################################################################
# calculate slip distribution as function of downdip distance from upper edge
#   and width of seismogenic zone.  From Kelin Wang and Jiangheng He (2008).
#       - non trench-breaking rupture, unless add in a(1-x) as described in paper

def slip_wang(x,w,s0,b,q):
    """x = downdip distance from upper edge
       w = total downdip width
       s0 = maximum slip [m]
       q = skewness parameter (0 -> 1), lower q skews distribution updip
       b = broadness parameter (0 -> 0.3)"""

    x_prime = x/w
    if 0 <= x_prime <= q:
        delta = (6/(q**3))*(x_prime**2)*((q/2) - (x_prime/3))
    elif q < x_prime <= 1:
        delta = (6/((1-q)**3))*(1-x_prime)**2*(((1-q)/2) - ((1-x_prime)/3))
    else:
        delta = 0

    slip = s0*delta*(1+np.sin(np.pi*delta**b))

    return slip

####################################################################################################

####################################################################################################
# calculate slip distribution as function of downdip distance from upper edge and width of
# seismogenic zone. Scales to the same average slip as UDS model with b = 0.3, q = 0.5.
def slip_tohoku(x, w, s0):
	x_prime = x/w

	slip = 0.27115*s0*(np.exp(2*x_prime) - 1)

	return slip

####################################################################################################

####################################################################################################
def ZeroCenteredColorMap(cmap, vmin, vmax, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      vmin : lowest point in the colormap's range.
      midpoint : The new center of the colormap. Defaults to
          0.0 1 - vmax/(vmax + abs(vmin))
      vmax : highest point in the colormap's range.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }


    # regular index to compute the colors
    reg_index = np.linspace(0, 1, 257)

    # calculate the normalized center from 'midpoint'
    midpoint = 1 - vmax/(vmax+abs(vmin))

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129,endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

####################################################################################################

####################################################################################################
def load_tt3(fname):
    with open(fname) as myfile:
            header = list(islice(myfile,9))
    myfile.close

    ncols = np.int(re.findall('\d+',header[0])[0])
    nrows = np.int(re.findall('\d+',header[1])[0])
    ntimes = np.int(re.findall('\d+',header[2])[0])
    xlower = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[3])[0])
    ylower = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[4])[0])
    t0 = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[5])[0])
    dx = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[6])[0])
    dy = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[7])[0])
    dt = float(re.findall('-?\ *\d+\.?\d*(?:[Ee]\ *\+?\-?\ *\d+)?',header[8])[0])

    times = [t0+dt*nt for nt in range(ntimes)]  # times of each output in dtopo (in seconds)

    all_data = np.genfromtxt(fname,skiprows=9)

    dz_arrays = []
    startrow = 0
    for i in range(0,np.size(times)):
        endrow = startrow + len(all_data)/2
        dz_arrays.append(np.flipud(all_data[startrow:endrow]))
        startrow = endrow

    return times, dz_arrays, {'ncols':ncols, 'nrows':nrows, 'ntimes':ntimes, 'xlower':xlower, 'ylower':ylower,\
                              't0':t0, 'dx':dx, 'dy':dy, 'dt':dt}
####################################################################################################



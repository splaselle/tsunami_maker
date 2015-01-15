from __future__ import division, absolute_import, print_function, unicode_literals
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
from netCDF4 import Dataset as NetCDFFile
import os
from mpl_toolkits.basemap import Basemap, interp
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import sys
import urllib.request as urllib2
import urllib.parse as urlparse
import tempfile, logging
from math import asin, acos, atan2, cos, sin, pi
from shapely.geometry import Point, MultiPoint, Polygon, LineString
import scipy.ndimage
from PyQt4 import QtGui

#####################################################################################
#Downloads file from input url
def download_file(url, desc=None):
        u = urllib2.urlopen(url)

        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'
        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("\nDownloading: {0} Bytes: {1}]\n".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)
                print(status, end="")
            print()

        return filename
#####################################################################################

#####################################################################################
#Choose a region to download SLAB1.0 data for
def get_slab_data(ftype = None):
        """
        Download data files from the slab 1.0 website.  There are two types of gridded
        data: NetCDF (.grd) or ASCII (.xyz).  The coordinates for the NetCDF are in GMT,
        so difficult to use unless you're using the GMT plotter.  ASCII files are in
        lon,lat, but are 4 times larger than the NetCDF files....
        Default file type is 'ascii', so leave blank.  Use ftype = 'netcdf' if you want to download netcdf files
        """

        #GET NAMES OF ALL REGIONS FROM SLAB WEBSITE
        url = 'http://earthquake.usgs.gov/research/data/slab/'
        content = urlopen(url)
        soup = BeautifulSoup(content)

        table = soup.find('table', {'class':'slab ten column'})
        headers = []
        for tag in table.findAll('strong'):
                headers.append(tag)

        regions=[]
        for header in headers:
                regions.append(BeautifulSoup(str(header)).text)

        #PAIR REGION NAMES WITH ABBREVIATIONS
        region_abrv = collections.OrderedDict([\
        ('Alaska-Aleutians','alu'), \
        ('Central America','mex'), \
        ('Cascadia','cas'), \
        ('Izu-Bonin','izu'), \
        ('Kermadec-Tonga','ker'), \
        ('Kamchatka/Kurils/Japan','kur'), \
        ('Philippines','phi'), \
        ('Ryukyu','ryu'), \
        ('Santa Cruz Islands/Vanuatu/Loyalty Islands','van'), \
        ('Scotia','sco'), \
        ('Solomon Islands','sol'), \
        ('South America','sam'), \
        ('Sumatra-Java','sum')])

        print('SELECT REGION TO DOWNLOAD DATA FOR')
        for i,k in enumerate(regions):
                print('%s: %s' %(i+1,k))


        select_region = input('Input Region #: ')

        try:
                proceed = input("\nYou've selected %s. Proceed with download? (y/n): " %regions[int(select_region) -1])
        except ValueError:
                print('\nRestart script. Input the integer corresponding to region desired\n')


        if proceed == 'y':
                abrv = region_abrv[regions[int(select_region)-1]]
                foldername = str("SLAB1.0_%s_data" %abrv)
                print('\nfiles will download to: %s' %str(os.getcwd() + r'\%s\n' %foldername))
                if not os.path.exists(os.getcwd() + r'\%s' %foldername):
                    os.makedirs(os.getcwd() + r'\%s' %foldername)
                else:
                    overwrite = input('\nDirectory %s already exists.  Overwrite files in %s? (y/n):' %(foldername,foldername))

                if overwrite == 'y':
                        pass
                else:
                        print('\n\naborting script\n\n')
                        sys.exit()


                if not ftype:
                        file_urls = ['http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_clip.xyz' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_strclip.xyz' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_dipclip.xyz' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0.clip' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_top.in' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_base.in' %abrv]
                elif ftype == 'netcdf':
                        file_urls = ['http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_clip.grd' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_strclip.grd' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0_dipclip.grd' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_slab1.0.clip' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_top.in' %abrv,\
                             'http://earthquake.usgs.gov/research/data/slab/models/%s_base.in' %abrv]


                os.chdir(os.getcwd() + '\%s' %foldername)


                for url in file_urls:
                    filename = download_file(url)
                    print('\n\nDownloaded %s\n\n\n\n\n\n\n\n\n\n' %filename)

                os.chdir('..')


        elif proceed == 'n':
                print('\n\naborting script\n\n')
                sys.exit()
        else:
                print("\nPlease input y or n\n")
                input("You've selected %s. Proceed with download? (y/n): " %regions[int(select_region) -1])
#####################################################################################

#####################################################################################
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    """from johndcook.com
    The following code returns the distance between 2 locations based on each point's lon and lat.  The distance
    returned is relative to Earth's radius.  TO get distance in miles, multiply by 3960.  Km: multiply by 6373.

    Lat is measured in degrees north of the equator, southern locations have negative lats.  Similarly, longitude
    is measured in degrees east of the Prime Meridian.  A location 10 degrees west of the P.M., for example, could
    be expressed as either 350 east , or as -10 east.

    Assumes earth is spherical."""

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cosine = (sin(phi1)*sin(phi2)*cos(theta1 - theta2) +
           cos(phi1)*cos(phi2))
    arc = acos( cosine )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc
#####################################################################################

#####################################################################################
def endpoint_from_bearing_dist(bearing, dist, lon1, lat1):
    """Given the starting coordinates (lon1, lat1), bearing (azimuth), and distance (km),
    calculate the end coordinates (lon2,lat2)"""
    def deg2rad(angle):
        return angle*np.pi/180

    def rad2deg(angle):
        return angle*180/np.pi


    rEarth = 6371.01 # Earth's average radius in km
    epsilon = 0.000001 # threshold for floating-point equality

    theta = deg2rad(bearing) # degrees to radians
    rlon1 = deg2rad(lon1)
    rlat1 = deg2rad(lat1)
    rdistance = dist / rEarth #normalize linear dist to radian angle

    rlat2 = asin( sin(rlat1) * cos(rdistance) + cos(rlat1) * sin(rdistance) * cos(theta) )

    if cos(rlat2) == 0 or abs(cos(rlat2)) < epsilon: # Endpoint a pole
    	rlon2=rlon1
    else:
    	rlon2 = rlon1 + atan2(sin(theta)*sin(rdistance)*cos(rlat1), cos(rdistance) - sin(rlat1)*sin(rlat2))

    lon1, lat1 = rad2deg(rlon1), rad2deg(rlat1)
    lon2, lat2 = rad2deg(rlon2), rad2deg(rlat2)

    return {'lon1': lon1, 'lat1': lat1, 'lon2': lon2, 'lat2': lat2, 
            'dist': dist, 'bearing':bearing}
#####################################################################################

#####################################################################################
def basemap_lcc(x,y,resolution = 'h'):
        """
        Center a basemap Lambert-Conformal projection to the x, y range of data desired
        """
        llcrnrlon = np.min(x)
        llcrnrlat = np.min(y)
        urcrnrlon = np.max(x)
        urcrnrlat = np.max(y)

        def secondlat(lat1, arc):
                degrees_to_radians = math.pi/180.0
                lat2 = (arc-((90-lat1)*degrees_to_radians))*(1./degrees_to_radians)+90
                return lat2

        def centerMap(lats,lons,scale):
                #Assumes -90 < Lat < 90 and -180 < Lon < 180, and
                # latitude and logitude are in decimal degrees
                earthRadius = 6378100.0 #earth's radius in meters

                northLat = max(lats)
                southLat = min(lats)
                westLon = max(lons)
                eastLon = min(lons)

                # average between max and min longitude
                lon0 = ((westLon-eastLon)/2.0)+eastLon

                # a = the height of the map
                b = distance_on_unit_sphere(northLat,westLon,northLat,eastLon)*earthRadius/2
                c = distance_on_unit_sphere(northLat,westLon,southLat,lon0)*earthRadius

                # use pythagorean theorom to determine height of plot
                mapH = pow(pow(c,2)-pow(b,2),1./2)
                arcCenter = (mapH/2)/earthRadius

                lat0 = secondlat(southLat,arcCenter)
                lat1 = southLat

                # distance between max E and W longitude at most souther latitude
                mapW = distance_on_unit_sphere(southLat,westLon,southLat,eastLon)*earthRadius

                return lat0,lon0,lat1,mapW*scale,mapH*scale

        lat_0, lon_0, lat_1, mapWidth,mapHeight = centerMap(y,x,1)

##        m  = Basemap(projection='lcc',resolution='h',width = mapWidth, height = mapHeight, lat_1 = lat_1,\
##                      lat_0 = lat_0, lon_0 = lon_0)
        m = Basemap(projection = 'lcc', resolution = 'h', llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat, lat_1 = lat_1, lon_0 = lon_0, lat_0 = lat_0)

        print('LAMBERT CONFORMAL PROJECTION:\nMap Width = %s\nMap Height = %s\nlon_0, lat_0 = %s, %s\nlat1 = %s' 
                % (mapWidth,mapHeight,lon_0,lat_0,lat_1))
        return m
#####################################################################################

#####################################################################################
def basemap_lonlat_grid(x,y):
    """Insert array or list of x and y from slab data, will return
        proper grids for basemap ('cylindrical' projection) and projection
    INPUT: x, y are arrays of coordinates
    OUTPUT: x, y, m  where x, y are grids in the projection 'm'"""
    urcrnrlon, urcrnrlat = max(x), max(y)
    llcrnrlon, llcrnrlat = min(x), min(y)

    m = Basemap(projection='cyl', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,\
                urcrnrlat=urcrnrlat, resolution='h')

    ny = y.size
    nx = x.size

    lons, lats = m.makegrid(nx,ny)
    x, y = m(lons, lats)

    return x, y, m
#####################################################################################

#####################################################################################
# Load netcdf file of slab
def load_slab_nc(fname):
    ncFile = NetCDFFile(fname)
    x = np.ma.array(ncFile.variables['x'])  # longitudes in degrees east
    x_wested = x-360.                       # longitudes in degrees west
    y = np.ma.array(ncFile.variables['y'])  # latitude
    z = np.array(ncFile.variables['z'])  # plate interface depth [km]
    z = np.ma.masked_invalid(z)             # mask nans

    return {'x':x, 'x_wested':x_wested, 'y':y, 'z':z}
#####################################################################################

#####################################################################################
def load_slab_xyz(fname):
    data = np.genfromtxt(fname)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    x_wested = x-360.

    return {'x':x, 'x_wested':x_wested, 'y':y, 'z':z}
#####################################################################################

#####################################################################################
def closest_on_line(line, lon, lat):
    dist = np.array([distance_on_unit_sphere(lat, lon, point[1], point[0]) for point in line])
    min_idx = np.where(dist == min(dist))
    return min_idx
#####################################################################################

#####################################################################################
def top2bot_edge(toplon,toplat,botlons,botlats):
    """Given a point (toplon, toplat), determine the lon, lat of the nearest point on the line defined by (botlons, botlats)m using
    a great circle distance function (slab_tools.distance_on_unit_sphere)"""
    distances = [distance_on_unit_sphere(toplat,toplon,botlats[i],botlons[i]) 
            for i in range(0,botlons.size)]
    index = np.where(distances == np.min(distances))

    return botlons[index], botlats[index]
#####################################################################################

#####################################################################################
def save2netcdf(fname,x,y,data):
    """given an m x 3 array, write to a xyz format text file"""


#####################################################################################

#####################################################################################
class slabfolderdialog(QtGui.QMainWindow):
    def __init__(self):
        super(slabfolderdialog, self).__init__()
        fol_name = []
        self.initUI()
    def initUI(self):
        self.setGeometry(300,300,350,300)
        self.setWindowTitle('Select directory containing SLAB regions')
        self.openfolderDialog()
    def openfolderDialog(self):
        fol_name = QtGui.QFileDialog.getExistingDirectory(self,'select directory',
                os.getcwd())
        self.fol_name = fol_name


#####################################################################################

#####################################################################################
class slabdata:
    def __init__(self,loc_name = None, directory = None):
        if not loc_name:
            try:
                name = str(input("\nType in region name (i.e. 'alu' for Aleutians):"))
                self.loc_name = name
            except ValueError:
                print('\nRestart script. Input region name\n')
        else:
            self.loc_name = loc_name
        if not directory:
            app = QtGui.QApplication(sys.argv)
            folderdialog = slabfolderdialog()
            self.directory = folderdialog.fol_name
        else:
            self.directory = directory


    def load_top_base(self, top_fname=None, base_fname=None):
    # load xyz of top and bottom edges
        if top_fname is not None:
            top_fname = top_fname
        else:
            top_fname = os.path.abspath(os.path.join(self.directory,
                str("SLAB1.0_%s_data/%s" % (self.loc_name, self.loc_name) + "_top.in")))
        top = np.genfromtxt(top_fname)
        top[:,0] = top[:,0]-360 # for some reason, top and base files are defined in +degrees east, unlike the netcdf files

        if base_fname is not None:
            base_fname = base_fname
        else:
            base_fname = os.path.abspath(os.path.join(self.directory,
                str("SLAB1.0_%s_data/%s" % (self.loc_name, self.loc_name) + "_base.in")))
        base = np.genfromtxt(base_fname)
        base = np.flipud(base)
        base[:,0] = base[:,0]-360

        self.top = top
        self.base = base


    def load_nc_depth(self, fname = None):
        #load netcdf of depth with lon (x) and lat (y) data
        if fname is not None:
            fname = fname
        else:
            fname =  os.path.abspath(os.path.join(self.directory,
                str("SLAB1.0_%s_data/%s" % (self.loc_name, self.loc_name) 
                    + str("_slab1.0_clip.grd"))))
        nc_data = load_slab_nc(fname)
        x = nc_data['x_wested']
        y = nc_data['y']
        depth = nc_data['z']

        self.x = x
        self.y = y
        self.depth =depth


    def load_nc_strike_dip(self, s_fname = None, d_fname = None):
    # load netcdf of strike and dip
        if s_fname is not None:
            s_fname = s_fname
        else:
            s_fname = os.path.abspath(os.path.join(self.directory,
                str("SLAB1.0_%s_data/%s" % (self.loc_name, self.loc_name) + 
                    str("_slab1.0_strclip.grd"))))
            strike = load_slab_nc(s_fname)['z']

        if d_fname is not None:
            d_fname = d_fname
        else:
            d_fname = os.path.abspath(os.path.join(self.directory,
                str("SLAB1.0_%s_data/%s" % (self.loc_name, self.loc_name) 
                    +str("_slab1.0_dipclip.grd"))))
            dip = load_slab_nc(d_fname)['z']
        self.strike = strike
        self.dip = dip


    # make a function to plot the contours so we don't have to write this out every time
    def plot_slab_contours(self):
        fig = plt.figure(figsize=(16,8))
        clevs=np.arange(-300,0,10)
        x,y = np.meshgrid(self.x,self.y)
        depth_contours = plt.contour(x,y,self.depth,clevs)
        cbar = plt.colorbar(depth_contours)
        cbar.set_label('depth [km]')
        plt.grid()
        return

    def extract_depth_line(self, desired_depth):
        x,y = np.meshgrid(self.x,self.y)
        contour = plt.contour(x, y, self.depth, [desired_depth])
        plt.close()
        paths = contour.collections[0].get_paths() # get the paths and then find the x,y points of the contour
        vertices = []
        for i in range(0, np.size(paths)):
            for j in paths[i].vertices:
                vertices.append([j[0],j[1]])
        vertices = np.array(vertices)
        return vertices

    def extract_line_from_edge(self,line=None, lon1=None, lat1=None, lon2=None, lat2=None):
        """The top edzage along a depth contour of a SLAB 1.0 model.  Formatted to read [location]_top.in or anything of the form
        [[lon0, lat0], [lon1, lat1], ... [lonN, latN]] for N points along line.
        Use [classname] = top_edge to give line data.
        Then use [classname].pick(lon1, lat1, lon2, lat2] to extract a subset of the original line
            - Must specify at least one value in each lon, lat pair
            - Input full coordinates if there are multiple values for a given lon or lat along the line"""

        if line is not None:
            line = line
        else:
            line = self.top
        # Break out arrays of just longitude and latitude
        lons = line[:,0]
        lats = line[:,1]

        # Calculate nearest indices of point on line
        argin = [lon1, lat1, lon2, lat2]
        if argin.count(None) > 2:
            print('Error: Need a lat and/or lon value for both points')
        elif argin.count(None) == 2:
            if [lon1, lon2] == [None, None]:
                idx1 = np.where(abs(lats-lat1) == min(abs(lats-lat1)))
                idx2 = np.where(abs(lats-lat2) == min(abs(lats-lat2)))
            if [lat1, lat2] == [None, None]:
                idx1 = np.where(abs(lons-lon1) == min(abs(lons-lon1)))
                idx2 = np.where(abs(lons-lon2) == min(abs(lons-lon2)))
            if [lon1, lat2] == [None, None]:
                idx1 = np.where(abs(lats-lat1) == min(abs(lats-lat1)))
                idx2 = np.where(abs(lons-lon2) == min(abs(lons-lon2)))
            if [lat1, lon2] == [None, None]:
                idx1 = np.where(abs(lons-lon1) == min(abs(lons-lon1)))
                idx2 = np.where(abs(lats-lat2) == min(abs(lats-lat2)))
        elif argin.count(None) == 1:
            if lon1 == None:
                idx1 = np.where(abs(lats-lat1) == min(abs(lats-lat1)))
                idx2 = closest_on_line(line, lon2, lat2)
            if lon2 == None:
                idx2 = np.where(abs(lats-lat2) == min(abs(lats-lat2)))
                idx1 = closest_on_line(line, lon1, lat1)
            if lat1 == None:
                idx1 = np.where(abs(lons-lon1) == min(abs(lons-lon1)))
                idx2 = closest_on_line(line, lon2, lat2)
            if lat2 == None:
                idx2 = np.where(abs(lons-lon2) == min(abs(lons-lon2)))
                idx1 = closest_on_line(line, lon1, lat1)
        else:
            idx1 = closest_on_line(line, lon1, lat1)
            idx2 = closest_on_line(line, lon2, lat2)

        # Extract between indices
        if idx1[0][0] > idx2[0][0]:
            temp = idx2
            idx2 = idx1[0][0]
            idx1 = temp[0][0]
        else:
            idx1 = idx1[0][0]
            idx2 = idx2[0][0]

        segment = line[idx1:idx2,:]

        return segment


    def make_polygon_mask(self, top_line=None, bottom_line=None):
        """ Will take the inputs of top_line and bottom_line
        and connect them into a shapely polygon
        The output 'mask' is a shapely polygon..."""
        if top_line is not None:
            top_line = top_line
        else:
            top_line = self.top

        if bottom_line is not None:
            bottom_line = bottom_line
        else:
            bottom_line = self.base

        lons = np.concatenate((top_line[:,0], bottom_line[:,0][::-1]))
        lats = np.concatenate((top_line[:,1], bottom_line[:,1][::-1]))
        shape_coords = np.transpose(np.vstack((lons, lats)))
        mask = Polygon(shape_coords)

        return mask


    def indices_for_extraction(self, mask=None):
        """Determine the indices of SLAB data to be extracted from under the mask defined in make_polygon_mask.
        INPUT:
            mask: the shapely polygon.
        OUTPUT:
            to_extract: an array of indices that represents the mask"""
        if mask is None:
            print("error in 'extraction_indices' Need to specify input for 'mask'")

        # condense the region to search through the rectangle defined by the corners of the mask
        exterior = np.asarray(mask.exterior)
        ll, ur = [min(exterior[:,0]),min(exterior[:,1])], [max(exterior[:,0]),max(exterior[:,1])]
        x_cut_i = np.where(np.logical_and(self.x >= ll[0], self.x <= ur[0]))[0]
        y_cut_i = np.where(np.logical_and(self.y >= ll[1], self.y <= ur[1]))[0]

        # now loop through all the points and test if point is in the
        # polygon
        to_extract = []
        for ix in x_cut_i:
            for iy in y_cut_i:
                test = mask.contains(Point(self.x[ix], self.y[iy]))
                if test == True:
                    to_extract.append([int(ix), int(iy)])

        return to_extract


    def extract_by_indices(self, data=None, indices=None):
        """Extract the SLAB data under the mask defined b 'indices'."""
        if indices is None:
            print("error in 'extract_by_indices' Need to specify input for 'indices'")
        if data is None:
            print("error in 'extract_by_indices' Need to specify input for 'data'")
        elif data is 'x':
            data = self.x
            extracted_data = np.array([data[row[0]] for row in indices])
        elif data is 'y':
            data = self.y
            extracted_data = np.array([data[row[1]] for row in indices])
        elif data is 'depth':
            data = self.depth
            extracted_data = np.array([data[row[1],row[0]] for row in indices])
        elif data is 'strike':
            data = self.strike
            extracted_data = np.array([data[row[1],row[0]] for row in indices])
        elif data is 'dip':
            data = self.dip
            extracted_data = np.array([data[row[1],row[0]] for row in indices])
        else:
            data = data
            extracted_data = np.array([data[row[1],row[0]] for row in indices])
        return extracted_data

    def extract_transect(self, dataout_line, data=None, data_x=None, data_y=None):
        if data is 'depth':
            data=self.depth
        elif data is 'strike':
            data=self.strike
        elif data is 'dip':
            data=self.dip
        elif data is None:
            print("Error in extract transect: need to specify input for 'data'")
        else:
            data=data

        if data_x is not None:
            data_x=data_x
        else:
            data_x=self.x
        if data_y is not None:
            data_y=data_y
        else:
            data_y=self.y

        dataout1 = interp(data, data_x, data_y, dataout_line[:,0],dataout_line[:,1], order=1)
        dataout2 = interp(data, data_x, data_y, dataout_line[:,0],dataout_line[:,1], order=0)

        for i in range(0,np.size(dataout1)):
            if dataout1[i] is np.ma.masked:
                if dataout2[i] is not np.ma.masked:
                    dataout1[i] = dataout2[i]
                else:
                    r = i

                    while dataout2[r] is np.ma.masked:
                        if r < np.size(dataout1) - 1:
                            r += 1
                    try:
                        right = dataout2[r]
                    except IndexError:
                        pass



                    l = i
                    while dataout2[l-1] is np.ma.masked:
                        l += -1
                    try:
                        left = dataout2[l-1]
                    except IndexError:
                        pass

                    dataout1[i] = np.average([right,left])

        return dataout1



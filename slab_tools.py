from __future__ import division, absolute_import, print_function, unicode_literals
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
from netCDF4 import Dataset as NetCDFFile
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import sys
import urllib.request as urllib2
import urllib.parse as urlparse
import tempfile, logging
import math

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
# Load netcdf file of slab
def load_slab_nc(fname):
        ncFile = NetCDFFile(fname)
        x = np.ma.array(ncFile.variables['x'])  # longitudes in degrees east
        x_wested = x-360.                       # longitudes in degrees west
        y = np.ma.array(ncFile.variables['y'])  # latitude
        z = np.ma.array(ncFile.variables['z'])  # plate interface depth [km]
        z = np.ma.masked_invalid(z)             # mask nans

        return {'x':x, 'x_wested':x_wested, 'y':y, 'z':z}
#####################################################################################

def load_slab_xyz(fname):
        data = np.genfromtxt(fname)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        x_wested = x-360.

        return {'x':x, 'x_wested':x_wested, 'y':y, 'z':z}

#####################################################################################
# from johndcook.com
"""The following code returns the distance between 2 locations based on each point's lon and lat.  The distance
returned is relative to Earth's radius.  TO get distance in miles, multiply by 3960.  Km: multiply by 6373.

Lat is measured in degrees north of the equator, southern locations have negative lats.  Similarly, longitude
is measured in degrees east of the Prime Meridian.  A location 10 degrees west of the P.M., for example, could
be expressed as either 350 east , or as -10 east.

Assumes earth is spherical."""
def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

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

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc
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

        print('LAMBERT CONFORMAL PROJECTION:\nMap Width = %s\nMap Height = %s\nlon_0, lat_0 = %s, %s\nlat1 = %s' % (mapWidth,mapHeight,lon_0,lat_0,lat_1))
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

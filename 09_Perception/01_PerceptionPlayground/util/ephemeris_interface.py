import ephem 
import geocoder
from datetime import datetime
import math

def _azimuthElevationToCartesian(azimuth_deg, elevation_deg):
  azimuth_rad   = math.radians(azimuth_deg)
  elevation_rad = math.radians(elevation_deg)
  
  x =   math.cos(elevation_rad) * math.cos(azimuth_rad)
  y =   math.cos(elevation_rad) * math.sin(azimuth_rad)
  z = - math.sin(elevation_rad)
  return [x, y, z]

def _getCoordsFromIp():
  # Get geo Location from IP 
  geoLoc = geocoder.ip('me')
  coordinates = geoLoc.latlng
  thisLat  = coordinates[0]
  thisLong = coordinates[1]

  return thisLat, thisLong

def _getDateTime():
  # Get current time
  today  = datetime.now()
  today.strftime("%Y/%m/%d %H:%M:%S")
  return today.strftime("%Y/%m/%d %H:%M:%S")

def _getSunDirection(long_rad, lat_rad, dateTime):
  # Create sun ephemeris instance 
  sun = ephem.Sun()
  # create Observer
  obs      = ephem.Observer()
  obs.lat  = f'{lat_rad}'
  obs.lon  = f'{long_rad}'
  obs.date = f'{dateTime}'

  # Compute sun relative to observer
  sun.compute(obs)

  # Convert azimuth and elevation to degrees
  sunAzimuth_deg   = math.degrees(float(ephem.degrees(sun.az)))
  sunElevation_deg = math.degrees(float(ephem.degrees(sun.alt)))

  return sunAzimuth_deg, sunElevation_deg

def computeSunGravVecInNed():
  """
  _summary_ Compute sun and gravity vector in NED frame

  """

  # Compute this location longitude and latitude from IP address
  lat_rad, long_rad = _getCoordsFromIp()
  # Generate this date/time string
  dateTime = _getDateTime()

  # Compute sun azimuth and elevation wrt the observer
  sunAzimuth_deg, sunElevation_deg = _getSunDirection(long_rad, 
                                                      lat_rad, 
                                                      dateTime)

  print(f'Sun Azimuth   : {sunAzimuth_deg} [deg]')
  print(f'Sun Elevation : {sunElevation_deg} [deg]')

  # Normalized sun vector in NED frame
  sunVec_NED = _azimuthElevationToCartesian(sunAzimuth_deg, sunElevation_deg)
  print(f'Sun vector NED: {sunVec_NED}')

  # Normalized gravity vector in NED frame
  gravVec_NED = [0,0,1]
  print(f'Gravity vector NED: {gravVec_NED}')

  return sunVec_NED, gravVec_NED


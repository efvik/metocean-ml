import numpy as np

def dirmag_to_xy(wind_direction, wind_speed):
    '''
    Get wind x (east) and y (north) component 
    from wind speed and direction (degrees, going to).
    Assumes 0 = north and 90 = east etc.
    '''
    wind_direction = np.radians(wind_direction)
    windx = wind_speed*np.sin(wind_direction)
    windy = wind_speed*np.cos(wind_direction)
    return windx, windy

def xy_to_dirmag(x,y):
    '''
    Get wind direction (degrees, going to) and magnitude 
    from x (east) and y (north) components.
    Assumes 0 = north, 90 = east etc.
    '''
    dir = np.degrees(np.arctan2(y,x))
    dir = (90-dir)%360
    mag = np.sqrt(x**2+y**2)
    return dir, mag

def wind_waves_holthuijsen(wind,fetch=1e6,depth=1e6):
    '''
    Calculate wind from wind speed [m/s], water depth [m], and fetch distance [m],
    using the equations of holthuijsen, 2007, note 8B. The equations are considered
    equally valid for any depth or fetch, but consider only fetch in one direction.
    
    Arguments:
    ---------
    wind: float or array.
        The wind speed, in m/s. 
    fetch: float or array.
        The fetch, in meters. 
    depth: float or array.
        The water depth, in meters. 
        
    Returns:
    H_s : float or array.
        Significant wave height.
    T_s : float or array.
        Significant wave period.
    '''
    
    wind = np.array(wind)
    fetch = np.array(fetch)
    depth = np.array(depth)
    
    g = 9.80665
    d_hat = (g*depth)/np.power(wind,2)
    F_hat = (g*fetch)/np.power(wind,2)
    
    # Empirical constants, from the reference.
    H_inf = 0.24;   T_inf = 7.69
    k_1 = 4.41e-4;  k_2 = 2.77e-7
    k_3 = 0.343;    k_4 = 0.10
    m_1 = 0.79;     m_2 = 1.45
    m_3 = 1.14;     m_4 = 2.01
    p = 0.572;      q = 0.187

    H_hat = H_inf*np.power(np.tanh(k_3*np.power(d_hat,m_3)) * np.tanh((k_1*np.power(F_hat,m_1))/(np.tanh(k_3*np.power(d_hat,m_3)))),p)
    T_hat = T_inf*np.power(np.tanh(k_4*np.power(d_hat,m_4)) * np.tanh((k_2*np.power(F_hat,m_2))/(np.tanh(k_4*np.power(d_hat,m_4)))),q)

    H_s = H_hat*np.power(wind,2)/g
    T_s = T_hat*wind/g
    return H_s, T_s

def hopkins_spherical_to_zernike(W040=None, W060=None, W080=None, rms_norm=False):
    ''' Computes the corresponding zernike coefficients for a set of Hopkins spherical aberration coefficients subject to
        the minimum RMS wavefront error focus condition.
    
    Args:
        W040 (`float`): third-order spherical coefficient.
        
        W060 (`float`): fifth-order spherical coefficient.
        
        W080 (`float`): seventh-order spherical coefficient.
    
    Returns:
        `tuple` containing base-1 fringe zernike coefficients Z8, Z15, Z24.
            The output will only contain zernike terms of high enough order to represent the original wavefront.
            i.e. if W080 is given, then z8, z15, and z24 will all be present.  If W060 is given, z8 and z15 will appear.
            If only W040 is given, then only z8 will appear.
    '''
    # first, a list of the rho^0x0 components of each fringe zernike term
    z24_w080 = 70
    z24_w060 = -140
    z24_w040 = 90
    
    z15_w060 = 20
    z15_w040 = -30
    
    z8_w040 = 6
      
    # then, their norms
    z24_norm = 3
    z15_norm = sqrt(7)
    z8_norm = sqrt(5)
    
    # initialize some variables
    z8, z15, z24 = 0, 0, 0
    return_z8, return_z15, return_z24 = False, False, False
    
    if rms_norm is False:
        if W080 is not None:
            # set Z24 to the appropriate scale
            z24 += W080/z24_w080          
            
            # set z15 such that the rho^6 component of the overall description is 0
            z15 -= (z24*z24_w060)/z15_w060
            
            # set z8 such that the rho^4 component of Z24 is 0
            z8 += z24_w040*z24
            # and Z15
            z8 += z15_w040*z15
            # scale appropriately
            z8 /= z8_w040
            
            return_z24 = True
        
        if W060 is not None:
            z15 += W060/z15_w060
            z8  += W060/z15_w040
            return_z15 = True
        
        if W040 is not None:
            z8 += W040/z8_w040
            return_z8 = True
    
    #return (0.25, 1/10, 1/70)
    if return_z24:
        return (z8, z15, z24)
    elif return_z15:
        return (z8, z15)
    else:
        return z8
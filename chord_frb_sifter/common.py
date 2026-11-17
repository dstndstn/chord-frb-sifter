
## In seconds
def dispersion_delay(dm, freq_MHz):
    k_dm = 4.148808e6
    return 1e-3 * k_dm * dm / (freq_MHz * freq_MHz)

# In degrees
def ra_difference(ra1, ra2):
    d = ra1 - ra2
    d = d + -360 * (d >  180)
    d = d +  360 * (d < -180)
    return d

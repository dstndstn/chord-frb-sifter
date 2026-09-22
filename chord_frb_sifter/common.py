
## In seconds
def dispersion_delay(dm, freq_MHz):
    k_dm = 4.148808e6
    return 1e-3 * k_dm * dm / (freq_MHz * freq_MHz)



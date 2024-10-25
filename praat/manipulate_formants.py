## Adjusted from Praat Vocal Toolkit plugin (https://www.praatvocaltoolkit.com/index.html) 
import parselmouth
from parselmouth.praat import call
import numpy as np

# Main formant manipulation and intensity copying function
def change_formants(sound, f1_target, f2_target, f3_target, max_formant):
    original_duration = sound.get_total_duration()
    midpoint = original_duration / 2
    original_intensity = sound.get_intensity()
    sf1 = sound.sampling_frequency

    formant = call(sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    vf1 = call(formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
    vf2 = call(formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
    vf3 = call(formant, "Get value at time", 3, midpoint, "Hertz", "Linear")

    df1 = f1_target - vf1
    df2 = f2_target - vf2
    df3 = f3_target - vf3

    hf = call(sound, "Filter (stop Hann band)", 0, max_formant, 100) 

    sf2 = max_formant * 2
    rs1 = call(sound, "Resample", sf2, 10)

    formant2 = call(sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)

    lpc1 = call(rs1, "To LPC (autocorrelation)", 10, 0.025, 0.005, 50) ## maybe we should try other options (burg, etc.)
    source = call([rs1, lpc1], "Filter (inverse)")

    if df1 != 0:
        call(formant, "Formula (frequencies)", f"if row = 1 then self + {df1} else self fi")
    if df2 != 0:
        call(formant, "Formula (frequencies)", f"if row = 2 then self + {df2} else self fi")
    if df3 != 0: 
        call(formant, "Formula (frequencies)", f"if row = 3 then self + {df3} else self fi")

    fGrid = call(formant, "Down to FormantGrid")
    new_sound = call([source, fGrid], "Filter")

    rs2 = call(new_sound, "Resample", sf1, 10)
    new_sound = rs2.values + hf.values
    
    # Create a new sound with the combined values and the same sampling frequency
    new_sound = parselmouth.Sound(new_sound, sampling_frequency=rs2.sampling_frequency)
    new_formant = call(new_sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    nvf1 = call(new_formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
    nvf2 = call(new_formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
    nvf3 = call(new_formant, "Get value at time", 3, midpoint, "Hertz", "Linear")
    return new_sound

def copy_mean_intensity(original, changed):
    original_intensity = original.get_intensity()
    call(changed, "Scale intensity", original_intensity)
    return changed

def original(sound, f1_target, f2_target, f3_target, max_formant):
    return sound

# Example usage with correct file path
sound = parselmouth.Sound("../corner_vowels/DR1_AKS0_SA1_11_dark_aa.wav")
fsound = change_formants(sound, 500.0, 1500.0, 2500.0, 5500)
result = copy_mean_intensity(sound, fsound)
result.save("test.wav", "WAV")

# result2 = original(sound, 500.0, 1500.0, 2500.0, 5500)
# result2.save("original.wav", "WAV")


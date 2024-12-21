## Adjusted from Praat Vocal Toolkit plugin (https://www.praatvocaltoolkit.com/index.html) 
import parselmouth
from parselmouth.praat import call
import numpy as np

# Main formant manipulation and intensity copying function
def change_formants(sound, f1_target, f2_target, f3_target, max_formant):
    original_duration = sound.get_total_duration()
    original_intensity = sound.get_intensity()
    sf1 = sound.sampling_frequency

    formant = call(sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    vf1 = call(formant, "Get mean", 1, 0.0, 0.0, "Hertz")
    vf2 = call(formant, "Get mean", 2, 0.0, 0.0, "Hertz")
    vf3 = call(formant, "Get mean", 3, 0.0, 0.0, "Hertz")

    df1 = f1_target - vf1
    df2 = f2_target - vf2
    df3 = f3_target - vf3

    hf = call(sound, "Filter (stop Hann band)", 0, max_formant, 100) 

    sf2 = max_formant * 2
    rs1 = call(sound, "Resample", sf2, 10) ## resample to double the max formant

    lpc1 = call(rs1, "To LPC (autocorrelation)", 16, 0.005, 0.005, 50) ## maybe we should try other options (burg, etc.)
    source = call([lpc1, rs1], "Filter (inverse)") ## inverse filter to get source

    ## calculate new formants
    call(formant, "Formula (frequencies)", f"if row = 1 then self + {df1} else self fi") 
    call(formant, "Formula (frequencies)", f"if row = 2 then self + {df2} else self fi")
    call(formant, "Formula (frequencies)", f"if row = 3 then self + {df3} else self fi")

    fGrid = call(formant, "Down to FormantGrid")
    new_sound = call([source, fGrid], "Filter")

    rs2 = call(new_sound, "Resample", sf1, 10) ## resample to original sampling frequency
    new_sound = rs2.values + hf.values
    
    # Create a new sound with the combined values and the same sampling frequency
    new_sound = parselmouth.Sound(new_sound, sampling_frequency=rs2.sampling_frequency)
    # new_formant = call(new_sound, "To Formant (burg)", 0, 5, max_formant, 0.025, 50)
    # nvf1 = call(new_formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
    # nvf2 = call(new_formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
    # nvf3 = call(new_formant, "Get value at time", 3, midpoint, "Hertz", "Linear")
    return new_sound

def original(sound, f1_target, f2_target, f3_target, max_formant):
    return sound

# Example usage with correct file path
if __name__ == "__main__":
    sound = parselmouth.Sound("/data/user_data/eyeo2/data/CP/analyze_vowels/ADG0/DR4_ADG0_SX379_6_gives_ih.wav")
    result = change_formants(sound, 550, 1770, 2490, 5500)
    # result = copy_mean_intensity(sound, fsound)
    result.save("eh.wav", "WAV")

    # result2 = original(sound, 500.0, 1500.0, 2500.0, 5500)
    # result2.save("original.wav", "WAV")


; PERCEPTION AND PRODUCTION OF ENGLISH VOWELS BY BRAZILIAN EFL SPEAKERS
; Appendix K

form Play a sine wave
    integer f0
    integer f1
    integer f2
endform

createFolder: "./audios"
filename$ = "./audios/" + string$ (f0) + "_" + string$ (f1) + "_" + string$ (f2) + ".wav"

; f0
pitchTier = Create PitchTier: "f0", 0, 1.0
Add point: 0.0, f0
Add point: 1.0, f0
To PointProcess
;To Sound (phonation): 44100, 0.6, 0.05, 0.7, 0.03, 3.0, 4.0
To Sound (phonation): 44100, 1, 0.01, 0.7, 0.01, 3, 4
;To Sound (phonation): 16000, 0.6, 0.05, 0.7, 0.03, 3.0, 4.0
;To Sound (phonation): 16000, 1, 0.01, 0.7, 0.01, 3, 4

;removeObject: pitchTier, pulses
;selectObject: source
;plusObject: "Sound f0"
;Multiply
;Rename: "source"

; f1, f2
f3 = max (2500, f2 + 500)
f4 = max (3500, f3 + 400)
f5 = max (4000, f4 + 600)
f6 = f5 + 1000
f7 = f6 + 1000
f8 = f7 + 1000
f9 = f8 + 1000
f10 = f9 + 1000

for i to 10
    Filter with one formant (in-line)... f'i' sqrt(80^2+(f'i'/20)^2)
endfor

;f2_f1 = f2 - f1
;Create FormantGrid: "filter", 0, 1.0, 2, f1, f2_f1, 60, 50
;plusObject: "Sound source"
;Filter (no scale)


;writeInfoLine: "Test", filename$
;Play
Write to WAV file: filename$

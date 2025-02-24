import midi2audio
fs = midi2audio.FluidSynth("[GD] Steinway Model D274.sf2")

def mid(file,out):
    fs.midi_to_audio(file,out)

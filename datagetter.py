import os
import torchaudio
import kernmanager
from collections import defaultdict
import pickle
import midimanager

notefreqdata = defaultdict(list)

base = os.fsencode("data\\music")

for composer in os.listdir(base):
    print(composer)
    for file in os.listdir(os.path.join(base,composer)):
        print(file)
        # notefreqdata[composer.decode()].append((file.decode(), (kernmanager.notefreq(os.path.join(base, composer, file).decode()))))
        # kernmanager.midi(os.path.join(base,composer,file).decode(),os.path.splitext(os.path.join(os.fsencode("data\\midi"),composer,file).decode())[0]+".mid")
        # midimanager.mid(os.path.splitext(os.path.join(os.fsencode("data\\midi"),composer,file).decode())[0]+".mid",os.path.splitext(os.path.join(os.fsencode("data\\wavs"),composer,file).decode())[0]+".wav")
# notefreqdata = dict(notefreqdata)
# notefreqdata = {k:{x[0]:dict(x[1]) for x in notefreqdata[k]} for k in notefreqdata.keys()}
#
# with open("notefreqs.csv","wb") as file:
#     pickle.dump(notefreqdata, file)

data = []
base = os.fsencode("data\\wavs")
for composer in os.listdir(base):
    print(composer)
    for file in os.listdir(os.path.join(base,composer)):
        print(file)
        waveform,sample_rate = torchaudio.load(os.path.join(base,composer,file).decode(),normalize=True)
        data.append((composer.decode(),torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)))

with open("spectrograms.csv","wb") as file:
    pickle.dump(data,file)
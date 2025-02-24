import music21
from collections import defaultdict

def notefreq(path):
    file = music21.converter.parse(path)
    return music21.analysis.pitchAnalysis.pitchAttributeCount(file)

def rhyfreq(path):
    frequencies = {}
    file = music21.converter.parse(path)
    piece = defaultdict(list)
    for e in file.elements:
        if type(e) == music21.stream.base.Part:
            for el in e.elements:
                if type(el) == music21.stream.base.Measure:
                    for nt in el.elements:
                        if type(nt) == music21.note.Note:
                            piece[el.measureNumber].append(nt)
                        if type(nt) == music21.chord.Chord:
                            for no in nt.notes:
                                piece[el.measureNumber].append(no)

    piece = list(piece.values())

def midi(path,out):
    file = music21.converter.parse(path)
    mid = music21.midi.translate.streamToMidiFile(file)
    mid.open(out,"wb")
    mid.write()
    mid.close()
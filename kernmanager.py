import music21
from collections import defaultdict
from music21 import features

import music21extension

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
    file = music21extension.CustomHumdrumFile(path)
    file.parseFilename()
    mid = music21.midi.translate.streamToMidiFile(file.stream)
    mid.open(out,"wb")
    mid.write()
    mid.close()

features = [
    features.native.MostCommonNoteQuarterLength,
    features.native.MostCommonNoteQuarterLengthPrevalence,
    features.jSymbolic.AverageMelodicIntervalFeature,
    features.jSymbolic.AverageNoteDurationFeature,
    features.jSymbolic.AverageNumberOfIndependentVoicesFeature,
    features.jSymbolic.ChromaticMotionFeature,
    features.jSymbolic.DirectionOfMotionFeature,
    features.jSymbolic.ImportanceOfBassRegisterFeature,
    features.jSymbolic.ImportanceOfHighRegisterFeature,
    features.jSymbolic.MelodicOctavesFeature,
    features.jSymbolic.MelodicFifthsFeature,
    features.jSymbolic.MelodicTritonesFeature,
    features.jSymbolic.MelodicThirdsFeature,
    features.jSymbolic.MostCommonPitchFeature,
    features.jSymbolic.MostCommonPitchClassPrevalenceFeature,
    features.jSymbolic.PitchVarietyFeature,
    features.jSymbolic.PrimaryRegisterFeature,
    features.jSymbolic.RangeFeature,
    features.jSymbolic.StepwiseMotionFeature,
]

def get_features(path):
    file = music21.converter.parse(path)
    data = []
    for f in features:
        print(f)
        fe = f(file)
        data.append(fe.extract().vector[0])
    return data
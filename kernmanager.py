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
    features.native.MostCommonNoteQuarterLength, #0
    features.native.MostCommonNoteQuarterLengthPrevalence, #1
    features.jSymbolic.AverageMelodicIntervalFeature, #2
    features.jSymbolic.AverageNoteDurationFeature, #3
    features.jSymbolic.AverageNumberOfIndependentVoicesFeature, #4
    features.jSymbolic.ChromaticMotionFeature, #5
    features.jSymbolic.DirectionOfMotionFeature, #6
    features.jSymbolic.ImportanceOfBassRegisterFeature, #7
    features.jSymbolic.ImportanceOfHighRegisterFeature, #8
    features.jSymbolic.MelodicOctavesFeature, #9
    features.jSymbolic.MelodicFifthsFeature, #10
    features.jSymbolic.MelodicTritonesFeature, #11
    features.jSymbolic.MelodicThirdsFeature, #12
    features.jSymbolic.MostCommonPitchFeature, #13
    features.jSymbolic.MostCommonPitchClassPrevalenceFeature, #14
    features.jSymbolic.PitchVarietyFeature, #15
    features.jSymbolic.PrimaryRegisterFeature, #16
    features.jSymbolic.RangeFeature, #17
    features.jSymbolic.StepwiseMotionFeature, #18
]

def get_features(path):
    file = music21.converter.parse(path)
    data = []
    for f in features:
        print(f)
        fe = f(file)
        data.append(fe.extract().vector[0])
    return data

def pitch_unigrams(path):
    file = music21.converter.parse(path)
    file = file.transpose(music21.interval.Interval(file.analyze("key").tonic,music21.pitch.Pitch("C")))
    topnotes = [[max(x.notes) for x in e.elements if type(x) == music21.chord.Chord] for e in file.chordify().elements if type(e) == music21.stream.Measure]
    topnotes = [x.pitch.midi for y in topnotes for x in y]
    return topnotes

def rhythm_unigrams(path):
    file = music21.converter.parse(path)
    topnotes = [[max(x.notes) for x in e.elements if type(x) == music21.chord.Chord] for e in file.chordify().elements if type(e) == music21.stream.Measure]
    return [x.duration.quarterLength for y in topnotes for x in y]
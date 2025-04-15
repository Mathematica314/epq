import music21
import overrides
import pathlib


class CustomHumdrumFile(music21.humdrum.spineParser.HumdrumFile):
    def __init__(self, filename: str|pathlib.Path|None = None):
        super().__init__(filename)
    @overrides.override
    def parseNonOpus(self, dataStream):
        '''
                The main parse function for non-opus data collections.
                '''
        self.stream = music21.stream.Score()

        self.maxSpines = 0

        # parse global comments and figure out the maximum number of spines we will have

        self.parsePositionInStream = 0
        self.parseEventListFromDataStream(dataStream)  # sets self.eventList and fileLength
        try:
            assert self.parsePositionInStream == self.fileLength
        except AssertionError:  # pragma: no cover
            raise music21.humdrum.spineParser.HumdrumException('getEventListFromDataStream failed: did not parse entire file')
        self.parseProtoSpinesAndEventCollections()

        self.spineCollection = self.createHumdrumSpines()

        #<my addition>
        self.spineCollection.spines = [s for s in self.spineCollection.spines if s.spineType != "dynam"]
        #</my addition>

        self.spineCollection.createMusic21Streams()
        self.insertGlobalEvents()
        for thisSpine in self.spineCollection:
            thisSpine.stream.id = 'spine_' + str(thisSpine.id)
        for thisSpine in self.spineCollection:
            if thisSpine.parentSpine is None and thisSpine.spineType == 'kern':
                self.stream.insert(thisSpine.stream)

        self.parseMetadata()
class kNote:
    def __init__(self,string):
        self.string = string
        self.split = ["","",""]
        for c in string:
            if c in "0123456789.":
                self.split[0] += c
            elif c in "#n-":
                self.split[2] += c
            else:
                self.split[1] += c
        if self.split[2] == "":
            self.accidental = 0
        else:
            self.accidental = "-n#".index(self.split[2])-1
        self.pitch = self.split[1][0].upper()
        if ord(self.split[1][0]) > 96:
            self.octave = len(self.split[1])+3
        else:
            self.octave = 4-len(self.split[1])
    def __repr__(self):
        return f"{self.split[0]} {self.pitch}{self.octave} ({self.split[1]}) {self.accidental}"
import note

filepath = "data/music/mozart/sonata05-1.krn"
file = [x.strip("\n").split("\t") for x in open(filepath).readlines() if x[:3] != "!!!"]
key = []
parts = 2
currentnotes = ["",""]

unwantedchars = ["(",")","J","L","K","k","/","\\","{","}"]

for debug,line in enumerate(file):
    rline = line.copy()
    if line[0][0] == "*":
        pass
        #instruction
    elif line[0][0] == "!":
        pass
        #instruction
    elif line[0][0] == "=":
        pass
        #barline
    else:
        line = line[:parts]
        for i in range(parts):
            for c in unwantedchars:
                line[i] = line[i].replace(c,"")
            if "r" in line[i]:
                line[i] = None
            if line[i] == ".":
                line[i] = currentnotes[i]
            else:
                currentnotes[i] = line[i]
        nline = []
        for l in line:
            if l is not None:
                for x in l.split(" "):
                    nline.append(note.kNote(x))

        print(debug,nline,rline)
print()
# PARTS
# JUNK
# JUNK
# JUNK
# JUNK
# JUNK
# JUNK
# STRUCTURE
# JUNK
# JUNK
# CLEF
# KEY

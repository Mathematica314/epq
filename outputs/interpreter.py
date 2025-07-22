import os
import pickle
import generalised
import torch
from generalised import GeneralNeuralNetwork
from matplotlib import pyplot as plt
from collections import defaultdict

data = {}

pickles = os.fsencode("mlps/pickles")

for file in os.listdir(pickles):
    with open(os.path.join(pickles,file).decode(), "rb") as f:
        data[tuple(file.decode().split(".")[0].split("_"))] = pickle.load(f)

composers = defaultdict(lambda: defaultdict(dict))

for key in data:
    if key[2] in ["tr","te"]:
        acc,loss = [x[::-1] for x in list(zip(*data[key][::-1]))]
        composers[int(key[0][0])][key[1]][key[2]]=(acc,loss)
    else:
        composers[int(key[0][0])][key[1]][key[2]] = data[key]

for number in composers:
    for scale in composers[number]:
        plt.plot(composers[number][scale]["tr"][1], label="train_loss")
        plt.plot(composers[number][scale]["te"][1], label="test_loss")
        plt.legend()
        plt.title(f"{number} composers, {scale} featureset")
        plt.show()
        plt.plot(composers[number][scale]["tr"][0], label="train_acc")
        plt.plot(composers[number][scale]["te"][0], label="test_acc")
        plt.plot([1/number for _ in range(100000)],label="random_classifier")
        plt.ylim(0,1)
        plt.legend()
        plt.title(f"{number} composers, {scale} featureset")
        plt.show()

        print(number,scale,str(round(max(composers[number][scale]["te"][0])*100,1))+"%",str(round(composers[number][scale]["te"][0][-1]*100,1))+"%")

        if scale == "small":
            smaller_datapoints = [1, 4, 6, 13, 15, 16]
        else:
            smaller_datapoints = []
        with open("/home/james/Documents/repositories/epq/fets.pkl", "rb") as file:
            data = pickle.load(file)
        ordered_composers = ["beethoven", "bach", "chopin", "mozart", "scarlatti", "joplin", "haydn"][:number]
        data = [(d[0], [x for i, x in enumerate(d[1]) if i not in smaller_datapoints]) for d in data if d[0] in ordered_composers]
        ds = generalised.GeneralDataset(0,False,data)
        dataloader = torch.utils.data.DataLoader(ds)
        model = composers[number][scale]["model"]
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()

        confusion = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to("cpu")
                pred = model(x)
                confusion[ordered_composers[pred.argmax(1).item()]][ordered_composers[y.item()]]+= 1
        #       print(number,scale,{k:{ka:va/sum(v.values()) for ka,va in v.items()}for k,v in confusion.items()})
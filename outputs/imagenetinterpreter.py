import pickle
from matplotlib import pyplot as plt
with open("alx.pkl","rb") as f:
    alx = pickle.load(f)

with open("sqz.pkl","rb") as f:
    sqz = pickle.load(f)

plt.plot([x[1] for x in alx[0]], label="train_loss")
plt.plot([x[1] for x in alx[1]], label="test_loss")
plt.legend()
plt.title("AlexNet")
plt.show()

plt.plot([x[0] for x in alx[0]], label="train_acc")
plt.plot([x[0] for x in alx[1]], label="test_acc")
plt.legend()
plt.title("AlexNet")
plt.show()

print(max([x[0] for x in alx[1]]))
print(alx[1][-1][0])

plt.plot([x[1] for x in sqz[0]], label="train_loss")
plt.plot([x[1] for x in sqz[1]], label="test_loss")
plt.legend()
plt.title("SqueezeNet")
plt.show()

plt.plot([x[0] for x in sqz[0]], label="train_acc")
plt.plot([x[0] for x in sqz[1]], label="test_acc")
plt.legend()
plt.title("SqueezeNet")
plt.show()

print(max([x[0] for x in sqz[1]]))
print(sqz[1][-1][0])


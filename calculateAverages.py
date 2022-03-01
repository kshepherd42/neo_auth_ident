tree = [0.36106195, 0.3699115,  0.37168142, 0.34867257, 0.37168142, 0.33274336, 0.36460177, 0.3380531, 0.34513274]
knn = [0.28318584, 0.28495575, 0.31150442, 0.31681416, 0.28141593, 0.29026549, 0.31150442, 0.27610619, 0.28849558]
treeAverage = 0
knnAverage = 0

for i in range(len(tree)):
    treeAverage = treeAverage + tree[i]
    knnAverage = knnAverage + knn[i]

treeAverage = treeAverage / len(tree)
knnAverage = knnAverage / len(knn)

print("Tree:\t")
print(treeAverage)
print("\nKNN:\t")
print(knnAverage)

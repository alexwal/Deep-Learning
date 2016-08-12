### Decision Trees and Random Forests in Torch
Build a decision tree or random forest with the ID3 Algorithm.
- Use simple_decision_trees.lua for creating decision trees and random forests from arbitrary numerical and categorical data. (See file contents for how to format data)

Running:

  `th simple_decision_trees.lua -data iris.txt -testprop 0.35 -trees 8 -levels 5 -bins 3 -bagging`
  
results in the construction of a committee of 8 trees which go no deeper than 5 levels deep and at each node make a maximum of a 3-way split. The dataset is iris.txt and the proportion of data withheld for testing is 0.35%.

- Use mnist_decision_trees.lua to run an implementation of a MNIST decision tree. (Download MNIST data from: http://pjreddie.com/projects/mnist-in-csv/)

Running:

`th mnist_decision_trees.lua -data mnist_train.csv -testprop 0.35 -trees 8 -levels 5 -bins 3 -bagging`

results in the construction of a committee of 8 trees which go no deeper than 5 levels deep and at each node make a maximum of a 3-way split. The dataset is mnist_train.csv and the proportion of data withheld for testing is 0.35%.

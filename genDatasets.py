import math
import numpy as np


# Notes:
# - Input variables are denoted by x and/or y, evaluated value is denoted by z
# - U(a, b, c) denotes c random points uniformly sampled between a and b for EACH input variable. For all Nguyen cases,
#   c is defined to be 20.
# - Training and test datasets use DIFFERENT random seeds


def gen_datasets(name, train_seed, test_seed):
    ztrain = np.zeros(20)  # 20 samples specified in paper
    ztest = np.zeros(20)  # 20 samples specified in paper

    # Cases that use dataset U(−1, 1, 20) and have ONE input variable
    if name in ["Nguyen-1", "Nguyen-2", "Nguyen-3", "Nguyen-4", "Nguyen-5", "Nguyen-6"]:
        # Generate uniform random training inputs
        np.random.seed(train_seed)
        xtrain = np.random.uniform(-1, 1, 20)

        # Generate uniform random testing inputs
        np.random.seed(test_seed)
        xtest = np.random.uniform(-1, 1, 20)

        # Generate value of specified expression evaluated for the generated inputs
        for i in range(0, 20):
            if name == "Nguyen-1":
                ztrain[i] = (xtrain[i] ** 3) + (xtrain[i] ** 2) + xtrain[i]
                ztest[i] = (xtest[i] ** 3) + (xtest[i] ** 2) + xtest[i]
            elif name == "Nguyen-2":
                ztrain[i] = (xtrain[i] ** 4) + (xtrain[i] ** 3) + (xtrain[i] ** 2) + xtrain[i]
                ztest[i] = (xtest[i] ** 4) + (xtest[i] ** 3) + (xtest[i] ** 2) + xtest[i]
            elif name == "Nguyen-3":
                ztrain[i] = (xtrain[i] ** 5) + (xtrain[i] ** 4) + (xtrain[i] ** 3) + (xtrain[i]) ** 2 + xtrain[i]
                ztest[i] = (xtest[i] ** 5) + (xtest[i] ** 4) + (xtest[i] ** 3) + (xtest[i] ** 2) + xtest[i]
            elif name == "Nguyen-4":
                ztrain[i] = (xtrain[i] ** 6) + (xtrain[i] ** 5) + (xtrain[i] ** 4) + (xtrain[i] ** 3) + (
                            xtrain[i] ** 2) + xtrain[i]
                ztest[i] = (xtest[i] ** 6) + (xtest[i] ** 5) + (xtest[i] ** 4) + (xtest[i] ** 3) + (xtest[i] ** 2)\
                            + xtest[i]
            elif name == "Nguyen-5":
                ztrain[i] = math.sin(xtrain[i] ** 2) * math.cos(xtrain[i]) - 1
                ztest[i] = math.sin(xtest[i] ** 2) * math.cos(xtest[i]) - 1
            elif name == "Nguyen-6":
                ztrain[i] = math.sin(xtrain[i]) + math.sin(xtrain[i] + (xtrain[i] ** 2))
                ztest[i] = math.sin(xtest[i]) + math.sin(xtest[i] + (xtest[i] ** 2))

        # Only returns training/testing data for ONE variable and the evaluated expression
        return xtrain, xtest, ztrain, ztest

    # Only case that uses U(0, 2, 20) and has ONE input variable
    elif name == "Nguyen-7":
        # Generate uniform random training inputs
        np.random.seed(train_seed)
        xtrain = np.random.uniform(0, 2, 20)

        # Generate uniform random testing inputs
        np.random.seed(test_seed)
        xtest = np.random.uniform(0, 2, 20)

        # Generate value of specified expression evaluated for the generated inputs
        for i in range(0, 20):
            ztrain[i] = math.log(xtrain[i] + 1) + math.log((xtrain[i] ** 2) + 1)
            ztest[i] = math.log(xtest[i] + 1) + math.log((xtest[i] ** 2) + 1)

        # Only returns training/testing data for ONE variable and the evaluated expression
        return xtrain, xtest, ztrain, ztest

    # Only case that uses U(0, 4, 20) and has ONE input variable
    elif name == "Nguyen-8":
        # Generate uniform random training inputs
        np.random.seed(train_seed)
        xtrain = np.random.uniform(0, 4, 20)

        # Generate uniform random testing inputs
        np.random.seed(test_seed)
        xtest = np.random.uniform(0, 4, 20)

        # Generate value of specified expression evaluated for the generated inputs
        for i in range(0, 20):
            ztrain[i] = math.sqrt(xtrain[i])
            ztest[i] = math.sqrt(xtest[i])

        # Only returns training/testing data for ONE variable and the evaluated expression
        return xtrain, xtest, ztrain, ztest

    # Cases that use dataset U(0, 1, 20) and have TWO input variables
    elif name in ["Nguyen-9", "Nguyen-10", "Nguyen-11", "Nguyen-12"]:
        # Generate uniform random training inputs
        np.random.seed(train_seed)
        xtrain = np.random.uniform(0, 1, 20)
        ytrain = np.random.uniform(0, 1, 20)

        # Generate uniform random testing inputs
        np.random.seed(test_seed)
        xtest = np.random.uniform(0, 1, 20)
        ytest = np.random.uniform(0, 1, 20)

        # Generate value of specified expression evaluated for the generated inputs
        for i in range(0, 20):
            if name == "Nguyen-9":
                ztrain[i] = math.sin(xtrain[i]) + math.sin(ytrain[i] ** 2)
                ztest[i] = math.sin(xtest[i]) + math.sin(ytest[i] ** 2)
            elif name == "Nguyen-10":
                ztrain[i] = 2 * math.sin(xtrain[i]) * math.cos(ytrain[i])
                ztest[i] = 2 * math.sin(xtest[i]) * math.cos(ytest[i])
            elif name == "Nguyen-11":
                ztrain[i] = xtrain[i] ** ytrain[i]
                ztest[i] = xtest[i] ** ytest[i]
            elif name == "Nguyen-12":
                ztrain[i] = (xtrain[i] ** 4) - (xtrain[i] ** 3) + (0.5 * (ytrain[i] ** 2)) - ytrain[i]
                ztest[i] = (xtest[i] ** 4) - (xtest[i] ** 3) + (0.5 * (ytest[i] ** 2)) - ytest[i]

        # Returns training/testing data for TWO variables and the evaluated expression
        return xtrain, xtest, ytrain, ytest, ztrain, ztest

    else:
        print("ERROR: Name not in known set of cases.")

if __name__ == '__main__':
        
    # Example for Nguyen expression with a single input variable, x
    x_train, x_test, z_train, z_test = gen_datasets("Nguyen-1", 2022, 6254)
    print("Training data (x_train and z_train) for Nguyen-1")
    print(x_train)
    print(z_train)

    # Example for Nguyen expression with two input variables, x and y
    x_train, x_test, y_train, y_test, z_train, z_test = gen_datasets("Nguyen-9", 2022, 6254)
    print("\nTraining data (x_train, y_train, and z_train) for Nguyen-9")
    print(x_train)
    print(y_train)
    print(z_train)

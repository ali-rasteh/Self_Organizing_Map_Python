import numpy as np
from os.path import isfile, join
from sklearn.utils import shuffle
from som import SOM

Dataset_num = 7
Train_ratio = 0.7
Data_shuffling = True
data_lines_start = {"banana.dat":8, "haberman.dat":9, "titanic.dat":9, "balance.dat":10, "hayes-roth.dat":10, "newthyroid.dat":11, "wine.dat":19}
data_lines_end = {"banana.dat":5307, "haberman.dat":314, "titanic.dat":2209, "balance.dat":634, "hayes-roth.dat":169, "newthyroid.dat":225, "wine.dat":196}
data_dim =  {"banana.dat":2, "haberman.dat":3, "titanic.dat":3, "balance.dat":4, "hayes-roth.dat":4, "newthyroid.dat":5, "wine.dat":13}
data_FV_size = data_dim
data_class_num = {"banana.dat":2, "haberman.dat":2, "titanic.dat":2, "balance.dat":3, "hayes-roth.dat":3, "newthyroid.dat":3, "wine.dat":3}
# data_PV_size = {"banana.dat":2, "haberman.dat":2, "titanic.dat":2, "balance.dat":3, "hayes-roth.dat":3, "newthyroid.dat":3, "wine.dat":3}
data_PV_size = {"banana.dat":1, "haberman.dat":1, "titanic.dat":1, "balance.dat":1, "hayes-roth.dat":1, "newthyroid.dat":1, "wine.dat":1}
dataset_train_flag = {"banana.dat":1, "haberman.dat":1, "titanic.dat":1, "balance.dat":1, "hayes-roth.dat":1, "newthyroid.dat":1, "wine.dat":1}
dataset_iteration = {"banana.dat":100, "haberman.dat":2000, "titanic.dat":200, "balance.dat":1000, "hayes-roth.dat":3000, "newthyroid.dat":2000, "wine.dat":3000}
output_range_convert = {"banana.dat":{'-1.0\n':1, '1.0\n':2}, "haberman.dat":{1:1 , 2:2}, "titanic.dat":{'-1.0\n':1, '1.0\n':2}, "balance.dat":{1:1, 2:2, 3:3}, "hayes-roth.dat":{' 1\n':1, ' 2\n':2, ' 3\n':3}, "newthyroid.dat":{'1\n':1, '2\n':2, '3\n':3}, "wine.dat":{' 1\n':1, ' 2\n':2, ' 3\n':3}}
string_convert = {"balance.dat": {" L\n":1, " B\n":2, " R\n":3}, "haberman.dat": {" positive\n":1, " negative\n":2}}
file_path = "../datasets/"
files_name= ["banana.dat", "haberman.dat", "titanic.dat", "balance.dat", "hayes-roth.dat", "newthyroid.dat", "wine.dat"]


for i in range(Dataset_num):
    if(dataset_train_flag[files_name[i]]):
        Data_lines_num = sum(1 for line in open(join(file_path, files_name[i])))
        Data_file = open(join(file_path, files_name[i]), 'r')

        line_index=0
        word_index=0
        data_num = data_lines_end[files_name[i]]-data_lines_start[files_name[i]]+1
        data = np.zeros((data_num, 1+data_dim[files_name[i]]))
        temp = np.zeros((1,1))

        for line in Data_file:
            if (line != "\n"):
                if line_index>= data_lines_start[files_name[i]]-1:
                    for word in line.split(','):
                        # print(word)
                        if "\n" in str(word):
                            word.replace("\n", '')
                        if word_index>=0 and word_index<1+data_dim[files_name[i]]:
                            if((files_name[i]=="balance.dat" or files_name[i]=="haberman.dat") and word_index==data_dim[files_name[i]]):
                                data[line_index+1-data_lines_start[files_name[i]],word_index]=string_convert[files_name[i]][word]
                            else:
                                if(word_index==data_dim[files_name[i]]):
                                    data[line_index+1-data_lines_start[files_name[i]],word_index]=output_range_convert[files_name[i]][word]
                                else:
                                    data[line_index+1-data_lines_start[files_name[i]],word_index]=word
                        word_index+=1
            line_index+=1
            word_index=0

        # print(data[0:20,:-1])
        if Data_shuffling==True:
            data = shuffle(data)

        learning_rate = 0.01
        nodes_width = 1
        nodes_height = 15
        iterations = int(500000 / data_num)
        total_nodes_num = nodes_height * nodes_width
        train_data_num = int(data_num * Train_ratio)
        test_data_num = data_num - train_data_num
        train_input = data[:train_data_num, :-1]
        train_output = data[:train_data_num, -1]
        test_input = data[train_data_num:, :-1]
        test_output = data[train_data_num:, -1]
        test_prediction = np.zeros((test_data_num,1))
        train_best_match = np.zeros((train_data_num,1))
        test_best_match = np.zeros((test_data_num,1))
        nodes_class_score = np.zeros((total_nodes_num, data_class_num[files_name[i]]))
        nodes_best_class = np.zeros((total_nodes_num,1))

        print("Initialization for Dataset :", files_name[i])

        SOM_classifier=SOM(nodes_height,nodes_width,data_FV_size[files_name[i]],data_PV_size[files_name[i]],False, learning_rate, False)
        # SOM_classifier=SOM(data_class_num[files_name[i]],1,data_FV_size[files_name[i]],data_PV_size[files_name[i]],False, learning_rate)

        print("Training for Dataset :", files_name[i])
        # iterations = dataset_iteration[files_name[i]]
        # iterations = 2
        train_vector = []
        for j in range(train_data_num):
            train_vector.append([train_input[j], [train_output[j]]])
        SOM_classifier.train(iterations, train_vector, False)

        for j in range(train_data_num):
            train_best_match[j] = SOM_classifier.best_match(train_input[j])
            # print(train_best_match[j], int(train_best_match[j]), int(train_output[j]))
            nodes_class_score[int(train_best_match[j]), int(train_output[j])-1] += 1
        nodes_best_class = np.argmax(nodes_class_score, axis=1)


        print("Predictions for Dataset :", files_name[i])

        wrong_classified = 0
        for j in range(test_data_num):
            # print("test_input : ", test_input[j], "test_output : ", test_output[j])
            test_prediction[j] = round(SOM_classifier.predict(test_input[j])[0])
            test_best_match[j] = SOM_classifier.best_match(test_input[j])
            # print("test_prediction : ", test_prediction[j])
            # print("best_match : ", test_best_match[j])
            test_predict_class = nodes_best_class[int(test_best_match[j])]
            # print("test_predict_class : ", test_predict_class)
            if(test_predict_class != int(test_output[j])-1):
                wrong_classified += 1
        test_score = (test_data_num-wrong_classified)/test_data_num
        print("test_score for ", files_name[i], " = ", test_score)

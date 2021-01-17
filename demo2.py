import pandas as pd
import math
import numpy as np
class nativeBayesWithNormaltraining(object):
    def __init__(self):
        self.lapsample=None;#laplace
        self.prior_probability=None;
        self.condition_likelihood=None;
        self.finalclasses=None;#result
        self.prior_classes=None;#the classes of label
        self.bin =None

    def calculationOfPriorPro(self,training_label):
        self.prior_classes=np.unique(training_label)#types pf labels
        num_prior_classes=len(self.prior_classes)#num 0f types
        num_training_label=len(training_label)#num of training_label
        self.prior_probability= {};
        for each_class in self.prior_classes:
            traning_label_each_class= list(training_label).count(each_class)+self.lapsample
            self.prior_probability[each_class] = traning_label_each_class / (float(num_training_label)+num_prior_classes*self.lapsample)  # p = count(y) / count(Y)
            # print(each_class, self.prior_probability[each_class])

    def calcution_conditon_probability(self,eachsatifyCondition,EachtheWholeAttributeValue):
        juuge_redo=1;
        type_satify=EachtheWholeAttributeValue
        total_num_satify=float(len(eachsatifyCondition))
        satify_probability={}
        # for each_satity in type_satify:
        #     cal_satify_probability=np.sum(np.equal(eachsatifyCondition,each_satity))/total_num_satify
        #     satify_probability[each_satity]=cal_satify_probability;
        #     if cal_satify_probability==0:
        #         juuge_redo=1#when conditional probability = 0, use  laplace
        if juuge_redo==1:
            satify_probability = {}# clear the result
            for each_satity in type_satify:
                cal_satify_probability = (np.sum(np.equal(eachsatifyCondition, each_satity))+self.lapsample) / (total_num_satify+len(type_satify)*self.lapsample)
                satify_probability[each_satity] = cal_satify_probability;
        # print(satify_probability)
        return satify_probability

    def bin_distribution(self,satifyCondition,max_value,min_value):#each bin have at least one sample
        each_bin_size=(max_value-min_value)/float(self.bin)
        satisCondi=(np.array(satifyCondition)-min_value)/float(each_bin_size);
        total_binnum_satify = float(len(satifyCondition))
        simply_satis=[]
        for geshu in range(len(satifyCondition)):
            if(satifyCondition[geshu]>=max_value):
                bin_reach=self.bin
            if (satifyCondition[geshu] <= min_value ):
                bin_reach = 1
            bin_reach=math.ceil(satisCondi[geshu])
            # if(satifyCondition[geshu]<max_value and satifyCondition[geshu] > min_value):
            #     for num_size in range(1, self.bin + 1):
            #         if satifyCondition[geshu] >= ((num_size - 1) * each_bin_size + min_value) and satifyCondition[geshu] <= (num_size * each_bin_size + min_value):
            #             bin_reach = num_size
            #         if (satifyCondition[geshu] > (num_size * each_bin_size + min_value)):
            #             bin_reach = self.bin
            #         if (satifyCondition[geshu] < ((num_size - 1) * each_bin_size + min_value)):
            #             bin_reach = 1
            simply_satis.append(bin_reach)
        each_bin_probability={}
        each_bin_probability['mini'] = min_value
        each_bin_probability['maxi'] = max_value
        each_bin_probability['size']=each_bin_size
        for each_bin_satis in range(1,self.bin+1):
            each_bin_probability[each_bin_satis] = (np.sum(np.equal(simply_satis, each_bin_satis)) + self.lapsample) / (
                        total_binnum_satify + self.bin * self.lapsample)
        return each_bin_probability

    def judgenumber_float(self,strOfnum):
        # print(strOfnum)
        point = '.'
        if point in str(strOfnum):
            s = str(strOfnum).split('.')
            # print(s)
            if float(s[1]) == 0:
                return 1
            else:
                return 0
        else:
            return 1

    def calculationOfConditionalProbability(self,traning_attribute,training_label):
        self.condition_likelihood={}
        self.prior_classes = np.unique(training_label)  # types pf labels
        for each_class in self.prior_classes:
            self.condition_likelihood[each_class]={}
            for each_set in range(len(traning_attribute[0])):#num of features
                if self.judgenumber_float(traning_attribute[0][each_set])==1:#discretization num
                    satifyCondition = traning_attribute[np.equal(training_label,each_class)][:,each_set]
                    theWholeAttributeValue=np.unique(traning_attribute[:,each_set])
                     #np.equal(training_label,each_class) show the location satify the training_label[i]=each_class[j]
                    self.condition_likelihood[each_class][each_set]=self.calcution_conditon_probability(satifyCondition,theWholeAttributeValue)
                    # print("self.condition_likelihood",self.condition_likelihood)
                else:#Guassian
                    max_eachset=max(traning_attribute[:, each_set]);
                    min_eachset=min(traning_attribute[:, each_set]);
                    satifyCondition = traning_attribute[np.equal(training_label, each_class)][:, each_set]
                    self.condition_likelihood[each_class][each_set] =self.bin_distribution(satifyCondition,max_eachset,min_eachset)

class nativeBayesWithNormaltesting(object):
    def __init__(self):
        self.condition_likelihood_for_testing=None;
        self.prior_probability_for_testing=None;
        self.result_label=None;
        self.data_testing=None;
        self.prior_classes=None;
        self.bin=None

    def predict_sample_result(self,each_sample_test):
        max_posterior_prob={}
        # calculate the probability of each label given the testing data
        for index in range(len(self.prior_classes)):
            prior_label_pro = self.prior_classes[index]
            map_probability=1.0*self.prior_probability_for_testing[prior_label_pro]
            feature_probability=self.condition_likelihood_for_testing[self.prior_classes[index]]
            for each_attribute in range(len(each_sample_test)):
                if(self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute].__contains__('size')==False):
                    map_probability*=self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute][each_sample_test[each_attribute]]
                else:#use bin
                    size=self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['size']
                    max_range=self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['maxi']
                    min_range = self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['mini']

                    # print('max_range',max_range)
                    # print('min_range',min_range)
                    # print('size',size)
                    # print('each_sample_test[each_attribute]',each_sample_test[each_attribute])
                    reach_num_bin=(each_sample_test[each_attribute]-min_range)/float(size)
                    # print('reach_num_bin',reach_num_bin)
                    # bin_reach = 1
                    # for num_size in range(1,self.bin+1):
                    #     if each_sample_test[each_attribute]>((num_size-1)*size+min_range) and each_sample_test[each_attribute]<(num_size*size+min_range):
                    #         bin_reach=num_size
                    #     if(num_size==self.bin):
                    #         bin_reach=self.bin
                    if (each_sample_test[each_attribute] >= max_range):
                        bin_reach = self.bin
                    elif (each_sample_test[each_attribute] <= min_range):
                        bin_reach = 1
                    else:
                        bin_reach = math.ceil(reach_num_bin)
                    # bin_reach = math.ceil(reach_num_bin)#min value
                    # print('bin_reach',bin_reach)
                    sample_pro=self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute][bin_reach]
                    map_probability =map_probability*sample_pro
            max_posterior_prob[self.prior_classes[index]]=map_probability
        #     look for the max map and label it
        ex_max_pro=0;
        for max_index in range(len(self.prior_classes)):
            if(max_posterior_prob[self.prior_classes[max_index]]>ex_max_pro):
                each_label=self.prior_classes[max_index]
                ex_max_pro=max_posterior_prob[self.prior_classes[max_index]]
        return each_label


    def sample_result_foroutput(self,each_sample_test):
        max_posterior_prob={}
        # calculate the probability of each label given the testing data
        for index in range(len(self.prior_classes)):
            prior_label_pro = self.prior_classes[index]
            map_probability=1.0*self.prior_probability_for_testing[prior_label_pro]
            feature_probability=self.condition_likelihood_for_testing[self.prior_classes[index]]
            for each_attribute in range(len(each_sample_test)):
                if(self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute].__contains__('size')==False):
                    map_probability*=self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute][each_sample_test[each_attribute]]
                else:
                    size = self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['size']
                    max_range = self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['maxi']
                    min_range = self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute]['mini']
                    reach_num_bin=(each_sample_test[each_attribute]-min_range)/float(size)
                    if (each_sample_test[each_attribute] >= max_range):
                        bin_reach = self.bin
                    elif (each_sample_test[each_attribute] <= min_range):
                        bin_reach = 1
                    else:
                        bin_reach = math.ceil(reach_num_bin)
                    sample_pro = self.condition_likelihood_for_testing[self.prior_classes[index]][each_attribute][bin_reach]
                    map_probability = map_probability * sample_pro
            max_posterior_prob[self.prior_classes[index]]=map_probability
        return max_posterior_prob

    def output_five_sample(self,data_testing):
        result_label=[];
        result_num=1;
        for each_sample in data_testing[:]:
            result_label.append(self.sample_result_foroutput(each_sample))
            result_num+=1;
            if(result_num>5):
                return result_label
        return result_label

    def predict_whole_sample(self,data_testing):
        result_label=[];
        for each_sample in data_testing[:]:
            result_label.append(self.predict_sample_result(each_sample))
        return result_label

if __name__=='__main__':
    #import data
    training_data = pd.read_csv("training sample.csv", sep=",")
    trainingdataSetNP = np.array(training_data)  # 将数据由dataframe类型转换为数组类型
    training_data_attribute = trainingdataSetNP[:, 0:trainingdataSetNP.shape[1] - 1]  # 训练数据x1,x2
    training_data_label = trainingdataSetNP[:, trainingdataSetNP.shape[1] - 1]  # 训练数据所对应的所属类型Y
    test_data = pd.read_csv("testing sample.csv", sep=",")
    testingdataSetNP = np.array(test_data)  # 将数据由dataframe类型转换为数组类型
    testing_data_attribute = testingdataSetNP[:, 0:testingdataSetNP.shape[1]-1 ]  # 训练数据x1,x2
    testing_data_label = testingdataSetNP[:, testingdataSetNP.shape[1] - 1]  # 训练数据所对应的所属类型Y
    # training_data_attribute = training_data[training_data.columns.values[0:training_data.shape[1] - 1]]
    # training_data_label = training_data[training_data.columns.values[training_data.shape[1] - 1:training_data.shape[1]]]
    # print(training_data_attribute)
    # print(training_data_label)
    training_model=nativeBayesWithNormaltraining()
    training_model.lapsample=1;
    training_model.bin=9;
    training_model.calculationOfPriorPro(training_label=training_data_label)
    # for x in range(len(training_data_attribute[0])-5):
    #     print(x,':',np.unique(training_data_attribute[:,x]))
    training_model.calculationOfConditionalProbability(traning_attribute=training_data_attribute,training_label= training_data_label)
    # print(training_model.condition_likelihood)
    # testing training sample
    test_training_sample_model = nativeBayesWithNormaltesting()
    test_training_sample_model.bin = training_model.bin
    test_training_sample_model.prior_classes=training_model.prior_classes
    # print('training_model.prior_classes',training_model.prior_classes)
    test_training_sample_model.data_testing=training_data_attribute
    test_training_sample_model.condition_likelihood_for_testing=training_model.condition_likelihood
    test_training_sample_model.prior_probability_for_testing=training_model.prior_probability
    # print('training_model.prior_probability',training_model.prior_probability)
    final_result_label=test_training_sample_model.predict_whole_sample(data_testing=training_data_attribute)
    # print('final_result_label:',len(final_result_label))
    # print('training_data_label:',len(training_data_label))
    same_sample = sum(1 for a, b in zip(final_result_label, training_data_label) if (a == b))
    # print('same_sample:',same_sample)
    right_pre=float(same_sample)/float(len(final_result_label))
    print('The accuracy on training data is ',right_pre)
    test_testing_sample_model = nativeBayesWithNormaltesting()
    test_testing_sample_model.bin=training_model.bin
    test_testing_sample_model.prior_classes=training_model.prior_classes
    test_testing_sample_model.data_testing=testing_data_attribute
    test_testing_sample_model.condition_likelihood_for_testing=training_model.condition_likelihood
    test_testing_sample_model.prior_probability_for_testing=training_model.prior_probability
    final_result_label_test = test_testing_sample_model.predict_whole_sample(data_testing=testing_data_attribute)
    test_sample = sum(1 for a, b in zip(final_result_label_test, testing_data_label) if (a == b))
    # print('test_sample:', test_sample)
    test_pre = float(test_sample) / float(len(final_result_label_test))
    print('The accuracy on testing data is ',test_pre)
    testing_data_output = testingdataSetNP[0:5, 0:testingdataSetNP.shape[1] - 1]
    output_fivesample=test_testing_sample_model.output_five_sample(testing_data_attribute)
    for num in range(5):
        # print(testing_data_output[num])
        for key in output_fivesample[num].keys():
            print('P(',training_data.columns.values[len(training_data.columns.values)-1],'=',key,'|',testing_data_output[num],')=',output_fivesample[num][key])
    # print(output_fivesample)
    print('prior_probability: ',training_model.prior_probability)
    # print(len(training_data.columns.values));  #Check the column names
    for key in training_model.condition_likelihood.keys():# high low midian
        for num_attribute in training_model.condition_likelihood[key].keys():# attribute -----num=012345
            if(training_model.condition_likelihood[key][num_attribute].__contains__('size')==False):
                for each_label in training_model.condition_likelihood[key][num_attribute].keys():#label
                    print('P(',training_data.columns.values[num_attribute],'=',each_label,'|',training_data.columns.values[len(training_data.columns.values)-1],'=',key,')=',training_model.condition_likelihood[key][num_attribute][each_label])
            else:
                size=training_model.condition_likelihood[key][num_attribute]['size']
                for each in range(1,training_model.bin+1):
                    print('P(',training_data.columns.values[num_attribute],' from ', (each-1)*size+training_model.condition_likelihood[key][num_attribute]['mini'],' to ',each*size+training_model.condition_likelihood[key][num_attribute]['mini'],' | ',training_data.columns.values[len(training_data.columns.values)-1],'=',key,')=',training_model.condition_likelihood[key][num_attribute][each])
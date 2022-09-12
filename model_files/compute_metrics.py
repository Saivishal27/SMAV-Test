import pandas as pd
import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics

def getMetric(actual,prediction,method):
    if method == "accuracy_score":
        score = metrics.accuracy_score(actual, prediction)*100
        # if score<50:
        #     color='#eb3f38'
        # elif score>=50 and score<75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return score

    elif method == "precision_score":
        score = metrics.precision_score(actual, prediction, average='weighted')*100
        # if score<0.5:
        #     color='#eb3f38'
        # elif score>=0.5 and score<0.75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return score

    elif method == "recall_score":
        score = metrics.recall_score(actual, prediction, average='weighted')*100
        # if score<0.5:
        #     color='#eb3f38'
        # elif score>=0.5 and score<0.75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return score

    elif method == "f1_score":
        score = metrics.f1_score(actual, prediction, average='weighted')*100
        # if score<50:
        #     color='#eb3f38'
        # elif score>=50 and score<75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return score
    
    elif method == "r2_score":
        score = metrics.r2_score(list(actual), list(prediction),multioutput='raw_values')*100
        # if score<50:
        #     color='#eb3f38'
        # elif score>=50 and score<75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return list(score)

    elif method == "mean_absolute_error":
        score = metrics.mean_absolute_error(list(actual), list(prediction),multioutput='raw_values')
        # print score
        # if score<50:
        #     color='#eb3f38'
        # elif score>=50 and score<75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return list(score)

    elif method == "mean_squared_error":
        score = metrics.mean_squared_error(list(actual), list(prediction),multioutput='raw_values')
        # if score<50:
        #     color='#eb3f38'
        # elif score>=50 and score<75:
        #     color='#f8ba00'
        # else:
        #     color='#56c97d'
        # if score<thres:
        #     flag=1
        # else:
        #     flag=0
        return list(score)
    

def getFunctions(x,metrics,thres):
    response={}
    for i in range(len(metrics)):
        temp=getMetric(x['actual'],x['prediction'],metrics[i],thres[i])
        response[metrics[i]]=temp[0]
        response[metrics[i]+'_flag']=temp[3]
    return pd.Series(response)


def computeMetrics(metrics, data, window, responseRequired):
    
    classes = [0,1,2]
    # data = pd.DataFrame(data,columns=['prediction','actual','time_stamp'])
    data.set_index('time_stamp',inplace=True)
    data.index=pd.to_datetime(data.index, unit='ms')
    # print (data)
    # if eventSubType=='Classification':
    if responseRequired=='overall':
            # response = {"defaultMetric" : "accuracy"}
            response = {}
            for metric in metrics:
                calcMetric = getMetric(data['actual'],data['prediction'],metric)
                response[metric] = calcMetric

            return response

    elif responseRequired == 'window':
            print (data)
            # data['actual_cpy']=data['actual']
            computedMetrics = data.groupby([pd.TimeGrouper(window),'actual']).apply(lambda x : getFunctions(x,metrics,thres))
        
            computedMetrics['time_stamp'] = computedMetrics.index.get_level_values(0).astype('int')/1000000
            computedMetrics['time_stamp'] = computedMetrics['time_stamp'].astype('str')
            computedMetrics['actual'] = computedMetrics.index.get_level_values(1)

            print (computedMetrics)
            response={}
            for metric in metrics:
                response[metric]={}
                for clas in classes:
                    response[metric][clas] = computedMetrics[computedMetrics['actual']==clas][['time_stamp',metric,metric+'_flag']].values.tolist()
                response[metric]['overall'] = computedMetrics[['time_stamp',metric,metric+'_flag']].values.tolist()
            return response



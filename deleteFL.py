import os
import config_FL

M = config_FL.num_client()

os.remove('EVAL_metrics/FL/GlobalAccuracy'+str(M)+'.csv')

os.remove('EVAL_metrics/FL/global_model_size'+str(M)+'.csv')
os.remove('EVAL_metrics/FL/local_model_size'+str(M)+'.csv')
os.remove('EVAL_metrics/FL/runtime'+str(M)+'.csv')

for i in range(M):
    os.remove('EVAL_metrics/FL/local_Accuracy'+str(i)+str(M)+'.csv')

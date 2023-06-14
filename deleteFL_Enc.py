import os
import config_FL

M = config_FL.num_client()
   
os.remove('EVAL_metrics/FLEnc/runtime_Dealer'+str(M)+'.csv')
os.remove('EVAL_metrics/FLEnc/Params_size_Dealer'+str(M)+'.csv')
os.remove('EVAL_metrics/FLEnc/Global_model_dec'+str(M)+'.csv')
os.remove('EVAL_metrics/FLEnc/Global_model_Enc'+str(M)+'.csv')

os.remove('EVAL_metrics/FLEnc/GlobalAccuracy'+str(M)+'.csv')
os.remove('EVAL_metrics/FLEnc/runtime'+str(M)+'.csv')

for i in range(M):
    os.remove('EVAL_metrics/FLEnc/local_data_size_Dec'+str(i+1)+str(M)+'.csv')
    os.remove('EVAL_metrics/FLEnc/local_data_size'+str(i+1)+str(M)+'.csv')
    os.remove('EVAL_metrics/FLEnc/local_Accuracy'+str(i)+str(M)+'.csv')

#ifndef HDEVAI_H
#define HDEVAI_H

#include <halconcpp/HalconCpp.h>
#include <halconcpp/HDevThread.h>



using namespace HalconCpp;


//class HDevAI
//{
//public:
//    HDevAI();
//};


namespace HDevAI
{

//train涉及的接口
void train_dl_model (HTuple hv_DLDataset, HTuple hv_DLModelHandle, HTuple hv_TrainParam,\
                     HTuple hv_StartEpoch, HTuple *hv_TrainResults, HTuple *hv_TrainInfos, HTuple *hv_EvaluationInfos);

void tuple_shuffle (HTuple hv_Tuple, HTuple *hv_Shuffled);

void find_dl_samples (HTuple hv_Samples, HTuple hv_KeyName, HTuple hv_KeyValue, HTuple hv_Mode,
    HTuple *hv_SampleIndices);





//一些拓展api
}




#endif // HDEVAI_H

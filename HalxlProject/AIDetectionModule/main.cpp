#include "hdevai.h"

#include "mainwindow.h"
#include <QApplication>
#include <QDebug>
using namespace HalconCpp;

//2. Training of the model.
void action()
{

    // Local iconic variables

      // Local control variables
      HTuple  hv_DLDeviceHandles, hv_DLDevice, hv_DLDeviceType;
      HTuple  hv_NumThreadsTraining;
      HTuple  hv_ExampleDataDir, hv_InitialModelFileName;
      HTuple  hv_DataDirectory, hv_DLDatasetFileName, hv_BestModelBaseName;
      HTuple  hv_FinalModelBaseName, hv_BatchSize, hv_InitialLearningRate;
      HTuple  hv_Momentum, hv_NumEpochs, hv_EvaluationIntervalEpochs;
      HTuple  hv_ChangeLearningRateEpochs, hv_ChangeLearningRateValues;
      HTuple  hv_WeightPrior, hv_EnableDisplay, hv_RandomSeed;
      HTuple  hv_GenParamName, hv_GenParamValue, hv_AugmentationParam;
      HTuple  hv_ChangeStrategy, hv_SerializationStrategy, hv_SelectedPercentageTrainSamples;
      HTuple  hv_XAxisLabel, hv_DisplayParam, hv_DLModelHandle;
      HTuple  hv_DLDataset, hv_TrainParam, hv_TrainResults, hv_TrainInfos;
      HTuple  hv_EvaluationInfos;

      try
      {
          //GPU ѵ��
          QueryAvailableDlDevices((HTuple("runtime").Append("runtime")), (HTuple("gpu").Append("cpu")),
              &hv_DLDeviceHandles);
          if (0 != (int((hv_DLDeviceHandles.TupleLength())==0)))
          {
            throw HException("No supported device found to continue this example.");
          }
          //Due to the filter used in query_available_dl_devices, the first device is a GPU, if available.
          hv_DLDevice = ((const HTuple&)hv_DLDeviceHandles)[0];
          GetDlDeviceParam(hv_DLDevice, "type", &hv_DLDeviceType);
          if (0 != (int(hv_DLDeviceType==HTuple("cpu"))))
          {
            hv_NumThreadsTraining = 4;
            SetSystem("thread_num", hv_NumThreadsTraining);
          }
          qDebug()<<"00000000000000000000";

          //*****************************************************
          //***          1 ��������·��.            ***
          //*****************************************************
          // ���ݼ���Ŀ¼.
          hv_ExampleDataDir = "HalxAIDet/detect_pills_data";
          //Ԥѵ��ģ��·��.
          hv_InitialModelFileName = hv_ExampleDataDir+"/pretrained_dl_model_detection.hdl";
          //Ԥ����������·��.
          //Note: Adapt DataDirectory after preprocessing with another image size.
          hv_DataDirectory = hv_ExampleDataDir+"/dldataset_pill_bag_512x320";
          hv_DLDatasetFileName = hv_DataDirectory+"/dl_dataset.hdict";
          //
          //���1��best model·��.
          hv_BestModelBaseName = hv_ExampleDataDir+"/best_dl_model_detection";
          //���2��final trained model·��.
          hv_FinalModelBaseName = hv_ExampleDataDir+"/final_dl_model_detection";
          //
          //*****************************************************
          //***             2  ���������������ã�.            ***
          //*****************************************************
          qDebug()<<"11111111111111111111111";
          //The following parameters need to be adapted frequently.
          hv_BatchSize = 2;
          hv_InitialLearningRate = 0.0005;
          hv_Momentum = 0.99;
          hv_NumEpochs = 10;
          //�����������EpochΪ��λ�������ڼ�����֤�ָ��������ʩ��
          hv_EvaluationIntervalEpochs = 1;
          // ��������ʱ�ڵ�ѧϰ�ʣ����� [15, 30]��
          // �����Ӧ�øı�ѧϰ�ʣ���������Ϊ[]��
          hv_ChangeLearningRateEpochs = 30;
          // ��ѧϰ�ʸ���Ϊ����ֵ�����磬InitialLearningRate * [0.1, 0.01]��
          // Ԫ��ĳ��ȱ����� ChangeLearningRateEpochs �ĳ�����ͬ��
          hv_ChangeLearningRateValues = hv_InitialLearningRate*0.1;
          //
          //*****************************************************
          //***          3  �߼�������һ�㲻�Ķ���            ***
          //*****************************************************
          //The following parameters might need to be changed in rare cases.
          //
          //Model parameter.
          //��ʼȨ��
          hv_WeightPrior = 0.00001;
          //
          //* ѵ�����̿��ӻ� Ĭ��false
          hv_EnableDisplay = 0;
          hv_RandomSeed = 42;
          SetSystem("seed_rand", hv_RandomSeed);
          //
          // Ϊ����ͬһGPU�ϻ�ý���ȷ���Ե�ѵ�����
          //��ϵͳ����������cuda �汾�������Խ���cudnn_deterministic��ָ��Ϊ
          // 'true'�� ��ע�⣬����ܻ���΢����ѵ���ٶȡ�
          //set_system ('cudnn_deterministic', 'true')

          hv_GenParamName = HTuple();
          hv_GenParamValue = HTuple();
          //
          // Augmentation parameter/��������.
          CreateDict(&hv_AugmentationParam);
          SetDictTuple(hv_AugmentationParam, "augmentation_percentage", 50);
          //row / column.���о�������.
          SetDictTuple(hv_AugmentationParam, "mirror", "rc");
          hv_GenParamName = hv_GenParamName.TupleConcat("augment");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_AugmentationParam);

          //ѧϰ��˥����Change strategies.
          if (0 != (int((hv_ChangeLearningRateEpochs.TupleLength())>0)))
          {
            CreateDict(&hv_ChangeStrategy);
            SetDictTuple(hv_ChangeStrategy, "model_param", "learning_rate");
            SetDictTuple(hv_ChangeStrategy, "initial_value", hv_InitialLearningRate);
            //�ڵ�ChangeLearningRateEpochs��epochʱ���ı�ѧϰ��
            SetDictTuple(hv_ChangeStrategy, "epochs", hv_ChangeLearningRateEpochs);
            SetDictTuple(hv_ChangeStrategy, "values", hv_ChangeLearningRateValues);
            hv_GenParamName = hv_GenParamName.TupleConcat("change");
            hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_ChangeStrategy);
          }
          //
          //* ���л�������ԡ�
          //* �ж��ѡ������ڽ��м�ģ�ͱ��浽���̣������ create_dl_train_param����
          //*  BestModel��FinalModel�����浽�������õ�·���С�
          CreateDict(&hv_SerializationStrategy);
          SetDictTuple(hv_SerializationStrategy, "type", "best");
          SetDictTuple(hv_SerializationStrategy, "basename", hv_BestModelBaseName);
          hv_GenParamName = hv_GenParamName.TupleConcat("serialize");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_SerializationStrategy);
          CreateDict(&hv_SerializationStrategy);
          SetDictTuple(hv_SerializationStrategy, "type", "final");
          SetDictTuple(hv_SerializationStrategy, "basename", hv_FinalModelBaseName);
          hv_GenParamName = hv_GenParamName.TupleConcat("serialize");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_SerializationStrategy);

          //* ��ʾ������
          //* �ڱ����У�ѵ�����ε�������ʩ��ѵ�������в���ʾ
          //* ѵ����Ĭ�ϣ��� �����������������ѡ��һ��������ѵ��
          //* ������ѵ���ڼ�����ģ�͵������� �ϵ͵İٷֱ������ڼӿ������ٶȡ�
          //* ���ѵ�������������ʩӦ����ʾ���뽫 SelectedPercentageTrainSamples ����Ϊ 0��
          hv_SelectedPercentageTrainSamples = 0;
          hv_XAxisLabel = "epochs";
          CreateDict(&hv_DisplayParam);
          SetDictTuple(hv_DisplayParam, "selected_percentage_train_samples", hv_SelectedPercentageTrainSamples);
          SetDictTuple(hv_DisplayParam, "x_axis_label", hv_XAxisLabel);
          hv_GenParamName = hv_GenParamName.TupleConcat("display");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_DisplayParam);
          //
          //*****************************************************
          //***       4 ��ȡģ�ͺ����ݼ�.         ***
          //*****************************************************
          qDebug()<<"check_data_availability";
          HDevAI::check_data_availability(hv_ExampleDataDir, hv_InitialModelFileName, hv_DLDatasetFileName);
    //      HTuple hv_FileExists;
    //      FileExists(hv_ExampleDataDir, &hv_FileExists);
    //      if (0 != (hv_FileExists.TupleNot()))
    //      {
    //        throw HException(hv_ExampleDataDir+" does not exist. Please run part 1 of the example series.");
    //      }
    //      //
    //      FileExists(hv_InitialModelFileName, &hv_FileExists);
    //      if (0 != (hv_FileExists.TupleNot()))
    //      {
    //        throw HException(hv_InitialModelFileName+" does not exist. Please run part 1 of the example series.");
    //      }
    //      //
    //      FileExists(hv_DLDatasetFileName, &hv_FileExists);
    //      if (0 != (hv_FileExists.TupleNot()))
    //      {
    //        throw HException(hv_DLDatasetFileName+" does not exist. Please run part 1 of the example series.");
    //      }

          ReadDlModel(hv_InitialModelFileName, &hv_DLModelHandle);
          ReadDict(hv_DLDatasetFileName, HTuple(), HTuple(), &hv_DLDataset);
          //
          //*****************************************************
          //***              5 ����ģ�Ͳ���.             ***
          //*****************************************************
          qDebug()<<"SetDlModelParam";
          //������
          SetDlModelParam(hv_DLModelHandle, "learning_rate", hv_InitialLearningRate);
          SetDlModelParam(hv_DLModelHandle, "momentum", hv_Momentum);
          SetDlModelParam(hv_DLModelHandle, "batch_size", hv_BatchSize);
          if (0 != (int((hv_WeightPrior.TupleLength())>0)))
          {
            SetDlModelParam(hv_DLModelHandle, "weight_prior", hv_WeightPrior);
          }
          //When the batch size is determined, set the device.
          SetDlModelParam(hv_DLModelHandle, "device", hv_DLDevice);
          //
          //
          //*****************************************************
          //***               6  ѵ��/Train the model.                ***
          //*****************************************************
          qDebug()<<"create_dl_train_param";
          HDevAI::create_dl_train_param(hv_DLModelHandle, hv_NumEpochs, hv_EvaluationIntervalEpochs,
              hv_EnableDisplay, hv_RandomSeed, hv_GenParamName, hv_GenParamValue, &hv_TrainParam);
          //
          qDebug()<<"train_dl_model";
          //train_dl_model_batch () within the following procedure.
          HDevAI::train_dl_model(hv_DLDataset, hv_DLModelHandle, hv_TrainParam, 0.0, &hv_TrainResults,
              &hv_TrainInfos, &hv_EvaluationInfos);
          qDebug()<<"train over!";
    //      if (HDevWindowStack::IsOpen())
    //        DispText(HDevWindowStack::GetActive(),"Press Run (F5) to continue", "window",
    //            "bottom", "right", "black", HTuple(), HTuple());
    //      //
    //      //Close training windows.
    //      if (HDevWindowStack::IsOpen())
    //        CloseWindow(HDevWindowStack::Pop());
    //      if (HDevWindowStack::IsOpen())
    //        CloseWindow(HDevWindowStack::Pop());
    //      //
    //      //Show the final example screen.
    //      if (0 != hv_ShowExampleScreens)
    //      {
    //        //Hint at the DL detection evaluation and inference example.
    //        HDevAI::dev_display_screen_final(hv_ExampleInternals);
    //        // stop(...); only in hdevelop
    //        //Close example windows.
    //        HDevAI::dev_close_example_windows(hv_ExampleInternals);
    //      }

      }
      catch(HException error)
      {
          qDebug()<<"Error: "<<error.ErrorMessage().ToUtf8();
      }

}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //MainWindow w;
    //w.show();
    //return a.exec();
    action();
    return 0;

}

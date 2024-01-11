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
          //GPU 训练
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
          //***          1 输入和输出路径.            ***
          //*****************************************************
          // 数据集根目录.
          hv_ExampleDataDir = "HalxAIDet/detect_pills_data";
          //预训练模型路径.
          hv_InitialModelFileName = hv_ExampleDataDir+"/pretrained_dl_model_detection.hdl";
          //预处理后的数据路径.
          //Note: Adapt DataDirectory after preprocessing with another image size.
          hv_DataDirectory = hv_ExampleDataDir+"/dldataset_pill_bag_512x320";
          hv_DLDatasetFileName = hv_DataDirectory+"/dl_dataset.hdict";
          //
          //输出1：best model路径.
          hv_BestModelBaseName = hv_ExampleDataDir+"/best_dl_model_detection";
          //输出2：final trained model路径.
          hv_FinalModelBaseName = hv_ExampleDataDir+"/final_dl_model_detection";
          //
          //*****************************************************
          //***             2  超参数（自行设置）.            ***
          //*****************************************************
          qDebug()<<"11111111111111111111111";
          //The following parameters need to be adapted frequently.
          hv_BatchSize = 2;
          hv_InitialLearningRate = 0.0005;
          hv_Momentum = 0.99;
          hv_NumEpochs = 10;
          //评估间隔（以Epoch为单位），用于计算验证分割的评估措施。
          hv_EvaluationIntervalEpochs = 1;
          // 更改以下时期的学习率，例如 [15, 30]。
          // 如果不应该改变学习率，则将其设置为[]。
          hv_ChangeLearningRateEpochs = 30;
          // 将学习率更改为以下值，例如，InitialLearningRate * [0.1, 0.01]。
          // 元组的长度必须与 ChangeLearningRateEpochs 的长度相同。
          hv_ChangeLearningRateValues = hv_InitialLearningRate*0.1;
          //
          //*****************************************************
          //***          3  高级参数（一般不改动）            ***
          //*****************************************************
          //The following parameters might need to be changed in rare cases.
          //
          //Model parameter.
          //初始权重
          hv_WeightPrior = 0.00001;
          //
          //* 训练过程可视化 默认false
          hv_EnableDisplay = 0;
          hv_RandomSeed = 42;
          SetSystem("seed_rand", hv_RandomSeed);
          //
          // 为了在同一GPU上获得近乎确定性的训练结果
          //（系统、驱动程序、cuda 版本）您可以将“cudnn_deterministic”指定为
          // 'true'。 请注意，这可能会稍微减慢训练速度。
          //set_system ('cudnn_deterministic', 'true')

          hv_GenParamName = HTuple();
          hv_GenParamValue = HTuple();
          //
          // Augmentation parameter/增广设置.
          CreateDict(&hv_AugmentationParam);
          SetDictTuple(hv_AugmentationParam, "augmentation_percentage", 50);
          //row / column.行列镜像增广.
          SetDictTuple(hv_AugmentationParam, "mirror", "rc");
          hv_GenParamName = hv_GenParamName.TupleConcat("augment");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_AugmentationParam);

          //学习率衰减：Change strategies.
          if (0 != (int((hv_ChangeLearningRateEpochs.TupleLength())>0)))
          {
            CreateDict(&hv_ChangeStrategy);
            SetDictTuple(hv_ChangeStrategy, "model_param", "learning_rate");
            SetDictTuple(hv_ChangeStrategy, "initial_value", hv_InitialLearningRate);
            //在第ChangeLearningRateEpochs个epoch时，改变学习率
            SetDictTuple(hv_ChangeStrategy, "epochs", hv_ChangeLearningRateEpochs);
            SetDictTuple(hv_ChangeStrategy, "values", hv_ChangeLearningRateValues);
            hv_GenParamName = hv_GenParamName.TupleConcat("change");
            hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_ChangeStrategy);
          }
          //
          //* 序列化保存策略。
          //* 有多个选项可用于将中间模型保存到磁盘（请参阅 create_dl_train_param）。
          //*  BestModel和FinalModel被保存到上面设置的路径中。
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

          //* 显示参数。
          //* 在本例中，训练批次的评估措施在训练过程中不显示
          //* 训练（默认）。 如果你想这样做，请选择一定比例的训练
          //* 用于在训练期间评估模型的样本。 较低的百分比有助于加快评估速度。
          //* 如果训练分组的评估措施应不显示，请将 SelectedPercentageTrainSamples 设置为 0。
          hv_SelectedPercentageTrainSamples = 0;
          hv_XAxisLabel = "epochs";
          CreateDict(&hv_DisplayParam);
          SetDictTuple(hv_DisplayParam, "selected_percentage_train_samples", hv_SelectedPercentageTrainSamples);
          SetDictTuple(hv_DisplayParam, "x_axis_label", hv_XAxisLabel);
          hv_GenParamName = hv_GenParamName.TupleConcat("display");
          hv_GenParamValue = hv_GenParamValue.TupleConcat(hv_DisplayParam);
          //
          //*****************************************************
          //***       4 读取模型和数据集.         ***
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
          //***              5 设置模型参数.             ***
          //*****************************************************
          qDebug()<<"SetDlModelParam";
          //超参数
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
          //***               6  训练/Train the model.                ***
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

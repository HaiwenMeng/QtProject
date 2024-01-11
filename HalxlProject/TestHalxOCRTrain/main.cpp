#include "halxAI.h"



#include <QCoreApplication>

void run()
{

}

void action()
{

    // Local iconic variables
    HObject  ho_Image;

    // Local control variables
    HTuple  hv_DatasetFilename, hv_BaseNameDataset;
    HTuple  hv__, hv_OutputDir, hv_FolderExists, hv_RemoveOutputs;
    HTuple  hv_SkipTraining, hv_ImageWidth, hv_ImageHeight;
    HTuple  hv_MaxBatchSize, hv_NumEpochs, hv_UseAugmentation;
    HTuple  hv_AugmentationRate, hv_BrightnessVariation, hv_BrightnessVariationSpot;
    HTuple  hv_ContrastVariation, hv_SaturationVariation, hv_RotateRange;
    HTuple  hv_DeviceRuntime, hv_SplitPercentageTrain, hv_SplitPercentageValidation;
    HTuple  hv_OverwritePreprocessing, hv_OutputDirPreprocessing;
    HTuple  hv_BestModelFilename, hv_BaselineModelFilename;
    HTuple  hv_DLModelHandle, hv_DLDeviceHandles, hv_DLDevice;
    HTuple  hv_DLDataset, hv_InvalidSamplesIndices, hv_InvalidSamplesReasons;
    HTuple  hv_GenParamPreprocessing, hv_DLPreprocessParam;
    HTuple  hv_WindowDict, hv_Index, hv_SampleIndex, hv_DLSample;
    HTuple  hv_DLDatasetFileName, hv_DeviceType, hv_SerializationStrategy;
    HTuple  hv_BaseName, hv_Extension, hv_Directory, hv_AugmentationParam;
    HTuple  hv_DisplayParam, hv_TrainParam, hv_TrainResults;
    HTuple  hv_TrainInfos, hv_EvaluationInfos, hv_BatchSizeEval;
    HTuple  hv_EvaluationSplits, hv_EvaluationSplitsMaxNumSamples;
    HTuple  hv_DLModelHandleBaseline, hv_EvalDict, hv_WindowHandles;
    HTuple  hv_WindowPos, hv_I, hv_EvaluationSplit, hv_EvalResults;
    HTuple  hv_MaxNumSamples, hv_WindowHandleEval, hv_WidthEvalWindow;
    HTuple  hv_VisualizationDict, hv_SampleIndices, hv_NumSamplesComparison;
    HTuple  hv_IndexInference, hv_DLSampleInference, hv_DLResult;
    HTuple  hv_DeepOcrHandle, hv_Alphabet, hv_BestModelDeepOCRFilename;
    HTuple  hv_WindowHandle, hv_Sample, hv_ImageFile, hv_DeepOcrResult;
    HTuple  hv_xxx, hv___Tmp_Ctrl_Dict_Init_3, hv___Tmp_Ctrl_0;

    //
    //***   0) SET PARAMETERS  ***
    //
    //Here, we specify a compatible DLDataset.
    //The compatible dataset format is described in the
    //HALCON Operator Reference (OCR -> Deep OCR).
    hv_DatasetFilename = "punched_numbers.hdict";
    parse_filename(hv_DatasetFilename, &hv_BaseNameDataset, &hv__, &hv__);
    //
    //The folder OutputDir will contain all output data.
    //(e.g., preprocessed dataset and the trained model files).
    hv_OutputDir = "ocr_detection_data_"+hv_BaseNameDataset;
    FileExists(hv_OutputDir, &hv_FolderExists);
    if (0 != (hv_FolderExists.TupleNot()))
    {
        MakeDir(hv_OutputDir);
    }
    //
    //Set to true, if the results should be deleted after running
    //this program.
    hv_RemoveOutputs = 1;
    //
    //Optionally skip training.
    hv_SkipTraining = 0;
    //
    //Specify the model input image dimension.
    //Note, the parameter values have a big impact on results,
    //runtime and memory usage. The optimal setting is the smallest
    //dimension with highest F-score evaluation.
    hv_ImageWidth = 512;
    hv_ImageHeight = 384;
    //
    //If possible, the training uses this batch size.
    //Note, if batch size is increased it is recommended to
    //increase the NumEpochs by the same ratio and therefore
    //the number of training iterations.
    hv_MaxBatchSize = 2;
    //
    //Specify the number of training epochs.
    //Note, we choose a fairly low number of epochs here.
    //Please observe the training error and increase this value for
    //real world applications.
    hv_NumEpochs = 1;
    //
    //Specify if the training should use data augmentation.
    hv_UseAugmentation = 1;
    //
    //Specify the rate (percentage) of samples which should
    //undergo augmentation.
    hv_AugmentationRate = 75;
    //
    //Specify the augmentation parameters.
    hv_BrightnessVariation = 20;
    hv_BrightnessVariationSpot = 20;
    hv_ContrastVariation = 0.2;
    hv_SaturationVariation = 0.2;
    hv_RotateRange = 15;
    //
    //Specify the deep learning device runtime to work with.
    //Training supports CPU and GPU.
    hv_DeviceRuntime.Clear();
    hv_DeviceRuntime[0] = "gpu";
    hv_DeviceRuntime[1] = "cpu";
    //
    //The dataset split is done randomly.
    //These parameters determine the split percentages.
    hv_SplitPercentageTrain = 80;
    hv_SplitPercentageValidation = 10;
    //
    //Specify if the preprocessed data should be overwritten.
    //'auto':  It will be determined if the data has to be recomputed
    //'true':  Always preprocess the data
    //'false': Do not preprocess the data if it exists (raise error
    //         instead)
    hv_OverwritePreprocessing = "auto";
    //
    //***   1.) PREPARE   ***
    SetSystem("seed_rand", 42);
    //
    //Prepare output paths.
    hv_OutputDirPreprocessing = hv_OutputDir+"/dlsample_files";
    hv_BestModelFilename = ((hv_OutputDir+"/model_")+hv_BaseNameDataset)+"_best.hdl";
    //
    //Read the pretrained OCR detection model.
    hv_BaselineModelFilename = "pretrained_deep_ocr_detection.hdl";
    ReadDlModel(hv_BaselineModelFilename, &hv_DLModelHandle);
    //
    //Adjust the image dimensions.
    SetDlModelParam(hv_DLModelHandle, "image_size", hv_ImageWidth.TupleConcat(hv_ImageHeight));
    //
    //Determine deep learning device to work with.
    QueryAvailableDlDevices(HTuple(hv_DeviceRuntime.TupleLength(),"runtime"), hv_DeviceRuntime,
                            &hv_DLDeviceHandles);
    if (0 != (int((hv_DLDeviceHandles.TupleLength())==0)))
    {
        throw HException("No supported device found to continue this example.");
    }
    //Choose the first device.
    hv_DLDevice = ((const HTuple&)hv_DLDeviceHandles)[0];
    SetDlModelParam(hv_DLModelHandle, "device", hv_DLDevice);
    //
    //Read an OCR detection dataset.
    read_dl_dataset_ocr_detection(hv_DatasetFilename, HTuple(), HTuple(), &hv_DLDataset,
                                  &hv_InvalidSamplesIndices, &hv_InvalidSamplesReasons);
    //
    //Visualize invalid samples (if any).
    dev_display_dl_invalid_samples(hv_DLDataset, hv_DLModelHandle, hv_InvalidSamplesIndices,
                                   hv_InvalidSamplesReasons);
    //
    //Apply a random split to the dataset.
    split_dl_dataset(hv_DLDataset, hv_SplitPercentageTrain, hv_SplitPercentageValidation,
                     HTuple());
    //
    //Create preprocessing parameters based on the detection model.
    if (0 != hv_UseAugmentation)
    {
        CreateDict(&hv_GenParamPreprocessing);
        SetDictTuple(hv_GenParamPreprocessing, "augmentation", "true");
    }
    else
    {
        CreateDict(&hv_GenParamPreprocessing);
        SetDictTuple(hv_GenParamPreprocessing, "augmentation", "false");
    }
    create_dl_preprocess_param_from_model(hv_DLModelHandle, "none", "full_domain",
                                          HTuple(), HTuple(), hv_GenParamPreprocessing, &hv_DLPreprocessParam);
    //
    //Inspect 5 randomly selected dataset samples visually.
    CreateDict(&hv_WindowDict);
    for (hv_Index=0; hv_Index<=4; hv_Index+=1)
    {
        hv_SampleIndex = HTuple(HTuple::TupleRand(1)*(((hv_DLDataset.TupleGetDictTuple("samples")).TupleLength())-1)).TupleRound();
        gen_dl_samples(hv_DLDataset, hv_SampleIndex, "ocr_detection", HTuple(), &hv_DLSample);
        preprocess_dl_samples(hv_DLSample, hv_DLPreprocessParam);
        dev_display_dl_data(hv_DLSample, HTuple(), hv_DLDataset, "ocr_detection_ground_truth",
                            HTuple(), hv_WindowDict);
        if (HDevWindowStack::IsOpen())
            DispText(HDevWindowStack::GetActive(),"Press F5 to continue", "window", "bottom",
                     "right", "black", HTuple(), HTuple());
        // stop(...); only in hdevelop
    }
    dev_close_window_dict(hv_WindowDict);
    //
    //Preprocess the data in DLDataset.
    //
    //Existing preprocessed data is only overwritten if
    //preprocessing parameters or the dataset have changed.
    CreateDict(&hv___Tmp_Ctrl_Dict_Init_3);
    SetDictTuple(hv___Tmp_Ctrl_Dict_Init_3, "overwrite_files", hv_OverwritePreprocessing);
    preprocess_dl_dataset(hv_DLDataset, hv_OutputDirPreprocessing, hv_DLPreprocessParam,
                          hv___Tmp_Ctrl_Dict_Init_3, &hv_DLDatasetFileName);
    hv___Tmp_Ctrl_Dict_Init_3 = HTuple::TupleConstant("HNULL");
    //
    //Adapt the batch size to the given device.
    GetDlDeviceParam(hv_DLDevice, "type", &hv_DeviceType);
    if (0 != (int(hv_DeviceType==HTuple("gpu"))))
    {
        set_dl_model_param_max_gpu_batch_size(hv_DLModelHandle, hv_MaxBatchSize);
        GetDlModelParam(hv_DLModelHandle, "batch_size", &hv_MaxBatchSize);
    }
    else
    {
        SetDlModelParam(hv_DLModelHandle, "batch_size", hv_MaxBatchSize);
    }
    //
    if (0 != (hv_SkipTraining.TupleNot()))
    {
        //***   2.) TRAIN   ***
        //
        //For information on the parameters, see the documentation
        //of set_dl_model_param () and get_dl_model_param ().
        SetDlModelParam(hv_DLModelHandle, "learning_rate", 0.0001);
        //
        //Specify a serialization strategy.
        CreateDict(&hv_SerializationStrategy);
        SetDictTuple(hv_SerializationStrategy, "type", "best");
        parse_filename(hv_BestModelFilename, &hv_BaseName, &hv_Extension, &hv_Directory);
        SetDictTuple(hv_SerializationStrategy, "basename", hv_Directory+hv_BaseName);
        //
        //Prepare the augmentation parameters.
        if (0 != hv_UseAugmentation)
        {
            CreateDict(&hv_AugmentationParam);
            SetDictTuple(hv_AugmentationParam, "augmentation_percentage", hv_AugmentationRate);
            SetDictTuple(hv_AugmentationParam, "brightness_variation", hv_BrightnessVariation);
            SetDictTuple(hv_AugmentationParam, "brightness_variation_spot", hv_BrightnessVariationSpot);
            SetDictTuple(hv_AugmentationParam, "contrast_variation", hv_ContrastVariation);
            SetDictTuple(hv_AugmentationParam, "saturation_variation", hv_SaturationVariation);
            SetDictTuple(hv_AugmentationParam, "rotate_range", hv_RotateRange);
        }
        else
        {
            CreateDict(&hv_AugmentationParam);
            SetDictTuple(hv_AugmentationParam, "augmentation_percentage", 0);
        }
        //
        //Define the percentage of the training data that is used
        //during F-score visualization.
        CreateDict(&hv_DisplayParam);
        SetDictTuple(hv_DisplayParam, "selected_percentage_train_samples", 10);
        //
        //Prepare the training parameters.
        create_dl_train_param(hv_DLModelHandle, hv_NumEpochs, 1, "true", 42, (((HTuple("serialize").Append("augment")).Append("display")).Append("evaluate_before_train")),
                              ((hv_SerializationStrategy.TupleConcat(hv_AugmentationParam)).TupleConcat(hv_DisplayParam)).TupleConcat("true"),
                              &hv_TrainParam);
        //
        //Training can start/continue now.
        train_dl_model(hv_DLDataset, hv_DLModelHandle, hv_TrainParam, 0, &hv_TrainResults,
                       &hv_TrainInfos, &hv_EvaluationInfos);
        if (HDevWindowStack::IsOpen())
            DispText(HDevWindowStack::GetActive(),"Press F5 to continue or run train_dl_model again",
                     "window", "bottom", "left", "black", HTuple(), HTuple());
        // stop(...); only in hdevelop
        //We clear the training model because we want to read the
        //best model that was saved during training.
        ClearDlModel(hv_DLModelHandle);
        if (HDevWindowStack::IsOpen())
            CloseWindow(HDevWindowStack::Pop());
        if (HDevWindowStack::IsOpen())
            CloseWindow(HDevWindowStack::Pop());
    }
    //***   3.) EVALUATE   ***
    //
    //Use the highest possible batch size for evaluation.
    hv_BatchSizeEval = hv_MaxBatchSize;
    //
    //We evaluate the test split as well as a subset of the train
    //split. If the evaluation metrics of the train split are not
    //satisfactory there is a problem either with your training
    //data or with the training/augmentation parameters.
    //The training data could be too heterogeneous or you might not
    //have trained long enough.
    //By comparing the evaluation metrics of the validation and
    //test set, you can see how well the model generalizes to new data.
    //Note, the validation set has been used during training to find
    //the optimal model.
    //
    //Select the splits to be used in the evaluation.
    //You could add the 'validation' split also for
    //additional insights.
    hv_EvaluationSplits.Clear();
    hv_EvaluationSplits[0] = "test";
    hv_EvaluationSplits[1] = "train";
    //
    //To reduce evaluation time, you may choose the maximum number of
    //samples to be evaluated. If -1 is specified the full split
    //is evaluated.
    hv_EvaluationSplitsMaxNumSamples.Clear();
    hv_EvaluationSplitsMaxNumSamples[0] = -1;
    hv_EvaluationSplitsMaxNumSamples[1] = 100;
    //
    //Evaluate the baseline model first.
    ReadDlModel(hv_BaselineModelFilename, &hv_DLModelHandleBaseline);
    ReadDlModel(hv_BestModelFilename, &hv_DLModelHandle);
    //
    //Store evaluation results.
    CreateDict(&hv_EvalDict);
    hv_WindowHandles = HTuple();
    hv_WindowPos = 0;
    //Compute and visualize evaluation results on selected splits.
    {
        HTuple end_val237 = (hv_EvaluationSplits.TupleLength())-1;
        HTuple step_val237 = 1;
        for (hv_I=0; hv_I.Continue(end_val237, step_val237); hv_I += step_val237)
        {
            hv_EvaluationSplit = HTuple(hv_EvaluationSplits[hv_I]);
            CreateDict(&hv_EvalResults);
            SetDictTuple(hv_EvalDict, HTuple(hv_EvaluationSplits[hv_I]), hv_EvalResults);
            if (0 != (int((hv_EvaluationSplitsMaxNumSamples.TupleLength())>1)))
            {
                hv_MaxNumSamples = HTuple(hv_EvaluationSplitsMaxNumSamples[hv_I]);
            }
            else
            {
                hv_MaxNumSamples = hv_EvaluationSplitsMaxNumSamples;
            }
            //
            //Evaluate the pretrained model.
            evaluate_ocr_detection(hv_DLModelHandleBaseline, hv_DLDevice, hv_ImageWidth,
                                   hv_ImageHeight, hv_BatchSizeEval, hv_DLDataset, hv_EvaluationSplit, hv_MaxNumSamples,
                                   &hv___Tmp_Ctrl_0);
            SetDictTuple(hv_EvalResults, "pretrained", hv___Tmp_Ctrl_0);
            //
            //Evaluate the finetuned model.
            evaluate_ocr_detection(hv_DLModelHandle, hv_DLDevice, hv_ImageWidth, hv_ImageHeight,
                                   hv_BatchSizeEval, hv_DLDataset, hv_EvaluationSplit, hv_MaxNumSamples, &hv___Tmp_Ctrl_0);
            SetDictTuple(hv_EvalResults, "finetuned", hv___Tmp_Ctrl_0);
            //
            hv_WindowHandleEval = HTuple();
            dev_display_evaluation_comparison((HTuple("Deep OCR pretrained").Append("Deep OCR finetuned")),
                                              (hv_EvalResults.TupleGetDictTuple("pretrained")).TupleConcat(hv_EvalResults.TupleGetDictTuple("finetuned")),
                                              hv_WindowPos, &hv_WindowHandleEval);
            hv_WindowHandles = hv_WindowHandles.TupleConcat(hv_WindowHandleEval);
            GetWindowExtents(hv_WindowHandleEval, &hv__, &hv__, &hv_WidthEvalWindow, &hv__);
            hv_WindowPos = (hv_WindowPos+hv_WidthEvalWindow)+15;
        }
    }
    //
    // stop(...); only in hdevelop
    {
        HTuple end_val263 = (hv_WindowHandles.TupleLength())-1;
        HTuple step_val263 = 1;
        for (hv_I=0; hv_I.Continue(end_val263, step_val263); hv_I += step_val263)
        {
            HDevWindowStack::SetActive(HTuple(hv_WindowHandles[hv_I]));
            if (HDevWindowStack::IsOpen())
                CloseWindow(HDevWindowStack::Pop());
        }
    }
    //
    //***   4.) COMPARE FINETUNED AND BASELINE MODEL   ***
    //
    //Run sample-by-sample comparison with visual score map analysis.
    SetDlModelParam(hv_DLModelHandle, "batch_size", 1);
    SetDlModelParam(hv_DLModelHandleBaseline, "batch_size", 1);
    //
    prepare_detailed_comparison_visualization(hv_ImageWidth, hv_ImageHeight, &hv_VisualizationDict);
    //
    find_dl_samples(hv_DLDataset.TupleGetDictTuple("samples"), "split", "test", "match",
                    &hv_SampleIndices);
    //
    SetDlModelParam(hv_DLModelHandleBaseline, "batch_size", 1);
    SetDlModelParam(hv_DLModelHandle, "batch_size", 1);
    hv_NumSamplesComparison = 10;
    tuple_shuffle(hv_SampleIndices, &hv_SampleIndices);
    {
        HTuple end_val282 = HTuple((hv_SampleIndices.TupleLength())-1).TupleMin2(hv_NumSamplesComparison-1);
        HTuple step_val282 = 1;
        for (hv_IndexInference=0; hv_IndexInference.Continue(end_val282, step_val282); hv_IndexInference += step_val282)
        {
            read_dl_samples(hv_DLDataset, HTuple(hv_SampleIndices[hv_IndexInference]), &hv_DLSampleInference);
            ApplyDlModel(hv_DLModelHandleBaseline, hv_DLSampleInference, HTuple(), &hv_DLResult);
            dev_display_dl_data(hv_DLSampleInference, hv_DLResult, hv_DLDataset, hv_VisualizationDict.TupleGetDictTuple("keys_for_display"),
                                hv_VisualizationDict.TupleGetDictTuple("display_param"), hv_VisualizationDict.TupleGetDictTuple("window_dict_top"));
            HDevWindowStack::SetActive((hv_VisualizationDict.TupleGetDictTuple("window_dict_top")).TupleGetDictTuple("ocr_detection_both"));
            if (HDevWindowStack::IsOpen())
                DispText(HDevWindowStack::GetActive(),"pretrained", "window", "top", "right",
                         "black", HTuple(), HTuple());

            ApplyDlModel(hv_DLModelHandle, hv_DLSampleInference, HTuple(), &hv_DLResult);
            dev_display_dl_data(hv_DLSampleInference, hv_DLResult, hv_DLDataset, hv_VisualizationDict.TupleGetDictTuple("keys_for_display"),
                                hv_VisualizationDict.TupleGetDictTuple("display_param"), hv_VisualizationDict.TupleGetDictTuple("window_dict_bottom"));
            HDevWindowStack::SetActive((hv_VisualizationDict.TupleGetDictTuple("window_dict_bottom")).TupleGetDictTuple("ocr_detection_both"));
            if (HDevWindowStack::IsOpen())
                DispText(HDevWindowStack::GetActive(),"finetuned", "window", "top", "right",
                         "black", HTuple(), HTuple());
            // stop(...); only in hdevelop
        }
    }
    dev_close_window_dict((hv_VisualizationDict.TupleGetDictTuple("window_dict_top")).TupleConcat(hv_VisualizationDict.TupleGetDictTuple("window_dict_bottom")));
    // stop(...); only in hdevelop
    //
    //
    //***   5.) DEEP OCR INTEGRATION AND INFERENCE   ***
    //
    //Inference is based on using the Deep OCR operator set.
    CreateDeepOcr(HTuple(), HTuple(), &hv_DeepOcrHandle);
    //
    //In case of standard example dataset we can reduce the
    //alphabet to 0-9.
    if (0 != (int(hv_BaseNameDataset==HTuple("punched_numbers"))))
    {
        hv_Alphabet = HTuple::TupleGenSequence(0,9,1)+"";
        SetDeepOcrParam(hv_DeepOcrHandle, "recognition_alphabet", hv_Alphabet);
    }
    //
    //The finetuned model needs to be specified as the detection
    //component.
    SetDeepOcrParam(hv_DeepOcrHandle, "detection_model", hv_BestModelFilename);
    //
    //Additionally, we ensure that the runtime related parameters
    //are set optimally for inference.
    SetDeepOcrParam(hv_DeepOcrHandle, "detection_batch_size", 1);
    SetDeepOcrParam(hv_DeepOcrHandle, "detection_optimize_for_inference", "true");
    SetDeepOcrParam(hv_DeepOcrHandle, "device", hv_DLDevice);
    //
    //For convenience we write the Deep OCR model to the output
    //directory. That way it can be easily read via read_deep_ocr.
    hv_BestModelDeepOCRFilename = ((hv_OutputDir+"/model_")+hv_BaseNameDataset)+"_best.hdo";
    WriteDeepOcr(hv_DeepOcrHandle, hv_BestModelDeepOCRFilename);

    find_dl_samples(hv_DLDataset.TupleGetDictTuple("samples"), "split", "test", "match",
                    &hv_SampleIndices);
    tuple_shuffle(hv_SampleIndices, &hv_SampleIndices);
    hv_WindowHandle = HTuple();
    hv_NumSamplesComparison = 10;
    {
        HTuple end_val330 = HTuple((hv_SampleIndices.TupleLength())-1).TupleMin2(hv_NumSamplesComparison-1);
        HTuple step_val330 = 1;
        for (hv_IndexInference=0; hv_IndexInference.Continue(end_val330, step_val330); hv_IndexInference += step_val330)
        {
            hv_Sample = HTuple((hv_DLDataset.TupleGetDictTuple("samples"))[HTuple(hv_SampleIndices[hv_IndexInference])]);
            hv_ImageFile = ((hv_DLDataset.TupleGetDictTuple("image_dir"))+"/")+(hv_Sample.TupleGetDictTuple("image_file_name"));
            ReadImage(&ho_Image, hv_ImageFile);
            ApplyDeepOcr(ho_Image, hv_DeepOcrHandle, HTuple(), &hv_DeepOcrResult);
            if (0 != (int((hv_WindowHandle.TupleLength())==0)))
            {
                dev_open_window_fit_image(ho_Image, 0, 0, -1, -1, &hv_WindowHandle);
            }
            dev_display_deep_ocr_results(ho_Image, hv_WindowHandle, hv_DeepOcrResult, HTuple(),
                                         HTuple());
            // stop(...); only in hdevelop
        }
    }
    //
    if (HDevWindowStack::IsOpen())
        CloseWindow(HDevWindowStack::Pop());
    //
    //
    //***   6.) REMOVE FILES   ***
    clean_up_output(hv_OutputDir, hv_RemoveOutputs);
    //

    hv_xxx = (180/3.14)*1.58;
}




int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    return a.exec();
}

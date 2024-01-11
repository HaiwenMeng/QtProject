#include "halxAI.h"

void HDevAI:: train_dl_model (HTuple hv_DLDataset, HTuple hv_DLModelHandle, HTuple hv_TrainParam,
                              HTuple hv_StartEpoch, HTuple *hv_TrainResults, HTuple *hv_TrainInfos, HTuple *hv_EvaluationInfos)
{

    // Local iconic variables

    // Local control variables
    HTuple  hv_ModelType, hv_DLSamples, hv_TrainSampleIndices;
    HTuple  hv_NumTrainSamples, hv_EvaluationComparisonKeyExist;
    HTuple  hv_EvaluationComparisonKeys, hv_EvaluationOptimizationMethod;
    HTuple  hv_NumEpochs, hv_SeedRand, hv_SampleIndicesTrainRaw;
    HTuple  hv_Index, hv_Shuffled, hv_SampleSeedsTrainRaw, hv_BatchSize;
    HTuple  hv_EvaluateBeforeTraining, hv_ChangeStrategyData;
    HTuple  hv_SerializationData, hv_DisplayData, hv_DisplayEnabled;
    HTuple  hv_DisplayPreviewInitialized, hv_DisplayEvaluationEpochs;
    HTuple  hv_DisplayValidationEvaluationValues, hv_DisplayTrainEvaluationValues;
    HTuple  hv_DisplayLossEpochs, hv_DisplayLoss, hv_DisplayLearningRates;
    HTuple  hv_TrainResultsRestored, hv_StartTime, hv_ThresholdInformation;
    HTuple  hv_FirstIteration, hv_Epoch, hv_Iteration, hv_NumIterationsPerEpoch;
    HTuple  hv_BatchSizeDevice, hv_BatchSizeMultiplier, hv_BatchSizeModel;
    HTuple  hv_NumIterations, hv_SampleIndicesTrain, hv_IterationEvaluateOnly;
    HTuple  hv_BatchStart, hv_BatchEnd, hv_BatchIndices, hv_DLSampleBatch;
    HTuple  hv_AugmentationParam, hv_TrainResult, hv_EvaluationIntervalEpochs;
    HTuple  hv_EvaluationInterval, hv_ValidationEvaluationResult;
    HTuple  hv_TrainEvaluationResult, hv_DisplayParam, hv_SelectPercentageTrainSamples;
    HTuple  hv_EvaluationParam, hv__, hv_TrainEvaluationRatio;
    HTuple  hv_NumTrainEvaluationSampleIndices, hv_TrainEvaluationSampleIndices;
    HTuple  hv_Exception, hv_EvaluationInfo, hv_Valuevalidation;
    HTuple  hv_ValueTrain, hv_TrainInfoUpdateIntervalSeconds;
    HTuple  hv_LastUpdate, hv_Seconds, hv_NumSamplesMeanLoss;
    HTuple  hv_TrainInfo, hv_UpdateTime, hv_EpochsStatus, hv_MeanLoss;
    HTuple  hv_DisplayLearningRate, hv_NumImages, hv_UpdateImagesIntervalEpochs;
    HTuple  hv_UpdateImagesInterval, hv_WindowImages, hv_FirstCall;
    HTuple  hv_GenParamTiled, hv_TrainParamAnomaly, hv_WindowHandleInfo;
    HTuple  hv___Tmp_Ctrl_Dict_Init_0, hv___Tmp_Ctrl_0;

    //StartEpoch is always 0.

    //The procedure returns three dictionaries:
    //- TrainResults: Collected results returned by train_dl_model_batch of every iteration.
    //                For models of type 'anomaly_detection': The final error and the final epoch.
    //- TrainInfo: Collected information of the training progress. This dictionary is empty
    //             for models of type 'anomaly_detection'.
    //- EvaluationInfos: Evaluation results collected during training. This dictionary is empty
    //                   for models of type 'anomaly_detection'.
    //
    //Get the model type.
    GetDlModelParam(hv_DLModelHandle, "type", &hv_ModelType);
    if (0 != (HTuple(HTuple(HTuple(HTuple(HTuple(HTuple(HTuple(int(hv_ModelType!=HTuple("anomaly_detection"))).TupleAnd(int(hv_ModelType!=HTuple("classification")))).TupleAnd(int(hv_ModelType!=HTuple("detection")))).TupleAnd(int(hv_ModelType!=HTuple("gc_anomaly_detection")))).TupleAnd(int(hv_ModelType!=HTuple("ocr_detection")))).TupleAnd(int(hv_ModelType!=HTuple("ocr_recognition")))).TupleAnd(int(hv_ModelType!=HTuple("segmentation")))).TupleAnd(int(hv_ModelType!=HTuple("3d_gripping_point_detection")))))
    {
        throw HException(("Current model type is not supported: \""+hv_ModelType)+"\"");
    }
    //
    //Get the samples for training.
    GetDictTuple(hv_DLDataset, "samples", &hv_DLSamples);
    find_dl_samples(hv_DLSamples, "split", "train", "match", &hv_TrainSampleIndices);
    hv_NumTrainSamples = hv_TrainSampleIndices.TupleLength();
    //
    //Check inconsistent training parameters.
    check_train_dl_model_params(hv_DLDataset, hv_DLModelHandle, hv_NumTrainSamples,
                                hv_StartEpoch, hv_TrainParam);
    //
    //Determine evaluation optimization method.
    GetDictParam(hv_TrainParam, "key_exists", "evaluation_comparison_keys", &hv_EvaluationComparisonKeyExist);
    if (0 != hv_EvaluationComparisonKeyExist)
    {
        GetDictTuple(hv_TrainParam, "evaluation_comparison_keys", &hv_EvaluationComparisonKeys);
        get_dl_evaluation_optimization_method(hv_EvaluationComparisonKeys, &hv_EvaluationOptimizationMethod);
    }
    //
    if (0 != (int(hv_ModelType!=HTuple("anomaly_detection"))))
    {
        //
        //Check if training is required.
        GetDictTuple(hv_TrainParam, "num_epochs", &hv_NumEpochs);
        if (0 != (hv_StartEpoch.TupleIsNumber()))
        {
            if (0 != (int(hv_StartEpoch>=hv_NumEpochs)))
            {
                //Nothing to do.
                return;
            }
        }
        //
        //Set random seed according to parameter value.
        GetDictTuple(hv_TrainParam, "seed_rand", &hv_SeedRand);
        if (0 != (int((hv_SeedRand.TupleLength())>0)))
        {
            //Note, that setting this random seed will not enforce every training to
            //result in the exact same model because the cuDNN library uses approximate
            //algorithms on some architectures.
            //If you want to enforce bit-wise reproducibility, you should also set:
            //   'set_system('cudnn_deterministic', 'true')'
            //However, this can slow down computations on some architectures.
            SetSystem("seed_rand", hv_SeedRand);
        }
        //
        //Generate a random sample index for the whole training independent of batch size.
        hv_SampleIndicesTrainRaw = HTuple();
        {
            HTuple end_val63 = (hv_NumEpochs.TupleCeil())-1;
            HTuple step_val63 = 1;
            for (hv_Index=0; hv_Index.Continue(end_val63, step_val63); hv_Index += step_val63)
            {
                tuple_shuffle(hv_TrainSampleIndices, &hv_Shuffled);
                hv_SampleIndicesTrainRaw = hv_SampleIndicesTrainRaw.TupleConcat(hv_Shuffled);
            }
        }
        //
        //Generate a random seed pool for the whole training independent of batch size.
        hv_SampleSeedsTrainRaw = HTuple(((HTuple(2).TuplePow(31))-1)*HTuple::TupleRand(hv_SampleIndicesTrainRaw.TupleLength())).TupleInt();
        //
        //Initialize the variables for the training.
        //
        //Initialize the batch size with an invalid value so that
        //the while loop will initialize all values directly.
        hv_BatchSize = -1;
        //Initialize iteration overhead parameter to 0 or 1.
        //0: if no evaluation before training is performed
        //1: if evaluation before training is performed
        CreateDict(&hv___Tmp_Ctrl_Dict_Init_0);
        SetDictTuple(hv___Tmp_Ctrl_Dict_Init_0, "comp", "true");
        hv_EvaluateBeforeTraining = (hv_TrainParam.TupleConcat(hv___Tmp_Ctrl_Dict_Init_0)).TupleTestEqualDictItem("evaluate_before_train","comp");
        hv___Tmp_Ctrl_Dict_Init_0 = HTuple::TupleConstant("HNULL");
        //
        //Initialize change strategies.
        init_train_dl_model_change_strategies(hv_TrainParam, &hv_ChangeStrategyData);
        //
        //Initialize serialization strategies.
        init_train_dl_model_serialization_strategies(hv_TrainParam, &hv_SerializationData);
        //
        //Initialize visualizations if enabled.
        dev_display_init_train_dl_model(hv_DLModelHandle, hv_TrainParam, &hv_DisplayData);
        GetDictTuple(hv_DisplayData, "enabled", &hv_DisplayEnabled);
        hv_DisplayPreviewInitialized = 0;
        //
        //Initialize parameters to start new or resume previous training.
        restore_dl_train_info_for_resuming(hv_StartEpoch, hv_SerializationData, hv_TrainParam,
                                           hv_DisplayData, &(*hv_EvaluationInfos), &(*hv_TrainInfos), &hv_DisplayEvaluationEpochs,
                                           &hv_DisplayValidationEvaluationValues, &hv_DisplayTrainEvaluationValues,
                                           &hv_DisplayLossEpochs, &hv_DisplayLoss, &hv_DisplayLearningRates, &hv_TrainResultsRestored,
                                           &hv_StartEpoch);
        //
        //Start time for measurement of elapsed training time.
        CountSeconds(&hv_StartTime);
        //
        //In case of a 'gc_anomaly_detection' model it is necessary to normalize
        //the model outputs before training.
        if (0 != (int(hv_ModelType==HTuple("gc_anomaly_detection"))))
        {
            if (0 != hv_DisplayEnabled)
            {
                hv_ThresholdInformation.Clear();
                hv_ThresholdInformation[0] = "Preparing the model for training";
                hv_ThresholdInformation[1] = "by analyzing image statistics...";
                if (HDevWindowStack::IsOpen())
                    ClearWindow(HDevWindowStack::GetActive());
                if (HDevWindowStack::IsOpen())
                    DispText(HDevWindowStack::GetActive(),hv_ThresholdInformation, "window",
                             "top", "left", "black", "box", "false");
                CountSeconds(&hv___Tmp_Ctrl_0);
                SetDictTuple(hv_DisplayData, "last_update", hv___Tmp_Ctrl_0);
            }
            normalize_dl_gc_anomaly_features(hv_DLDataset, hv_DLModelHandle, HTuple());
        }
        //
        //The while loop needs to know if it is the very first iteration.
        hv_FirstIteration = 1;
        while (true)
        {
            //Do some initializations only for the very first iteration.
            if (0 != hv_FirstIteration)
            {
                //Jump to StartEpoch (Default: 0 but it could be used to resume training at given StartIteration).
                hv_Epoch = hv_StartEpoch;
            }
            else
            {
                hv_Epoch = (hv_Iteration+1)/(hv_NumIterationsPerEpoch.TupleReal());
            }
            //
            //Update any parameters based on strategies.
            update_train_dl_model_change_strategies(hv_DLModelHandle, hv_ChangeStrategyData,
                                                    hv_Epoch);
            //
            //Check if the current batch size and total model batch size differ.
            GetDlModelParam(hv_DLModelHandle, "batch_size", &hv_BatchSizeDevice);
            GetDlModelParam(hv_DLModelHandle, "batch_size_multiplier", &hv_BatchSizeMultiplier);
            hv_BatchSizeModel = hv_BatchSizeDevice*hv_BatchSizeMultiplier;
            //
            if (0 != (HTuple(int(hv_BatchSize!=hv_BatchSizeModel)).TupleOr(hv_FirstIteration)))
            {
                //Set the current value.
                hv_BatchSize = hv_BatchSizeModel;
                //Now, we compute all values which are related to the batch size of the model.
                //That way, the batch_size can be changed during the training without issues.
                //All inputs/outputs/visualizations are based on epochs.
                //
                //Calculate total number of iterations.
                hv_NumIterationsPerEpoch = ((hv_NumTrainSamples/(hv_BatchSize.TupleReal())).TupleFloor()).TupleInt();
                hv_NumIterations = (hv_NumIterationsPerEpoch*hv_NumEpochs).TupleInt();
                //Select those indices that fit into the batch size.
                hv_SampleIndicesTrain = hv_SampleIndicesTrainRaw.TupleSelectRange(0,(hv_NumIterations*hv_BatchSize)-1);
                //The TrainResults tuple will be updated every iteration.
                //Hence, we initialize it as a constant tuple for speedup.
                //It is based on the iterations and hence cannot be reused if the batch size changes.
                TupleGenConst(hv_NumIterations, -1, &(*hv_TrainResults));
                if (0 != (hv_FirstIteration.TupleNot()))
                {
                    hv_Iteration = (((hv_Epoch.TupleReal())*hv_NumIterationsPerEpoch).TupleFloor()).TupleInt();
                    hv_Epoch = (hv_Iteration+1)/(hv_NumIterationsPerEpoch.TupleReal());
                }
            }
            //
            //In the first iteration do some initializations.
            if (0 != hv_FirstIteration)
            {
                //Jump to StartEpoch (Default: 0 but it could be used to resume training at given StartIteration).
                hv_Iteration = (((hv_StartEpoch.TupleReal())*hv_NumIterationsPerEpoch).TupleFloor()).TupleInt();
                hv_FirstIteration = 0;
                if (0 != (int(((hv_Iteration*hv_BatchSize)+hv_BatchSize)>(hv_SampleIndicesTrain.TupleLength()))))
                {
                    hv_Iteration = hv_NumIterations-1;
                    break;
                }
                if (0 != (HTuple(int(hv_StartEpoch>0.0)).TupleAnd(int((hv_TrainResultsRestored.TupleLength())>0))))
                {
                    //Overwrite the first train results.
                    if (0 != (int((hv_TrainResultsRestored.TupleLength())>hv_Iteration)))
                    {
                        hv_TrainResultsRestored = hv_TrainResultsRestored.TupleSelectRange((hv_TrainResultsRestored.TupleLength())-hv_Iteration,(hv_TrainResultsRestored.TupleLength())-1);
                    }
                    (*hv_TrainResults)[HTuple::TupleGenSequence(hv_Iteration-(hv_TrainResultsRestored.TupleLength()),hv_Iteration-1,1)] = hv_TrainResultsRestored;
                }
                //
                //Add an iteration before starting the training for the evaluation if specified.
                hv_IterationEvaluateOnly = hv_Iteration-1;
                hv_Iteration = hv_Iteration-hv_EvaluateBeforeTraining;
            }
            if (0 != (int(hv_Iteration>hv_IterationEvaluateOnly)))
            {
                //
                //Generate the sample batch indices.
                hv_BatchStart = hv_Iteration*hv_BatchSize;
                hv_BatchEnd = (hv_BatchStart+hv_BatchSize)-1;
                hv_BatchIndices = hv_SampleIndicesTrain.TupleSelectRange(hv_BatchStart,hv_BatchEnd);
                //
                //Set a random seed for the sample batch.
                SetSystem("seed_rand", HTuple(hv_SampleSeedsTrainRaw[hv_BatchEnd]));
                //
                //Read preprocessed samples.
                read_dl_samples(hv_DLDataset, hv_BatchIndices, &hv_DLSampleBatch);
                //
                //Augment samples based on train parameter.
                GetDictTuple(hv_TrainParam, "augmentation_param", &hv_AugmentationParam);
                augment_dl_samples(hv_DLSampleBatch, hv_AugmentationParam);
                //
                //Train the model on current batch.
                TrainDlModelBatch(hv_DLModelHandle, hv_DLSampleBatch, &hv_TrainResult);
                //
                //We store each train result.
                (*hv_TrainResults)[hv_Iteration] = hv_TrainResult;
            }
            //
            //Evaluation handling.
            GetDictTuple(hv_TrainParam, "evaluation_interval_epochs", &hv_EvaluationIntervalEpochs);
            hv_EvaluationInterval = ((hv_EvaluationIntervalEpochs*hv_NumIterationsPerEpoch).TupleFloor()).TupleInt();
            hv_ValidationEvaluationResult = HTuple();
            hv_TrainEvaluationResult = HTuple();
            GetDictTuple(hv_DisplayData, "display_param", &hv_DisplayParam);
            //Get percentage of evaluated training samples from display parameters.
            GetDictTuple(hv_DisplayParam, "selected_percentage_train_samples", &hv_SelectPercentageTrainSamples);
            //
            //Evaluate the current model.
            if (0 != (int(hv_EvaluationInterval>0)))
            {
                //Evaluate the model at given intervals.
                if (0 != (HTuple(HTuple(HTuple(int(hv_EvaluationInterval==1)).TupleOr(HTuple(int((hv_Iteration%hv_EvaluationInterval)==0)).TupleAnd(int(hv_Iteration!=0)))).TupleOr(int(hv_Iteration==(hv_NumIterations-1)))).TupleOr(int(hv_Iteration==hv_IterationEvaluateOnly))))
                {
                    GetDictTuple(hv_TrainParam, "evaluation_param", &hv_EvaluationParam);
                    //Evaluate on validation split.
                    evaluate_dl_model(hv_DLDataset, hv_DLModelHandle, "split", "validation",
                                      hv_EvaluationParam, &hv_ValidationEvaluationResult, &hv__);
                    //Evaluate a subset of the train split.
                    hv_TrainEvaluationRatio = hv_SelectPercentageTrainSamples/100.0;
                    hv_NumTrainEvaluationSampleIndices = (hv_TrainEvaluationRatio*(hv_TrainSampleIndices.TupleLength())).TupleInt();
                    if (0 != (int(hv_NumTrainEvaluationSampleIndices>0)))
                    {
                        tuple_shuffle(hv_TrainSampleIndices, &hv_TrainEvaluationSampleIndices);
                        //It might happen that the subset is too small for evaluation.
                        try
                        {
                            evaluate_dl_model(hv_DLDataset, hv_DLModelHandle, "sample_indices",
                                              hv_TrainEvaluationSampleIndices.TupleSelectRange(0,hv_NumTrainEvaluationSampleIndices-1),
                                              hv_EvaluationParam, &hv_TrainEvaluationResult, &hv__);
                        }
                        // catch (Exception)
                        catch (HException &HDevExpDefaultException)
                        {
                            HDevExpDefaultException.ToHTuple(&hv_Exception);
                        }
                    }
                    CreateDict(&hv_EvaluationInfo);
                    SetDictTuple(hv_EvaluationInfo, "epoch", hv_Epoch);
                    SetDictTuple(hv_EvaluationInfo, "iteration", hv_Iteration+hv_EvaluateBeforeTraining);
                    SetDictTuple(hv_EvaluationInfo, "result", hv_ValidationEvaluationResult);
                    SetDictTuple(hv_EvaluationInfo, "result_train", hv_TrainEvaluationResult);
                    (*hv_EvaluationInfos) = (*hv_EvaluationInfos).TupleConcat(hv_EvaluationInfo);
                    if (0 != hv_DisplayEnabled)
                    {
                        GetDictTuple(hv_TrainParam, "evaluation_comparison_keys", &hv_EvaluationComparisonKeys);
                        reduce_dl_evaluation_result(hv_ValidationEvaluationResult, hv_EvaluationComparisonKeys,
                                                    &hv_Valuevalidation, &hv__);
                        reduce_dl_evaluation_result(hv_TrainEvaluationResult, hv_EvaluationComparisonKeys,
                                                    &hv_ValueTrain, &hv__);
                        hv_DisplayValidationEvaluationValues = hv_DisplayValidationEvaluationValues.TupleConcat(hv_Valuevalidation);
                        hv_DisplayTrainEvaluationValues = hv_DisplayTrainEvaluationValues.TupleConcat(hv_ValueTrain);
                        hv_DisplayEvaluationEpochs = hv_DisplayEvaluationEpochs.TupleConcat(hv_Epoch);
                    }
                }
            }
            if (0 != (int(hv_Iteration>hv_IterationEvaluateOnly)))
            {
                //
                //Check if an update is needed.
                GetDictTuple(hv_TrainParam, "update_interval_seconds", &hv_TrainInfoUpdateIntervalSeconds);
                GetDictTuple(hv_DisplayData, "last_update", &hv_LastUpdate);
                CountSeconds(&hv_Seconds);
                //Check for next update (enough time has elapsed or last iteration).
                if (0 != (HTuple(HTuple(int(((hv_LastUpdate-hv_Seconds).TupleAbs())>hv_TrainInfoUpdateIntervalSeconds)).TupleOr(int(hv_Iteration==(hv_NumIterations-1)))).TupleOr(int((hv_ValidationEvaluationResult.TupleLength())>0))))
                {
                    SetDictTuple(hv_DisplayData, "last_update", hv_Seconds);
                    GetDictTuple(hv_TrainParam, "evaluation_comparison_keys", &hv_EvaluationComparisonKeys);
                    GetDictTuple(hv_TrainParam, "num_samples_mean_loss", &hv_NumSamplesMeanLoss);
                    collect_train_dl_model_info(hv_DLModelHandle, (*hv_TrainResults), (*hv_EvaluationInfos),
                                                hv_EvaluationComparisonKeys, hv_EvaluationOptimizationMethod, hv_Iteration,
                                                hv_NumIterations, hv_NumIterationsPerEpoch, hv_NumSamplesMeanLoss,
                                                &hv_TrainInfo);

                    SetDictTuple(hv_TrainInfo, "start_epoch", hv_StartEpoch);
                    SetDictTuple(hv_TrainInfo, "start_time", hv_StartTime);
                    CountSeconds(&hv_UpdateTime);
                    SetDictTuple(hv_TrainInfo, "time_elapsed", hv_UpdateTime-hv_StartTime);
                    (*hv_TrainInfos) = (*hv_TrainInfos).TupleConcat(hv_TrainInfo);
                    //
                    //Display handling.
                    if (0 != hv_DisplayEnabled)
                    {
                        GetDictTuple(hv_TrainInfo, "epoch", &hv_EpochsStatus);
                        hv_DisplayLossEpochs = hv_DisplayLossEpochs.TupleConcat(hv_EpochsStatus);
                        GetDictTuple(hv_TrainInfo, "mean_loss", &hv_MeanLoss);
                        hv_DisplayLoss = hv_DisplayLoss.TupleConcat(hv_MeanLoss);
                        GetDlModelParam(hv_DLModelHandle, "learning_rate", &hv_DisplayLearningRate);
                        hv_DisplayLearningRates = hv_DisplayLearningRates.TupleConcat(hv_DisplayLearningRate);
                        dev_display_update_train_dl_model(hv_TrainParam, hv_DisplayData, hv_TrainInfo,
                                                          hv_DisplayLossEpochs, hv_DisplayLoss, hv_DisplayLearningRates, hv_DisplayEvaluationEpochs,
                                                          hv_DisplayValidationEvaluationValues, hv_DisplayTrainEvaluationValues);
                    }
                }
                //
                //Image result preview handling.
                if (0 != hv_DisplayEnabled)
                {
                    //Show interim results for test images.
                    //For models of type 'gc_anomaly_detection' this is not possible.
                    GetDictTuple(hv_DisplayParam, "num_images", &hv_NumImages);
                    if (0 != (int(hv_NumImages>0)))
                    {
                        //Check if the image preview has to be updated.
                        GetDictTuple(hv_DisplayParam, "update_images_interval_epochs", &hv_UpdateImagesIntervalEpochs);
                        hv_UpdateImagesInterval = (((hv_UpdateImagesIntervalEpochs.TupleReal())*hv_NumIterationsPerEpoch).TupleFloor()).TupleInt();
                        if (0 != (int(hv_UpdateImagesInterval==0)))
                        {
                            hv_UpdateImagesInterval = 1;
                        }
                        if (0 != (HTuple(int((hv_Iteration%hv_UpdateImagesInterval)==0)).TupleOr(hv_DisplayPreviewInitialized.TupleNot())))
                        {
                            GetDictTuple(hv_DisplayData, "window_images", &hv_WindowImages);
                            hv_FirstCall = int((hv_WindowImages.TupleLength())==0);
                            GetDictTuple(hv_DisplayParam, "tiled_param", &hv_GenParamTiled);
                            //
                            dev_display_dl_data_tiled(hv_DLDataset, hv_DLModelHandle, hv_NumImages,
                                                      "validation", hv_GenParamTiled, hv_WindowImages, &hv_WindowImages);
                            //
                            if (0 != hv_FirstCall)
                            {
                                SetDictTuple(hv_DisplayData, "window_images", hv_WindowImages);
                                set_display_font(hv_WindowImages, 12, "mono", "true", "false");
                            }
                            dev_display_tiled_legend(hv_WindowImages, hv_GenParamTiled);
                            hv_DisplayPreviewInitialized = 1;
                        }
                    }
                }
                //
                //Serialization handling.
                update_train_dl_model_serialization(hv_TrainParam, hv_SerializationData,
                                                    hv_Iteration, hv_NumIterations, hv_Epoch, hv_ValidationEvaluationResult,
                                                    hv_EvaluationOptimizationMethod, hv_DLModelHandle, (*hv_TrainInfos),
                                                    (*hv_EvaluationInfos));
            }
            //
            //Check for end of training.
            if (0 != (int(hv_Iteration>=(hv_NumIterations-1))))
            {
                break;
            }
            if (0 != (int(hv_Iteration==hv_IterationEvaluateOnly)))
            {
                hv_EvaluateBeforeTraining = 0;
            }
            //
            //Continue with next iteration.
            hv_Iteration += 1;
        }
        //
    }
    else
    {
        //Case for models of type 'anomaly_detection'.
        //
        //Read the training samples.
        read_dl_samples(hv_DLDataset, hv_TrainSampleIndices, &hv_DLSamples);
        //
        //Get training parameters for anomaly detection.
        GetDictTuple(hv_TrainParam, "anomaly_param", &hv_TrainParamAnomaly);
        //
        //Display information about training.
        GetDictTuple(hv_TrainParam, "display_param", &hv_DisplayParam);
        GetDictTuple(hv_DisplayParam, "enabled", &hv_DisplayEnabled);
        if (0 != hv_DisplayEnabled)
        {
            dev_display_train_info_anomaly_detection(hv_TrainParam, &hv_WindowHandleInfo);
        }
        //
        //Train the model.
        TrainDlModelAnomalyDataset(hv_DLModelHandle, hv_DLSamples, hv_TrainParamAnomaly,
                                   &(*hv_TrainResults));
        //
        //Initialize TrainInfos and EvaluationInfos
        (*hv_TrainInfos) = HTuple();
        (*hv_EvaluationInfos) = HTuple();
        //
        //Close window with information about the training.
        if (0 != hv_DisplayEnabled)
        {
            HDevWindowStack::SetActive(hv_WindowHandleInfo);
        }
    }
    //
    return;
}

void HDevAI:: tuple_shuffle (HTuple hv_Tuple, HTuple *hv_Shuffled)
{

    // Local iconic variables

    // Local control variables
    HTuple  hv_ShuffleIndices;

    //This procedure sorts the input tuple randomly.
    //
    if (0 != (int((hv_Tuple.TupleLength())>0)))
    {
        //Create a tuple of random numbers,
        //sort this tuple, and return the indices
        //of this sorted tuple.
        hv_ShuffleIndices = HTuple::TupleRand(hv_Tuple.TupleLength()).TupleSortIndex();
        //Assign the elements of Tuple
        //to these random positions.
        (*hv_Shuffled) = HTuple(hv_Tuple[hv_ShuffleIndices]);
    }
    else
    {
        //If the input tuple is empty,
        //an empty tuple should be returned.
        (*hv_Shuffled) = HTuple();
    }
    return;
}
void HDevAI:: find_dl_samples (HTuple hv_Samples, HTuple hv_KeyName, HTuple hv_KeyValue, HTuple hv_Mode,
                               HTuple *hv_SampleIndices)
{
    // Local control variables
    HTuple  hv_NumKeyValues, hv_NumFound, hv_SampleIndex;
    HTuple  hv_Sample, hv_KeyExists, hv_Tuple, hv_Hit, hv_ValueIndex;
    HTuple  hv_Value;

    //Check input parameters.
    if (0 != (int((hv_KeyName.TupleLength())!=1)))
    {
        throw HException(HTuple("Invalid KeyName size: ")+(hv_KeyName.TupleLength()));
    }
    if (0 != (int((hv_Mode.TupleLength())!=1)))
    {
        throw HException(HTuple("Invalid Mode size: ")+(hv_Mode.TupleLength()));
    }
    if (0 != (HTuple(HTuple(int(hv_Mode!=HTuple("match"))).TupleAnd(int(hv_Mode!=HTuple("or")))).TupleAnd(int(hv_Mode!=HTuple("contain")))))
    {
        throw HException("Invalid Mode value: "+hv_Mode);
    }
    hv_NumKeyValues = hv_KeyValue.TupleLength();
    if (0 != (HTuple(int(hv_Mode==HTuple("contain"))).TupleAnd(int(hv_NumKeyValues<1))))
    {
        throw HException("Invalid KeyValue size for contain Mode: "+hv_NumKeyValues);
    }
    //
    //Find the indices.
    (*hv_SampleIndices) = HTuple(hv_Samples.TupleLength(),0);
    hv_NumFound = 0;
    //
    {
        HTuple end_val24 = (hv_Samples.TupleLength())-1;
        HTuple step_val24 = 1;
        for (hv_SampleIndex=0; hv_SampleIndex.Continue(end_val24, step_val24); hv_SampleIndex += step_val24)
        {
            hv_Sample = HTuple(hv_Samples[hv_SampleIndex]);
            GetDictParam(hv_Sample, "key_exists", hv_KeyName, &hv_KeyExists);
            if (0 != hv_KeyExists)
            {
                GetDictTuple(hv_Sample, hv_KeyName, &hv_Tuple);
                if (0 != (int(hv_Mode==HTuple("match"))))
                {
                    //Mode 'match': Tuple must be equal KeyValue.
                    hv_Hit = int(hv_Tuple==hv_KeyValue);
                }
                else if (0 != (HTuple(int(hv_Mode==HTuple("or"))).TupleAnd(int((hv_Tuple.TupleLength())==1))))
                {
                    //Mode 'or': Tuple must have only 1 element and it has to be equal to any of KeyValues elements.
                    hv_Hit = int((hv_KeyValue.TupleFindFirst(hv_Tuple))>=0);
                }
                else if (0 != (int(hv_Mode==HTuple("contain"))))
                {
                    //Mode 'contain': Tuple must contain any of the elements in KeyValue.
                    {
                        HTuple end_val37 = hv_NumKeyValues-1;
                        HTuple step_val37 = 1;
                        for (hv_ValueIndex=0; hv_ValueIndex.Continue(end_val37, step_val37); hv_ValueIndex += step_val37)
                        {
                            hv_Value = HTuple(hv_KeyValue[hv_ValueIndex]);
                            hv_Hit = int((hv_Tuple.TupleFindFirst(hv_Value))>=0);
                            if (0 != hv_Hit)
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {
                    //Unsupported configuration.
                    hv_Hit = 0;
                }
                if (0 != hv_Hit)
                {
                    (*hv_SampleIndices)[hv_NumFound] = hv_SampleIndex;
                    hv_NumFound += 1;
                }
            }
        }
    }
    (*hv_SampleIndices) = (*hv_SampleIndices).TupleSelectRange(0,hv_NumFound-1);
    return;
}
